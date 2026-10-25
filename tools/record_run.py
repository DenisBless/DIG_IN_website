"""One run of dig_in_rl.train, then a recording of the final policy, for the
videos and the reinforcement-learning film of the project page.  Copy this file into
rl_jax/dig_in_rl/ of the code repository and run, from rl_jax/,

    python -m dig_in_rl.record_run env.name=WalkerWalk seed=7 hydra.run.dir=<out>

Writes to the run directory
  metrics.npz   as dig_in_rl.train
  episode.npz   the best of `rec_envs` greedy episodes of the final policy:
                qpos, qvel, mocap_pos, mocap_quat (the state before every step),
                obs, action, rewards, and the returns of all episodes
  demo.npz      the generative process at a few states of that episode: one
                random order, the categorical of every revealed dimension, the
                sampled bins and action, the critic's Gibbs labels and Q values
                at the bin centres, and the policy's conditionals given all
                other dimensions
  policy.pkl    the parameters of policy and critic and the observation statistics
"""

import logging
import os
import pickle
import time

import hydra
import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from omegaconf import DictConfig, OmegaConf

from dig_in_rl import train as T
from dig_in_rl.envs import DMCEnv
from dig_in_rl.policy import random_schedule, sample_action

log = logging.getLogger(__name__)


def train_state(cfg):
    env = DMCEnv(cfg.env.name, cfg.num_envs, cfg.env.episode_length, cfg.env.normalize_obs)
    eval_env = DMCEnv(cfg.env.name, cfg.num_envs, cfg.env.episode_length, cfg.env.normalize_obs)

    def log_fn(metrics):
        log.info(f"step={int(np.max(metrics['time_step']))} return={float(np.mean(metrics['eval/episode_return'])):.1f}"
                 f" entropy={float(np.mean(metrics['train/entropy'])):.3g} temp={float(np.mean(metrics['train/temp'])):.3g}")

    init, block, num_blocks = T.make_train(cfg, env, eval_env, log_fn)

    @jax.jit
    def run(key):
        ts, keys = init(key)
        return jax.lax.scan(block, ts, keys)

    start = time.perf_counter()
    ts, metrics = run(jax.random.PRNGKey(cfg.seed))
    metrics = jax.device_get(metrics)
    log.info(f"training took {time.perf_counter() - start:.0f} s")
    return ts, metrics


def record(cfg, ts, n, seed):
    env = DMCEnv(cfg.env.name, n, cfg.env.episode_length, cfg.env.normalize_obs)
    actor = nnx.merge(ts.actor.graphdef, ts.actor.params)

    @jax.jit
    def go(key):
        key, k0 = jax.random.split(key)
        obs, st = env.reset(jax.random.split(k0, n), stats=ts.env_state)

        def step(carry, _):
            key, st, obs = carry
            key, k = jax.random.split(key)
            action, *_ = sample_action(actor, k, obs, greedy=True, with_entropy=False)
            d = st.inner.data
            frame = dict(qpos=d.qpos, qvel=d.qvel, mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat,
                         obs=obs, action=action)
            obs, st, reward, _, _ = env.step(st, action, update_stats=False)
            return (key, st, obs), (frame, reward)

        _, (frames, rewards) = jax.lax.scan(step, (key, st, obs), None, length=cfg.env.episode_length)
        return frames, rewards

    frames, rewards = jax.device_get(go(jax.random.PRNGKey(seed)))
    returns = np.asarray(rewards).sum(0)
    best = int(np.argmax(returns))
    ep = {k: np.asarray(v)[:, best] for k, v in frames.items()}
    ep.update(rewards=np.asarray(rewards)[:, best], returns=returns, best=np.asarray(best))
    return ep


def demo(ts, ep, times, seed):
    actor = nnx.merge(ts.actor.graphdef, ts.actor.params)
    critic = nnx.merge(ts.critic.graphdef, ts.critic.params)
    m, L = actor.m, actor.L
    centres = actor.binning.centres
    temp = actor.temperature().squeeze()
    out = {k: [] for k in ["order", "probs", "z", "action", "q", "label", "cond", "marginal"]}
    for i, t in enumerate(times):
        obs = jnp.asarray(ep["obs"][t])[None]
        k_ord, k_tok, k_deq = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), i), 3)
        order, step_idx = random_schedule(k_ord, 1, m)
        z = jnp.zeros((1, m), dtype=jnp.int32)
        probs = []
        for k in range(m):
            d = order[0, k]
            lp = jax.nn.log_softmax(actor.logits(actor.trunk(obs, z, step_idx < k))[0, d])
            zk = jax.random.categorical(jax.random.fold_in(k_tok, k), lp)
            z = z.at[0, d].set(zk)
            probs.append(jnp.exp(lp))
        action = actor.binning.decode(z, k_deq)[0]
        # the critic at the bin centres of every dimension, the others at the action
        a_lab = jnp.broadcast_to(action, (m, L, m))
        a_lab = jnp.where(jnp.eye(m, dtype=bool)[:, None, :], centres[None, :, None], a_lab)
        q = critic.q_slice(jnp.repeat(obs, m, axis=0), a_lab)  # (m, L)
        # the policy's conditional of d given all other dimensions, and its marginals at the start
        rev = ~jnp.eye(m, dtype=bool)
        cond = jax.nn.softmax(actor.logits(actor.trunk(jnp.repeat(obs, m, axis=0), jnp.repeat(z, m, axis=0), rev)), -1)
        cond = cond[jnp.arange(m), jnp.arange(m)]
        marg = jax.nn.softmax(actor.logits(actor.trunk(obs, jnp.zeros_like(z), jnp.zeros((1, m), bool)))[0], -1)
        for k, v in dict(order=order[0], probs=jnp.stack(probs), z=z[0], action=action, q=q,
                         label=jax.nn.softmax(q / temp, -1), cond=cond, marginal=marg).items():
            out[k].append(np.asarray(v))
    res = {k: np.stack(v) for k, v in out.items()}
    res.update(times=np.asarray(times), temp=np.asarray(temp), centres=np.asarray(centres))
    return res


@hydra.main(version_base=None, config_path="../configs", config_name="dmc")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    ts, metrics = train_state(cfg)
    np.savez("metrics.npz", **{k: np.asarray(v) for k, v in metrics.items()})
    log.info(f"final return {float(np.mean(metrics['eval/episode_return'][-1])):.1f}")
    ts = jax.tree.map(lambda x: x[0], ts)  # the one seed
    ep = record(cfg, ts, n=int(cfg.get("rec_envs", 64)), seed=cfg.seed + 1000)
    np.savez_compressed("episode.npz", **ep)
    log.info(f"recorded: best return {float(ep['returns'][int(ep['best'])]):.1f}, "
             f"mean {float(ep['returns'].mean()):.1f} over {ep['returns'].size} greedy episodes")
    try:
        np.savez_compressed("demo.npz", **demo(ts, ep, times=[n * cfg.env.episode_length // 10 for n in (1, 3, 6)], seed=cfg.seed + 2000))
    except Exception as e:  # the recording above is what matters
        log.warning(f"demo failed: {e!r}")
    try:
        state = {"actor": jax.device_get(nnx.state(nnx.merge(ts.actor.graphdef, ts.actor.params))),
                 "critic": jax.device_get(nnx.state(nnx.merge(ts.critic.graphdef, ts.critic.params))),
                 "obs_mean": np.asarray(ts.env_state.obs_mean), "obs_var": np.asarray(ts.env_state.obs_var),
                 "obs_count": np.asarray(ts.env_state.obs_count), "cfg": OmegaConf.to_container(cfg)}
        with open("policy.pkl", "wb") as f:
            pickle.dump(state, f)
    except Exception as e:
        log.warning(f"saving the parameters failed: {e!r}")
    log.info(f"outputs in {os.getcwd()}")


if __name__ == "__main__":
    main()
