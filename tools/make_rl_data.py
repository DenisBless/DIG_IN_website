"""Write the data of the reinforcement-learning film: assets/rl.json and the image of the
state the policy acts in, assets/rl/state.jpg.

The data is one draw of a trained DIG-IN policy at a state of a recorded greedy episode
(demo.npz and episode.npz, written by tools/record_run.py): a random reveal order, the
categorical over the bins of every dimension at the step it is revealed, the sampled bins
and the executed action; and, for that action, the critic at the bin centres of every
dimension (the others held at the action), the Gibbs labels softmax(Q / tau), and the
policy's conditionals given all other dimensions.  Of the recorded states and dimensions,
the film shows the most peaked label whose mode lies at least EDGE bins inside the action
bounds, so that the sweep of the critic shows a whole bump; the choice depends on the label
alone, not on how well the policy fits it.

    python tools/make_rl_data.py <results> [--task WalkerWalk]

<results>/<Task>/s<seed>/{episode,demo}.npz; the run whose 64 recorded episodes have the
highest mean return is used.  Needs mujoco, mujoco_playground and imageio.
"""
import argparse
import glob
import json
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from mujoco_playground import registry

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EDGE = 8  # bins between the mode of the shown label and either end of the action range


def entropy(p):
    p = np.clip(p, 1e-12, 1)
    return float(-(p * np.log(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--task", default="WalkerWalk")
    ap.add_argument("--size", type=int, default=352)
    args = ap.parse_args()
    runs = []
    for path in glob.glob(os.path.join(args.results, args.task, "s*", "demo.npz")):
        run = os.path.dirname(path)
        ep = np.load(os.path.join(run, "episode.npz"))
        runs.append((float(ep["returns"].mean()), int(os.path.basename(run)[1:]), ep, np.load(path)))
    _, seed, ep, demo = max(runs, key=lambda r: r[0])
    ret = float(ep["rewards"].sum())
    # the recorded state and dimension with the most peaked Gibbs label whose mode is inside
    L = demo["label"].shape[2]
    inside = [(i, d) for i in range(demo["label"].shape[0]) for d in range(demo["label"].shape[1])
              if EDGE <= demo["label"][i, d].argmax() < L - EDGE]
    i, d = min(inside, key=lambda id_: entropy(demo["label"][id_]))
    t = int(demo["times"][i])
    m = demo["probs"].shape[1]
    r4 = lambda a: np.round(np.asarray(a, dtype=float), 4).tolist()
    out = {
        "task": args.task, "seed": seed, "return": round(ret, 1), "t": t, "m": int(m), "L": int(L),
        "temp": float(demo["temp"]), "order": demo["order"][i].tolist(), "z": demo["z"][i].tolist(),
        "action": r4(demo["action"][i]), "probs": r4(demo["probs"][i]), "q": np.round(demo["q"][i], 5).tolist(),
        "label": r4(demo["label"][i]), "cond": r4(demo["cond"][i]), "dlab": int(d),
    }
    with open(os.path.join(ROOT, "assets", "rl.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    # the state, from the first camera of the model, as the episode videos
    model = registry.load(args.task).mj_model
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = ep["qpos"][t], ep["qvel"][t]
    if model.nmocap:
        data.mocap_pos[:], data.mocap_quat[:] = ep["mocap_pos"][t], ep["mocap_quat"][t]
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=args.size, width=args.size) as r:
        r.update_scene(data, camera=0)
        frame = r.render()
    os.makedirs(os.path.join(ROOT, "assets", "rl"), exist_ok=True)
    imageio.imwrite(os.path.join(ROOT, "assets", "rl", "state.jpg"), frame, quality=85)
    print(f"{args.task} seed {seed} (return {ret:.1f}), state at step {t}, label of dimension {d + 1} "
          f"(entropy {entropy(demo['label'][i, d]):.2f} nats), tau {float(demo['temp']):.3g}")


if __name__ == "__main__":
    main()
