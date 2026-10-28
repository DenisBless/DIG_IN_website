"""assets/bridge.json: the free-energy film (the DIG-IN bridge on ALDP-S and ALDP-T).

    python tools/make_bridge_data.py --runs_root <training outputs> --data_root <FEAT data> \
        --out assets/bridge.json

Needs the `digin_bridge` package of the code repository (free_energy/ on the PYTHONPATH) and the
trained runs <runs_root>/*/<task>_s<seed>/ (args.json, checkpoints and the final evaluation in
final/).  FEAT's ALDP data is downloaded on first use.

Per task, the trained bridge of seed 0 carries held-out MD frames to endpoint 1 and the paths are
recorded every few tau-leap steps: the Ramachandran density of X_t at 17 times, 32 traces in
(phi, psi), and one path in 3D with its jumps, its accumulated kernel log-ratio and the energy of
endpoint 1 along it.  From the final evaluations of the three seeds come the distribution of path
log-weights, the running estimate against the number of paths and the estimates.  The references
are those of the paper: for ALDP-S, BAR on the MD of both states (standard error from a bootstrap
over frames, as in digin_bridge.paper), for ALDP-T our umbrella sampling; FEAT's estimates as
reported by He et al. (2025) are the baseline.  Surfaces are ln p on 60 x 60 cells relative to the
maximum, clipped at -8 and quantized to bytes.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os

import numpy as np
import torch

from digin_bridge import bridge as br
from digin_bridge import data
from digin_bridge import zmatrix as zm
from digin_bridge.train import Trainer, grouped_se, parse

FE_BINS, FE_CAP = 60, 8.0
SEEDS = (0, 1, 2)
NAMES = ["HH31", "CH3", "HH32", "HH33", "C", "O", "N", "H", "CA", "HA", "CB", "HB1", "HB2", "HB3", "C", "O",
         "N", "H", "CH3", "HH31", "HH32", "HH33"]
RES = ["ACE"] * 6 + ["ALA"] * 10 + ["NME"] * 6
REF_UMBRELLA = {"aldpt": (-4.44, 0.02)}
FEAT_REPORTED = {"aldps": (29.38, 0.04), "aldpt": (-4.56, 0.08)}
BAR_BOOTSTRAP = 200


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def surface(phi, psi, w=None) -> str:
    h, _, _ = np.histogram2d(phi, psi, bins=FE_BINS, range=[[-math.pi, math.pi]] * 2, weights=w)
    with np.errstate(divide="ignore"):
        f = np.log(h / h.sum())
    f -= f[np.isfinite(f)].max()
    f = np.where(np.isfinite(f), np.clip(f, -FE_CAP, 0.0), -FE_CAP - 1)
    q = np.where(f < -FE_CAP, 0, np.round(1 + (f + FE_CAP) / FE_CAP * 254)).astype(np.uint8)
    return b64(q.T.copy())


def find(root, name, sub=""):
    c = [os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", name, sub, "args.json" if not sub else "meta.json"))]
    if not c:
        raise SystemExit(f"no {name}/{sub} under {root}")
    return max(c, key=os.path.getmtime)


def load_trainer(run_dir: str, data_root: str) -> Trainer:
    saved = json.load(open(os.path.join(run_dir, "args.json")))
    a = parse(["--out", run_dir, "--task", saved["task"]])
    vars(a).update({k: v for k, v in saved.items() if k not in ("device", "data_root", "out")})
    a.data_root, a.no_wandb = data_root, True
    t = Trainer(a)
    if not t.load():
        raise SystemExit(f"no checkpoint in {run_dir}")
    return t


@torch.no_grad()
def record(t: Trainer, n: int, every: int, chunk: int, seed: int) -> dict:
    """n paths of the EMA bridge from held-out frames, the state every `every` steps (ctmc_sample with
    the bookkeeping), their log-weights and, per frame, the jumps and the kernel log-ratio so far."""
    tg, net, K, g = t.tg, t.ema, t.a.ctmc_steps, t.a.gibbs_rate
    torch.manual_seed(seed)
    j = torch.randperm(len(t.test_b0), device=t.device)[:n]
    out = {"frames": [], "jumps": [], "lr": [], "log_w": [], "ends": [], "start": j}
    for s in range(0, n, chunk):
        x = t.test_b0[j[s:s + chunk]].long()
        lp0 = t.test_lp0[j[s:s + chunk]]
        m, dt = x.shape[0], 1.0 / K
        lr = torch.zeros(m, dtype=torch.float64, device=t.device)
        jumps = torch.zeros(m, device=t.device)
        frames, fj, flr = [x.to(torch.uint8)], [jumps.clone()], [lr.clone()]
        prev = None
        for k in range(K + 1):
            q, phi = br._heads(net, x, k / K)
            here = x.unsqueeze(-1)
            phi_x = phi.gather(-1, here)
            if prev is not None:
                Rb = (q * ((phi_x - phi).clamp_min(0.0) + g)).scatter(-1, here, 0.0)
                lr += br._step_logprob(Rb, x, prev, dt)
                if k % every == 0:
                    frames.append(x.to(torch.uint8)); fj.append(jumps.clone()); flr.append(lr.clone())
            if k == K:
                break
            R = (q * ((phi - phi_x).clamp_min(0.0) + g)).scatter(-1, here, 0.0)
            lam = R.sum(-1)
            move = torch.rand_like(lam) < -torch.expm1(-lam * dt)
            u = torch.rand_like(lam).unsqueeze(-1) * lam.unsqueeze(-1)
            c = (R.cumsum(-1) < u).sum(-1).clamp(max=R.shape[-1] - 1)
            nxt = torch.where(move, c, x)
            lr -= br._step_logprob(R, x, nxt, dt)
            jumps += (nxt != x).float().sum(-1)
            prev, x = x, nxt
        z1 = tg.dequantize(x)
        _, U1 = tg.reduced(z1)
        ends = -U1 - lp0
        out["frames"].append(torch.stack(frames, 1).cpu())                      # (m, F, D)
        out["jumps"].append(torch.stack(fj, 1).cpu()); out["lr"].append(torch.stack(flr, 1).cpu())
        out["log_w"].append((ends + lr).cpu()); out["ends"].append(ends.cpu())
        print(f"  paths {s + m}/{n}", flush=True)
    for k in ("frames", "jumps", "lr", "log_w", "ends"):
        out[k] = torch.cat(out[k])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_paths", type=int, default=16384)
    ap.add_argument("--every", type=int, default=4, help="tau-leap steps between recorded frames")
    ap.add_argument("--chunk", type=int, default=4096)
    a = ap.parse_args(argv)
    root = data.ensure(a.data_root)
    els = [nm[0] for nm in NAMES]
    out = {"fe_cap": FE_CAP, "fe_bins": FE_BINS,
           "molecule": {"atoms": [{"el": e, "name": nm, "res": r} for e, nm, r in zip(els, NAMES, RES)],
                        "bonds": [list(s["atoms"]) for s in zm.SPEC if s["kind"] == "bond"]},
           "tasks": {}}
    for task in ("aldps", "aldpt"):
        t = load_trainer(find(a.runs_root, f"{task}_s0"), root)
        tg = t.tg
        z0, z1, _, _ = data.endpoint_frames(tg, root)
        T = {"md0": surface(z0[:, zm.PHI_INDEX], z0[:, zm.PSI_INDEX]),
             "md1": surface(z1[:, zm.PHI_INDEX], z1[:, zm.PSI_INDEX]),
             "n0": int(len(z0)), "n1": int(len(z1)), "K": int(t.a.ctmc_steps), "every": a.every,
             "phi_a": list(tg.phi_a) if tg.phi_a else None}
        rec = record(t, a.n_paths, a.every, a.chunk, seed=5)
        F = rec["frames"].shape[1]
        cent = tg.centres.cpu()                                                   # (D, L)
        phi_c, psi_c = cent[zm.PHI_INDEX], cent[zm.PSI_INDEX]
        fphi = phi_c[rec["frames"][..., zm.PHI_INDEX].long()].numpy()             # (n, F)
        fpsi = psi_c[rec["frames"][..., zm.PSI_INDEX].long()].numpy()
        tf = np.arange(F) * a.every / T["K"]
        T["t"] = [round(float(v), 5) for v in tf]
        # the marginal of X_t at 17 times, unweighted
        sel = [int(round(i * (F - 1) / 16)) for i in range(17)]
        T["density_t"] = [T["t"][i] for i in sel]
        T["density"] = [surface(fphi[:, i], fpsi[:, i]) for i in sel]
        lw = rec["log_w"].numpy()
        fin = np.isfinite(lw)
        # traces: 32 paths with finite weight, spread over the start frames
        rng = np.random.default_rng(1)
        tr = rng.choice(np.where(fin)[0], 32, replace=False)
        T["traces"] = b64(np.round(np.stack([fphi[tr], fpsi[tr]], -1) * 10000).astype(np.int16))   # (32, F, 2)
        T["trace_lw"] = [round(float(v - lw[fin].max()), 3) for v in lw[tr]]
        # the path shown in 3D: for ALDP-S one that moves from the extended (C7eq/beta) region to
        # alphaR, for ALDP-T any path that reaches S_a; the median weight among them
        ps0, ps1 = np.degrees(fpsi[:, 0]), np.degrees(fpsi[:, -1])
        ph1 = np.degrees(fphi[:, -1])
        if task == "aldps":
            cand = np.where(fin & (ps0 > 100) & (ps1 > -120) & (ps1 < 50) & (ph1 < 0))[0]
        else:
            cand = np.where(fin)[0]
        if len(cand) == 0:
            cand = np.where(fin)[0]
        cand = cand[np.argsort(lw[cand])]
        h = int(cand[len(cand) // 2])
        zf = cent.gather(1, rec["frames"][h].long().T).T.double()                # (F, D) bin centres
        pos = zm.to_cartesian(zf).float().numpy()                                 # (F, 22, 3) nm
        U1 = torch.cat([tg.reduced(zf[i:i + 64].to(t.device))[1].cpu() for i in range(0, F, 64)]).numpy()
        T["hero"] = {"pos": b64(np.round(pos * 1000).astype(np.int16)),
                     "rama": [[round(float(fphi[h, i]), 4), round(float(fpsi[h, i]), 4)] for i in range(F)],
                     "jumps": [int(v) for v in rec["jumps"][h].tolist()],
                     "log_ratio": [round(float(v), 3) for v in rec["lr"][h].tolist()],
                     "U1": [round(float(v), 3) if np.isfinite(v) else None for v in U1],
                     "log_w": round(float(lw[h]), 3), "ends": round(float(rec["ends"][h]), 3),
                     "path": round(float(rec["lr"][h, -1]), 3)}
        # the final evaluations: weights, running estimates, estimates
        est, se, ess, conv = [], [], [], {}
        for seed in SEEDS:
            d = find(a.runs_root, f"{task}_s{seed}", "final")                     # .../<run>/final
            lws = np.load(os.path.join(d, "log_w.npy"))
            start = np.load(os.path.join(d, "start_frame.npy"))
            m = np.max(lws)
            w = np.exp(lws - m)
            est.append(float(m + math.log(w.sum() / len(w))))
            se.append(grouped_se(lws, start))
            ess.append(float(w.sum() ** 2 / (len(w) * (w ** 2).sum())))
            grid = np.unique(np.geomspace(30, len(lws), 80).astype(int))
            perm = np.random.default_rng(seed).permutation(len(lws))
            c = np.cumsum(w[perm])
            conv[f"s{seed}"] = [round(float(m + math.log(c[M - 1] / M)), 4) if c[M - 1] > 0 else None for M in grid]
            conv["M"] = grid.tolist()
            if seed == 0:
                rel = lws[np.isfinite(lws)] - m
                edges = np.linspace(max(rel.min(), -40.0), 0.0, 81)
                hist, _ = np.histogram(np.clip(rel, edges[0], 0), bins=edges)
                T["weights"] = {"hist": hist.tolist(), "lo": float(edges[0]),
                                "n": int(len(lws)), "n_inf": int((~np.isfinite(lws)).sum())}
        if task == "aldps":
            # BAR on the MD of both states, exactly as digin_bridge.paper computes it
            polar = lambda z: tg.parts(z)[1]                                          # noqa: E731
            du0, du1 = (torch.cat([polar(torch.as_tensor(z[i:i + 16384]).to(t.device)).double().cpu()
                                   for i in range(0, len(z), 16384)]).numpy() / tg.kT for z in (z0, z1))
            rng = np.random.default_rng(10)
            boot = [data.bar(rng.choice(du0, len(du0)), -rng.choice(du1, len(du1)), iters=80)
                    for _ in range(BAR_BOOTSTRAP)]
            ref = {"name": "BAR", "value": [round(data.bar(du0, -du1), 4), round(float(np.std(boot, ddof=1)), 4)]}
        else:
            ref = {"name": "umbrella", "value": list(REF_UMBRELLA[task])}
        T.update({"estimate": [round(v, 4) for v in est], "se": [round(v, 4) for v in se],
                  "ess": [round(v, 4) for v in ess], "conv": conv, "ref": ref, "feat": list(FEAT_REPORTED[task])})
        out["tasks"][task] = T
        print(task, "estimate", T["estimate"], "ESS", T["ess"], "hero log w", T["hero"]["log_w"], flush=True)
        del t
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote", a.out, os.path.getsize(a.out) // 1024, "kB", flush=True)


if __name__ == "__main__":
    main()
