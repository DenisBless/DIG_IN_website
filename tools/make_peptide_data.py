"""assets/peptides.json: the discretized Boltzmann-generator film (alanine dipeptide and tetrapeptide).

    python tools/make_peptide_data.py --runs_root <training outputs> --samples <sample sets> \
        --ref_dir <MD references> --out assets/peptides.json

Needs the `digin` package of the code repository (peptides/ on the PYTHONPATH), the trained runs
(<runs_root>/*/<system>_<coords>_s<seed>/ with args.json and checkpoints), their 2M-sample sets
(<samples>/<run>/, digin.sample) and the MD references (<ref_dir>/<system>_ref.npz).

Internal coordinates only (the paper's main text).  For both peptides one draw of the EMA model is
recorded network call by network call: the reveal order, the categorical over the L bins of the
coordinate revealed at each call, the network's most likely bin of every coordinate still masked,
and the molecule after each call (masked coordinates at that guess), plus the Gibbs labels of the
finished draw.  The Ramachandran surfaces are ln p(phi, psi) on 60 x 60 cells relative to the
maximum, clipped at -8 and quantized to bytes: MD, the samples as drawn and reweighted (three seeds
pooled), with a few thousand sample points and their weights.  The gallery holds finished draws
from different basins.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os

import numpy as np
import torch

from digin import metrics
from digin.sample import find_run
from digin.train import Trainer, parse

FE_BINS, FE_CAP = 60, 8.0
SHOWN = [("aldp", "internal"), ("al4", "internal")]
SEEDS = (0, 1, 2)
# basins of a residue: alphaR (phi < 0, -120 < psi < 50), beta/PPII (rest of phi < 0), alphaL (phi >= 0)
BASINS = ("αR", "β", "αL")


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def basin_of(rama: np.ndarray) -> np.ndarray:
    """(..., 2) radians -> (...) 0 alphaR, 1 beta/PPII, 2 alphaL."""
    phi, psi = np.degrees(rama[..., 0]), np.degrees(rama[..., 1])
    ar = (phi < 0) & (psi > -120) & (psi < 50)
    return np.where(phi >= 0, 2, np.where(ar, 0, 1))


def surface(phi, psi, w=None) -> np.ndarray:
    h, _, _ = np.histogram2d(phi, psi, bins=FE_BINS, range=[[-math.pi, math.pi]] * 2, weights=w)
    with np.errstate(divide="ignore"):
        f = np.log(h / h.sum())
    f -= f[np.isfinite(f)].max()
    f = np.where(np.isfinite(f), np.clip(f, -FE_CAP, 0.0), -FE_CAP - 1)
    q = np.where(f < -FE_CAP, 0, np.round(1 + (f + FE_CAP) / FE_CAP * 254)).astype(np.uint8)
    return q.T.copy()            # rows: psi (bottom to top), columns: phi


def load_trainer(run_dir: str) -> Trainer:
    saved = json.load(open(os.path.join(run_dir, "args.json")))
    a = parse(["--system", saved["system"], "--coords", saved["coords"], "--out", run_dir])
    vars(a).update({k: v for k, v in saved.items() if k != "device"})
    a.out, a.label_budget, a.no_wandb = run_dir, saved.get("label_budget", 0), True
    t = Trainer(a)
    if not t.load():
        raise SystemExit(f"no checkpoint in {run_dir}")
    return t


def molecule(tg) -> dict:
    mol = tg.mol
    atoms = [{"el": a["element"], "name": a["name"], "res": a["resname"], "resid": a["resid"]} for a in mol.atoms]
    return {"atoms": atoms, "bonds": [list(b) for b in mol.bonds], "rama_res": mol.rama_res}


def coordinates(tg) -> list[dict]:
    """What every coordinate is: its kind, the atom it places and a short label."""
    mol = tg.mol
    nm = lambda i: mol.atoms[i]["name"].replace("CA", "Cα").replace("CB", "Cβ")
    out = []
    if tg.coords == "internal":
        bb = {}
        for r, (qphi, qpsi) in enumerate(mol.rama_quads.tolist()):
            bb[frozenset(qphi[1:3])] = ("φ", r)
            bb[frozenset(qpsi[1:3])] = ("ψ", r)
        for s in mol.spec:
            a = s["atoms"]
            kind = s["kind"] if s["kind"] != "dihedral" else s["dih"]
            sym = {"bond": "–", "angle": "–", "dihedral": "–"}[s["kind"]]
            label = sym.join(nm(i) for i in a)
            rama = None
            if s["kind"] == "dihedral" and s["dih"] == "torsion" and frozenset(a[1:3]) in bb:
                rama = bb[frozenset(a[1:3])]
            out.append({"kind": kind, "atom": int(a[0]), "label": label,
                        "rama": None if rama is None else {"angle": rama[0], "res": rama[1]}})
    else:
        frame = [("Nx", mol.second), ("Cx", mol.third), ("Cy", mol.third)]
        for lab, i in frame:
            out.append({"kind": "frame", "atom": int(i), "label": f"{nm(i)} {lab[1]}", "rama": None})
        for i, p in tg.tree:
            for ax in "xyz":
                out.append({"kind": "vector", "atom": int(i), "label": f"{nm(p)}→{nm(i)} {ax}", "rama": None})
    return out


@torch.no_grad()
def record(t: Trainer, n: int, seed: int) -> dict:
    """n draws of the EMA model, recorded call by call (any_order_sample with the bookkeeping)."""
    tg, net, dev = t.tg, t.ema, t.device
    D, L = tg.D, tg.L
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.full((n, D), L, dtype=torch.long, device=dev)
    order = torch.argsort(torch.rand(n, D, device=dev, generator=g), dim=1)
    ar = torch.arange(n, device=dev)
    probs = torch.zeros(n, D, L, device=dev)
    guess = torch.zeros(n, D + 1, D, dtype=torch.long, device=dev)
    logq = torch.zeros(n, D, dtype=torch.float64, device=dev)
    lq = torch.zeros(n, dtype=torch.float64, device=dev)
    for k in range(D):
        lg = net(x).float().log_softmax(-1)                                   # (n, D, L)
        guess[:, k] = torch.where(x == L, lg.argmax(-1), x)
        d = order[:, k]
        row = lg[ar, d]
        u = torch.rand(n, 1, device=dev, generator=g)
        tok = (row.exp().cumsum(-1) < u).sum(-1).clamp(max=L - 1)
        probs[:, k] = row.exp()
        lq += row[ar, tok].double()
        logq[:, k] = lq
        x[ar, d] = tok
    guess[:, D] = x
    z = tg.dequantize(x, g)
    lw = tg.log_p(z) - lq
    obs = tg.observables(z)
    return {"x": x, "z": z, "order": order, "probs": probs, "guess": guess, "logq": logq, "log_w": lw,
            "energy": obs["energy"].double(), "rama": obs["rama"].float()}


def pick(rec: dict, want: tuple[int, ...]) -> int:
    """A draw whose residues sit in the wanted basins, with the median weight among them."""
    lw = rec["log_w"].cpu().numpy()
    bas = basin_of(rec["rama"].cpu().numpy())                                 # (n, R)
    ok = np.where((bas == np.array(want)).all(1) & np.isfinite(lw))[0]
    if len(ok) == 0:                                  # fall back to the first residue's basin only
        ok = np.where((bas[:, 0] == want[0]) & np.isfinite(lw))[0]
    ok = ok[np.argsort(lw[ok])]
    return int(ok[len(ok) // 2])


@torch.no_grad()
def trajectory(t: Trainer, rec: dict, i: int) -> dict:
    tg = t.tg
    D, L = tg.D, tg.L
    revealed = torch.zeros(D + 1, D, dtype=torch.bool, device=t.device)
    for k in range(D):
        revealed[k + 1:, rec["order"][i, k]] = True
    cent = tg.centres                                                          # (D, L)
    zg = cent.gather(1, rec["guess"][i].T).T                                   # (D + 1, D) at the guesses
    zk = torch.where(revealed, rec["z"][i].unsqueeze(0), zg)
    pos = tg.to_cartesian(zk).float()                                          # (D + 1, n, 3) nm
    gl = tg.gibbs_labels(rec["z"][i:i + 1])[0].exp()                           # (D, L)
    q8 = lambda p: np.round(255 * (p / p.max(-1, keepdim=True).values.clamp_min(1e-30)).cpu().numpy()).astype(np.uint8)
    lw, E = float(rec["log_w"][i]), float(rec["energy"][i])
    rama = rec["rama"][i].cpu().numpy()
    return {"order": rec["order"][i].tolist(), "bins": rec["x"][i].tolist(),
            "guess": b64(rec["guess"][i].cpu().numpy().astype(np.uint16)),
            "probs": b64(q8(rec["probs"][i])), "gibbs": b64(q8(gl)),
            "pos": b64(np.round(pos.cpu().numpy() * 1000).astype(np.int16)),
            "log_q": [round(float(v), 3) for v in rec["logq"][i]],
            "log_w": round(lw, 3), "energy": round(E, 2),
            "rama": [[round(float(a), 4) for a in r] for r in rama],
            "basin": [BASINS[b] for b in basin_of(rama).tolist()]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--samples", required=True, help="directory of the sample sets <system>_<coords>_s<seed>/")
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_record", type=int, default=512)
    ap.add_argument("--n_points", type=int, default=4000)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    out = {"fe_cap": FE_CAP, "fe_bins": FE_BINS, "systems": {}}
    for system in ("aldp", "al4"):
        r = np.load(os.path.join(a.ref_dir, f"{system}_ref.npz"))
        ref = {"rama": np.asarray(r["rama"], np.float32), "energy": np.asarray(r["energy"], np.float64)}
        R = ref["rama"].shape[1]
        S = {"md": [b64(surface(ref["rama"][:, k, 0], ref["rama"][:, k, 1])) for k in range(R)],
             "n_md": int(len(ref["energy"])), "md_basins": metrics.basin_fractions(ref["rama"]).round(4).tolist(),
             "settings": {}}
        for coords in ("internal",):
            pooled, per = {"rama": [], "w": [], "wn": []}, []
            gallery = None
            for seed in SEEDS:
                d = os.path.join(a.samples, f"{system}_{coords}_s{seed}")
                lw = np.load(os.path.join(d, "log_w.npy")).astype(np.float64)
                rama = np.load(os.path.join(d, "rama.npy"), mmap_mode="r")
                rama = np.asarray(rama, np.float32)
                w = np.exp(lw - lw.max())
                per.append({"ess": float(w.sum() ** 2 / (len(w) * (w ** 2).sum())),
                            "log_Z": float(lw.max() + math.log(w.sum() / len(w))),
                            "js_rw": [metrics.js(metrics.rama_hist(ref["rama"][:, k, 0], ref["rama"][:, k, 1]),
                                                 metrics.rama_hist(rama[:, k, 0], rama[:, k, 1], w / w.sum()))
                                      for k in range(R)]})
                pooled["rama"].append(rama); pooled["w"].append(w / w.sum() / len(SEEDS))
                pooled["wn"].append(np.full(len(w), 1.0 / len(w) / len(SEEDS)))
                if seed == 0:
                    # a few thousand points as drawn, with their self-normalized weights times n
                    j = rng.choice(len(lw), a.n_points, replace=False)
                    wn = w[j] / w.mean()
                    pts = {"rama": b64(np.round(rama[j] * 10000).astype(np.int16)),
                           "w": b64(np.round(np.clip(np.sqrt(wn), 0, 5) * 50).astype(np.uint8))}
                    # finished draws from different basins, positions in pm
                    x = np.load(os.path.join(d, "x.npy"), mmap_mode="r")
                    bas = basin_of(rama[: 200000])
                    wants = [(0,) * R, (1,) * R, (2,) * R] if R == 1 else [(0,) * R, (1,) * R, (1, 0, 0), (0, 0, 1), (1, 1, 0)]
                    gallery = []
                    for want in wants:
                        ok = np.where((bas == np.array(want)).all(1))[0]
                        if len(ok) == 0:
                            continue
                        ok = ok[np.argsort(lw[ok])]
                        q = int(ok[len(ok) // 2])
                        gallery.append({"basin": [BASINS[b] for b in want],
                                        "pos": b64(np.round(np.asarray(x[q]) * 1000).astype(np.int16)),
                                        "rama": [[round(float(v), 4) for v in rr] for rr in rama[q]]})
            rp = np.concatenate(pooled["rama"]); w, wn = np.concatenate(pooled["w"]), np.concatenate(pooled["wn"])
            entry = {"raw": [b64(surface(rp[:, k, 0], rp[:, k, 1], wn)) for k in range(R)],
                     "rw": [b64(surface(rp[:, k, 0], rp[:, k, 1], w)) for k in range(R)],
                     "basins_raw": metrics.basin_fractions(rp, wn).round(4).tolist(),
                     "basins_rw": metrics.basin_fractions(rp, w).round(4).tolist(),
                     "points": pts, "gallery": gallery,
                     "ess": [round(p["ess"], 5) for p in per], "log_Z": [round(p["log_Z"], 4) for p in per],
                     "js_rw": [[round(v, 6) for v in p["js_rw"]] for p in per]}
            if (system, coords) in SHOWN:
                run_dir = find_run(f"{system}_{coords}_s0", a.runs_root)
                t = load_trainer(run_dir)
                rec = record(t, a.n_record, seed=11)
                hero = pick(rec, (1,) * R if system == "aldp" and coords == "internal"
                            else (0,) * R)
                entry.update({"L": t.tg.L, "D": t.tg.D, "coords": coordinates(t.tg), "traj": trajectory(t, rec, hero),
                              "box": {"center": t.tg.center.tolist(), "half": t.tg.half.tolist()}})
                S.setdefault("molecule", molecule(t.tg))
                del t
                torch.cuda.empty_cache()
            S["settings"][coords] = entry
            print(system, coords, "ESS", entry["ess"], "log Z", entry["log_Z"], flush=True)
        out["systems"][system] = S
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote", a.out, os.path.getsize(a.out) // 1024, "kB", flush=True)


if __name__ == "__main__":
    main()
