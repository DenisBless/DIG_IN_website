# DIG-IN project page

Project page for *Fixed-point neural samplers on discrete spaces*
([arXiv 2610.01739](https://arxiv.org/pdf/2610.01739)), which introduces DIG-IN, the
Discrete Gibbs Iterative Neural Sampler.

The page is a single static `index.html`. The two animations are drawn on a canvas
from the data in `assets/`:

- `assets/alloy.json`: phase-diagram boundaries for Ag–Pd and Cu–Au, the fcc site
  positions, and one recorded sampling trajectory per highlighted phase.
- `assets/lattice.json`: the 9 × 9 (β, h) grids of Ising (50 × 50) and Potts
  (q = 3, 24 × 24) samples, and recorded trajectories at three temperatures.
- `assets/fig_algorithm.svg`: Figure 1 of the paper.

Each trajectory is a draw from a trained checkpoint, recorded network call by network
call: the final configuration plus the order in which its sites were revealed.

## Running locally

The page loads its data with `fetch`, so serve the folder over http instead of
opening the file directly:

```
python -m http.server 8000
```

and open http://localhost:8000/.

## Code

The code that reproduces the paper's experiments is at
https://github.com/jiajunhe98/DIG-IN-Fixed-point-neural-samplers-on-discrete-spaces.
