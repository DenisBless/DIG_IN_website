"""Draw the opening scenes of the method film into assets/method_intro.svg.

The scenes share Figure 1's canvas (394.346 x 184.583 pt), colours and Computer Modern
type, so the film can cut from them straight into the figure. Text and formulas are set
with matplotlib's mathtext and stored as outlines, so the page needs no fonts or math
library for them. Parts the page animates carry a name (data-k) and optionally a delay
in seconds (data-delay) that is added to the start of their cue in index.html.

    python tools/make_method_intro.py            # writes assets/method_intro.svg
    python tools/make_method_intro.py --preview DIR   # one finished frame per scene
"""
import math
import os
import sys

import matplotlib
import numpy as np
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath, text_to_path

rcParams['mathtext.fontset'] = 'cm'
TTF = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
RM = FontProperties(fname=os.path.join(TTF, 'cmr10.ttf'))
BF = FontProperties(fname=os.path.join(TTF, 'cmb10.ttf'))

W, H = 394.346, 184.583
# Figure 1's palette
SYM = {'B': '#4477aa', 'G': '#228855', 'O': '#ee9933'}
GREY, EDGE, INK, MUTED = '#c8c8c8', '#333333', '#000000', '#4d4d4d'
ARROW, BASE = '#737373', '#999999'
PANEL, PANEL_EDGE = '#f6f8fa', '#abc2d9'


def num(v):
    return f'{v:.2f}'.rstrip('0').rstrip('.')


def outline(p, dx, dy):
    out = []
    for verts, code in p.iter_segments(simplify=False, curves=True):
        xy = ' '.join(f'{num(dx + x)} {num(dy - y)}' for x, y in verts.reshape(-1, 2))
        out.append({Path.MOVETO: 'M', Path.LINETO: 'L', Path.CURVE3: 'Q', Path.CURVE4: 'C'}.get(code, 'Z') + (xy if code != Path.CLOSEPOLY else ''))
    return ''.join(out)


# every glyph is outlined once, in <defs>, and placed with <use>
GLYPHS = {}


def text(s, x, y, size=8, font=RM, anchor='start', fill=INK):
    """Text or mathtext with its baseline at y."""
    e = TextPath((0, 0), s, size=size, prop=font).get_extents()
    x -= {'start': e.x0, 'middle': (e.x0 + e.x1) / 2, 'end': e.x1}[anchor]
    if '$' in s:
        info, glyph_map, rects = text_to_path.get_glyphs_mathtext(font, s)
    else:
        info, glyph_map, rects = text_to_path.get_glyphs_with_font(text_to_path._get_font(font), s)
    k = size / text_to_path.FONT_SCALE
    out = []
    for gid, gx, gy, scale in info:
        if gid not in GLYPHS:
            GLYPHS[gid] = (f'mi-g{len(GLYPHS)}', outline(Path(*glyph_map[gid]), 0, 0))
        out.append(f'<use href="#{GLYPHS[gid][0]}" transform="translate({num(x + gx * k)} {num(y - gy * k)}) scale({scale * k:.5g})"/>')
    for verts, codes in rects:
        out.append(f'<path d="{outline(Path(np.asarray(verts) * k, codes), x, y)}"/>')
    return f'<g fill="{fill}">' + ''.join(out) + '</g>'


def g(k=None, *kids, delay=None, **attrs):
    a = f' data-k="{k}"' if k else ''
    if delay:
        a += f' data-delay="{num(delay)}"'
    a += ''.join(f' {n.replace("_", "-")}="{v}"' for n, v in attrs.items())
    return f'<g{a}>\n' + '\n'.join(kids) + '\n</g>'


def rect(x, y, w, h, fill='none', stroke=None, sw=0.6, rx=0.6, opacity=None):
    s = f' stroke="{stroke}" stroke-width="{num(sw)}"' if stroke else ''
    o = f' fill-opacity="{opacity}"' if opacity is not None else ''
    return f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" rx="{num(rx)}" fill="{fill}"{o}{s}/>'


def line(x0, y0, x1, y1, stroke=ARROW, sw=0.8, cap='butt'):
    return f'<path fill="none" stroke="{stroke}" stroke-width="{num(sw)}" stroke-linecap="{cap}" d="M{num(x0)} {num(y0)}L{num(x1)} {num(y1)}"/>'


def head(x, y, ang, color=ARROW, size=3.2):
    pts = [(x, y)] + [(x - size * math.cos(ang + s * 0.42), y - size * math.sin(ang + s * 0.42)) for s in (-1, 1)]
    return f'<path fill="{color}" d="M' + 'L'.join(f'{num(a)} {num(b)}' for a, b in pts) + 'Z"/>'


def arrow(x0, y0, x1, y1, color=ARROW, sw=0.8):
    ang = math.atan2(y1 - y0, x1 - x0)
    return line(x0, y0, x1 - 2.4 * math.cos(ang), y1 - 2.4 * math.sin(ang), color, sw) + head(x1, y1, ang, color)


def curve(x0, y0, cx, cy, x1, y1, color=ARROW, sw=0.8):
    ang = math.atan2(y1 - cy, x1 - cx)
    ex, ey = x1 - 2.4 * math.cos(ang), y1 - 2.4 * math.sin(ang)
    return (f'<path fill="none" stroke="{color}" stroke-width="{num(sw)}" d="M{num(x0)} {num(y0)}Q{num(cx)} {num(cy)} {num(ex)} {num(ey)}"/>'
            + head(x1, y1, ang, color))


def heading(s):
    return g('head', text(s, 12, 16, size=10, font=BF))


def swap(x, y, s, before, after, delay):
    """A site that changes look: the new one pops in over a copy of the old one."""
    return g('swap', g('under', rect(x, y, s, s, before), opacity='0'), g('top', rect(x, y, s, s, after)), delay=delay)


# ------------------------------------------------------------------ scene a: the target
def scene_target():
    grid = ['BBBGGGG',
            'BBBBGGG',
            'BBOBGGO',
            'BBOOOGG',
            'GBOOOOG',
            'BBOOOGG',
            'BOOOGGG']
    s, gap, x0, y0 = 10, 4.4, 24, 30
    p = s + gap
    sites = [g('a-site', rect(x0 + c * p, y0 + r * p, s, s, SYM[v]), delay=(r + c) * 0.035)
             for r, row in enumerate(grid) for c, v in enumerate(row)]
    bonds = []
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if c + 1 < len(row) and row[c + 1] == v:
                y = y0 + r * p + s / 2
                bonds.append(line(x0 + c * p + s + 0.4, y, x0 + (c + 1) * p - 0.4, y, EDGE, 1.4))
            if r + 1 < len(grid) and grid[r + 1][c] == v:
                x = x0 + c * p + s / 2
                bonds.append(line(x, y0 + r * p + s + 0.4, x, y0 + (r + 1) * p - 0.4, EDGE, 1.4))
    cx = x0 + (7 * p - gap) / 2
    X = 150
    return g(None,
             heading('The target'),
             *sites,
             g('a-bonds', *bonds),
             g('a-lab', text(r'a configuration $x \in \mathcal{A}^D$', cx, y0 + 7 * p + 9, 8, anchor='middle'),
               text(r'$D = 49$ sites, $|\mathcal{A}| = 3$ symbols', cx, y0 + 7 * p + 20, 7.5, anchor='middle', fill=MUTED)),
             g('a-pi', text(r'$\pi(x) = \frac{e^{-U(x)}}{Z}$', X, 56, 15)),
             g('a-u', text(r'$U(x)$: the energy of $x$, cheap to evaluate', X, 86, 8.5),
               text('lower energy means higher probability', X, 97, 7.5, fill=MUTED)),
             g('a-potts', text(r'e.g. Potts: $U(x) = -\beta {\sum}_{\langle d, d^\prime \rangle} \mathbf{1}\{x_d = x_{d^\prime}\}$', X, 121, 8.5),
               text(r'every bond between equal neighbours lowers $U$ by $\beta$', X, 135, 7.5, fill=MUTED)),
             g('a-z', text(r'$Z = {\sum}_{x}\, e^{-U(x)}$ runs over all $|\mathcal{A}|^D$ configurations', X, 158, 8.5),
               text('intractable, and there are no samples to learn from', X, 170, 7.5, fill=MUTED)),
             data_scene='a')


# ------------------------------------------------------------------ scene b: the denoiser
def scene_denoiser():
    final = 'BGOBGO'
    s, gap, x0, y0 = 12, 2, 22, 84
    p = s + gap
    sx = lambda i: x0 + i * p
    # (site, start of its round in seconds after the first round); the last three are quick
    rounds = [(2, 0), (0, 2.4), (4, 4.8)]
    quick = [(3, 5.6), (5, 5.9), (1, 6.2)]
    probs = [(0.30, 0.25, 0.45), (0.55, 0.25, 0.20), (0.20, 0.55, 0.25)]
    sites = [swap(sx(i), y0, s, GREY, SYM[final[i]], d) for i, d in sorted(rounds + quick)]
    hls = [g('b-hl', rect(sx(i) - 1.4, y0 - 1.4, s + 2.8, s + 2.8, stroke=EDGE, sw=1.1, rx=1.2), delay=d) for i, d in rounds]
    bx, by, bw, bh = 122, y0 - 12, 48, 36
    net = g('b-net',
            arrow(sx(5) + s + 4, y0 + s / 2, bx - 3, y0 + s / 2),
            rect(bx, by, bw, bh, PANEL, PANEL_EDGE, rx=4),
            text(r'$\mu_\theta$', bx + bw / 2, by + bh / 2 + 1.5, 14, anchor='middle'),
            text('denoiser', bx + bw / 2, by + bh - 6, 7.5, anchor='middle', fill=MUTED),
            arrow(bx + bw + 3, y0 + s / 2, bx + bw + 15, y0 + s / 2))
    base, bar_w, bar_x = y0 + 28, 9, bx + bw + 20
    bars = []
    for r, ((i, d), pr) in enumerate(zip(rounds, probs)):
        chosen = 'BGO'.index(final[i])
        kids = [line(bar_x - 3, base, bar_x + 3 * bar_w + 2 * 3 + 3, base, BASE, 0.6)]
        for j, (v, q) in enumerate(zip('BGO', pr)):
            x, h = bar_x + j * (bar_w + 3), q * 52
            kids.append(g('b-bar', rect(x, base - h, bar_w, h, SYM[v], rx=0.3,
                                        stroke=EDGE if j == chosen else None, sw=0.8), delay=d + 0.06 * j))
        top = base - pr[chosen] * 52
        cxb = bar_x + chosen * (bar_w + 3) + bar_w / 2
        kids.append(g('b-curve', curve(cxb, top - 2, (cxb + sx(i) + s / 2) / 2, y0 - 46, sx(i) + s / 2, y0 - 3), delay=d))
        bars.append(g('b-bars', *kids, delay=d))
    mid = x0 + (6 * p - gap) / 2
    X = 236
    return g(None,
             heading('The denoiser'),
             g('b-row', *sites),
             g('b-rowlab', text('grey sites are hidden,', mid, y0 + s + 13, 7.5, anchor='middle', fill=MUTED),
               text(r'$X_R$ are the revealed ones', mid, y0 + s + 24, 7.5, anchor='middle', fill=MUTED)),
             net, *hls, *bars,
             g('b-barlab', text(r'$\mu^d_\theta(\cdot \mid X_R)$', bar_x + 1.5 * bar_w + 3, base + 13, 8.5, anchor='middle')),
             g('b-f1', text(r'$\mu^d_\theta(c \mid X_R)$', X, 48, 12)),
             g('b-f1n', text(r'probability that hidden site $d$ takes symbol $c$,', X, 62, 7.5, fill=MUTED),
               text(r'given the revealed sites $X_R$', X, 72, 7.5, fill=MUTED)),
             g('b-f2', text(r'$q_\theta(X) = {\prod}_{\mathrm{reveals}}\, \mu^d_\theta(X_d \mid X_R)$', X, 100, 10)),
             g('b-f2n', text(r'the probability of generating $X$', X, 117, 7.5, fill=MUTED)),
             g('b-f3', text(r'goal: $\mu^d_\theta(c \mid x_R) = P_\pi(X_d = c \mid X_R = x_R)$', X, 146, 8.5)),
             g('b-f3n', text("the target's own conditional distribution", X, 158, 7.5, fill=MUTED)),
             data_scene='b')


# ------------------------------------------------------------------ scene c: Gibbs labels
def width(s, size):
    return TextPath((0, 0), s, size=size, prop=RM).get_extents().width


def scene_gibbs():
    grid = ['GBBBG',
            'BBBOG',
            'BBOGG',
            'OOOGG',
            'OOBGG']
    s, gap, x0, y0 = 10, 1.5, 18, 32
    p = s + gap
    d = (2, 2)
    near = {(1, 2), (2, 1), (2, 3), (3, 2)}
    cells = [rect(x0 + c * p, y0 + r * p, s, s, SYM[v]) for r, row in enumerate(grid) for c, v in enumerate(row)]
    dim = [rect(x0 + c * p - 0.3, y0 + r * p - 0.3, s + 0.6, s + 0.6, '#ffffff', rx=0.6, opacity=0.72)
           for r in range(5) for c in range(5) if (r, c) != d and (r, c) not in near]
    rings = [rect(x0 + c * p - 1, y0 + r * p - 1, s + 2, s + 2, stroke=EDGE, sw=0.6, rx=1) for r, c in near]
    rings.append(rect(x0 + d[1] * p - 1.4, y0 + d[0] * p - 1.4, s + 2.8, s + 2.8, stroke=EDGE, sw=1.4, rx=1.2))
    cx = x0 + (5 * p - gap) / 2
    # replacing x_d by c changes U by -beta * (equal neighbours of c - equal neighbours of x_d); beta = 1
    nb = [grid[r][c] for r, c in sorted(near)]
    cur = grid[d[0]][d[1]]
    dU = {c: nb.count(cur) - nb.count(c) for c in 'BGO'}
    Z = sum(math.exp(-v) for v in dU.values())
    h = {c: math.exp(-dU[c]) / Z for c in 'BGO'}
    ps, pg = 7.5, 1.1
    pp = ps + pg
    plus, dus = [], []
    for j, c in enumerate('BGO'):
        px, py = 108, 48 + 28 * j
        cross = {(0, 0): c, (-1, 0): grid[1][2], (0, -1): grid[2][1], (0, 1): grid[2][3], (1, 0): grid[3][2]}
        plus.append(g('c-plus', *[rect(px + dc * pp - ps / 2, py + dr * pp - ps / 2, ps, ps, SYM[v])
                                  for (dr, dc), v in cross.items()],
                      rect(px - ps / 2 - 0.9, py - ps / 2 - 0.9, ps + 1.8, ps + 1.8, stroke=EDGE, sw=1.1, rx=1), delay=0.35 * j))
        dus.append(g('c-du', text(rf'$\nabla_{{d,c}}U(x) = {dU[c]}$', 124, py + 3, 8.5), delay=0.3 * j))
    bar_x, base, bw = 218, 108, 10
    bars, vals = [], []
    for j, c in enumerate('BGO'):
        x, hh = bar_x + j * (bw + 4), h[c] * 76
        bars.append(g('c-bar', rect(x, base - hh, bw, hh, SYM[c], rx=0.3), delay=0.08 * j))
        vals.append(text(f'{h[c]:.2f}', x + bw / 2, base - hh - 2.5, 6.5, anchor='middle', fill=MUTED))
    # the label formula runs along the bottom, in two parts so the second can follow the first
    f1 = r'$h^d(c \mid x^{-d}) = \dfrac{\exp[-\nabla_{d,c}U(x)]}{\sum_{a \in \mathcal{A}} \exp[-\nabla_{d,a}U(x)]}$'
    f2 = r'$= P_\pi(X_d = c \mid X^{-d} = x^{-d})$'
    fs, fgap = 9, 3.5
    fx = W / 2 - (width(f1, fs) + fgap + width(f2, fs)) / 2
    X = 282
    return g(None,
             heading('Gibbs labels: a local score'),
             g('c-grid', *cells),
             g('c-focus', *dim, *rings),
             g('c-lab', text(r'configuration $x$, site $d$', cx, y0 + 5 * p + 9, 8, anchor='middle'),
               text('and its four neighbours', cx, y0 + 5 * p + 19, 7.5, anchor='middle', fill=MUTED)),
             g('c-rep', text(r'replace $x_d$ by each $c \in \mathcal{A}$', 140, 29, 8, anchor='middle')),
             *plus, *dus,
             g('c-def', text(r'$\nabla_{d,c}U(x) = U(x^{d \leftarrow c}) - U(x)$', X, 44, 8),
               text(r'Potts with $\beta = 1$: every extra equal', X, 56, 7, fill=MUTED),
               text(r'neighbour lowers $U$ by 1', X, 65, 7, fill=MUTED)),
             g('c-arrow', arrow(186, 76, 208, 76), data_dir='r'),
             g('c-norm', text('normalize', 197, 70, 7, anchor='middle', fill=MUTED),
               text(r'$e^{-\nabla_{d,c}U}$', 197, 87, 7.5, anchor='middle', fill=MUTED)),
             g('c-base', line(bar_x - 3, base, bar_x + 3 * bw + 8 + 3, base, BASE, 0.6)),
             *bars,
             g('c-val', *vals, text(r'$h^d(c \mid x^{-d})$', bar_x + 1.5 * bw + 4, base + 13, 8.5, anchor='middle')),
             g('c-h', text(f1, fx, 156, fs)),
             g('c-p', text(f2, fx + width(f1, fs) + fgap, 156, fs)),
             g('c-z', text('only energy differences enter,', X, 84, 7.5, fill=MUTED),
               text(r'so $Z$ cancels', X, 94, 7.5, fill=MUTED)),
             g('c-pn', text('exactly the conditional that', X, 110, 7.5, fill=MUTED),
               text('Gibbs sampling uses', X, 120, 7.5, fill=MUTED)),
             data_scene='c')


SCENES = [scene_target, scene_denoiser, scene_gibbs]


def svg(body):
    defs = '\n'.join(f'<path id="{i}" d="{d}"/>' for i, d in GLYPHS.values())
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
            f'<defs>\n{defs}\n</defs>\n' + body + '\n</svg>\n')


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scenes = [f() for f in SCENES]
    if len(sys.argv) > 2 and sys.argv[1] == '--preview':
        # finished frames: every part shown, under-layers hidden
        for i, sc in enumerate(scenes):
            with open(os.path.join(sys.argv[2], f'scene_{"abc"[i]}.svg'), 'w', encoding='utf-8') as f:
                f.write(svg(f'<rect width="{W}" height="{H}" fill="#fff"/>\n' + sc))
    else:
        body = '\n'.join(sc.replace('data-scene=', 'opacity="0" data-scene=', 1) for sc in scenes)
        with open(os.path.join(root, 'assets', 'method_intro.svg'), 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n' + svg(body))
