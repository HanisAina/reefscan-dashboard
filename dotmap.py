"""dotmap.py: an abstract dot-grid picture of a simulated reef (built from reefsim data).

Sand = faint dots, healthy coral = teal, bleached coral = glowing red.
The adaptive robot's lawnmower path is traced as a thin line.
"""
import base64
import numpy as np
import reefsim as rs
import rl_agent as rl


def dotmap_svg(seed=7, sand=0.45, lanes=24, cells=60, gap=14, r=4.2, path=True):
    m = dict(rs.MAP); m["lanes"], m["lane_cells"] = lanes, cells
    reef = rs.make_reef(np.random.default_rng(seed), sand, m)
    W, H = cells * gap + gap, lanes * gap + gap
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
         '<defs><filter id="g" x="-50%" y="-50%" width="200%" height="200%">'
         '<feGaussianBlur stdDeviation="2.4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>']
    if path:
        pts = []
        for i in range(lanes):
            y = gap + i * gap
            xs = (gap, gap + (cells - 1) * gap) if i % 2 == 0 else (gap + (cells - 1) * gap, gap)
            pts += [f"{xs[0]},{y}", f"{xs[1]},{y}"]
        o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#7aa7ff" stroke-width="1" opacity=".22"/>')
    for i in range(lanes):
        for j in range(cells):
            x, y = gap + j * gap, gap + i * gap
            c = reef[i, j]
            if c == rs.SAND:
                o.append(f'<circle cx="{x}" cy="{y}" r="{r * 0.55:.1f}" fill="#8ea2d6" opacity=".18"/>')
            elif c == rs.HEALTHY:
                o.append(f'<circle cx="{x}" cy="{y}" r="{r:.1f}" fill="#2dd4bf" opacity=".75"/>')
            else:
                o.append(f'<circle cx="{x}" cy="{y}" r="{r * 1.1:.1f}" fill="#ff4d6d" filter="url(#g)"/>')
    o.append("</svg>")
    return "".join(o)


def dotmap_uri(**kw):
    return "data:image/svg+xml;base64," + base64.b64encode(dotmap_svg(**kw).encode()).decode("ascii")
