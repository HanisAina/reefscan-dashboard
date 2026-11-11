"""
knowledge_graph.py: a "map of connections" between bleaching detections.

Plain-language summary
  * Every place where the vehicle reported bleaching becomes a node.
  * Two nodes are linked if they are close together on the seabed
    (within LINK_RADIUS_M metres).
  * Degree centrality (Equation 5 in the interim report):
        C_D(v) = deg(v) / (N - 1)
    says how connected a detection is. A detection with several close
    neighbours is part of a bleaching patch; an isolated one is more likely
    to be a false alarm.
  * Groups of linked nodes (connected components) are bleaching hotspots.
  * A simple rule-check layer marks detections with fewer than MIN_DEGREE
    neighbours as "unconfirmed". This is the "double-checking" layer
    promised in the interim report (Gap 3), and it is evaluated against the
    true reef here.
"""
import numpy as np
import networkx as nx
import reefsim as rs

LINK_RADIUS_M = 2.3   # links neighbours along a lane (1 m), across lanes (2 m) and diagonally (2.24 m)
MIN_DEGREE = 2        # at least this many linked neighbours to count as "confirmed"


def build_graph(reef, rows, cols, seen, radius=LINK_RADIUS_M, min_degree=MIN_DEGREE):
    """Nodes = cells reported as bleached. Positions in metres."""
    G = nx.Graph()
    idx = np.where(seen == rs.BLEACHED)[0]
    lw, cl = rs.MAP["lane_width_m"], rs.MAP["cell_len_m"]
    pts = {}
    for order, i in enumerate(idx):
        r, c = int(rows[i]), int(cols[i])
        x, y = c * cl, r * lw
        G.add_node(int(i), lane=r, cell=c, x=x, y=y, visit_order=order,
                   truly_bleached=bool(reef[r, c] == rs.BLEACHED), true_class=int(reef[r, c]))
        pts[int(i)] = (x, y)
    # link close detections (grid hashing keeps this fast)
    cell = radius
    buckets = {}
    for n, (x, y) in pts.items():
        buckets.setdefault((int(x // cell), int(y // cell)), []).append(n)
    for (bx, by), members in buckets.items():
        near = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                near += buckets.get((bx + dx, by + dy), [])
        for a in members:
            xa, ya = pts[a]
            for b in near:
                if b <= a:
                    continue
                xb, yb = pts[b]
                d = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
                if d <= radius + 1e-9:
                    G.add_edge(a, b, distance_m=round(d, 2))
    N = G.number_of_nodes()
    for n in G.nodes:
        G.nodes[n]["degree"] = G.degree[n]
        G.nodes[n]["centrality"] = G.degree[n] / (N - 1) if N > 1 else 0.0
        G.nodes[n]["confirmed"] = G.degree[n] >= min_degree
    return G


def hotspots(G):
    """Connected groups of detections, largest first."""
    out = []
    for k, comp in enumerate(sorted(nx.connected_components(G), key=len, reverse=True)):
        xs = [G.nodes[n]["x"] for n in comp]; ys = [G.nodes[n]["y"] for n in comp]
        out.append(dict(hotspot=k + 1, detections=len(comp),
                        centre_x_m=round(float(np.mean(xs)), 1), centre_y_m=round(float(np.mean(ys)), 1),
                        lanes=f"{min(G.nodes[n]['lane'] for n in comp)}-{max(G.nodes[n]['lane'] for n in comp)}",
                        mean_degree=round(float(np.mean([G.degree[n] for n in comp])), 2),
                        share_truly_bleached=round(float(np.mean([G.nodes[n]['truly_bleached'] for n in comp])), 3)))
    return out


def true_patches(reef):
    """Groups of truly bleached cells (8-neighbour connectivity), for evaluation."""
    lanes, n = reef.shape
    lab = -np.ones(reef.shape, dtype=int)
    k = 0
    for r in range(lanes):
        for c in range(n):
            if reef[r, c] == rs.BLEACHED and lab[r, c] < 0:
                stack = [(r, c)]; lab[r, c] = k
                while stack:
                    a, b = stack.pop()
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            aa, bb = a + da, b + db
                            if 0 <= aa < lanes and 0 <= bb < n and reef[aa, bb] == rs.BLEACHED and lab[aa, bb] < 0:
                                lab[aa, bb] = k; stack.append((aa, bb))
                k += 1
    return lab, k


def evaluate(reef, G):
    """How well does the rule-check layer separate real bleaching from false alarms?"""
    nodes = list(G.nodes(data=True))
    if not nodes:
        return {}
    conf = [d for _, d in nodes if d["confirmed"]]
    unconf = [d for _, d in nodes if not d["confirmed"]]
    prec = lambda L: float(np.mean([d["truly_bleached"] for d in L])) if L else float("nan")
    false_alarms = [d for _, d in nodes if not d["truly_bleached"]]
    lab, n_patches = true_patches(reef)
    found = set()
    for _, d in nodes:
        l = lab[d["lane"], d["cell"]]
        if l >= 0:
            found.add(l)
    sizes = np.bincount(lab[lab >= 0].ravel()) if n_patches else np.array([])
    big = set(np.where(sizes >= 5)[0]) if n_patches else set()
    return dict(
        detections=len(nodes),
        precision_all=prec([d for _, d in nodes]),
        precision_confirmed=prec(conf),
        precision_unconfirmed=prec(unconf),
        share_confirmed=len(conf) / len(nodes),
        false_alarms=len(false_alarms),
        false_alarms_flagged=float(np.mean([not d["confirmed"] for d in false_alarms])) if false_alarms else float("nan"),
        true_detections_kept=float(np.mean([d["confirmed"] for _, d in nodes if d["truly_bleached"]])),
        true_patches=int(n_patches),
        patches_found_pct=100 * len(found) / n_patches if n_patches else float("nan"),
        patches_5plus=len(big),
        patches_5plus_found_pct=100 * len(found & big) / len(big) if big else float("nan"),
        hotspots_all=nx.number_connected_components(G),
        hotspots_confirmed=nx.number_connected_components(G.subgraph([n for n, d in nodes if d["confirmed"]])),
    )


def to_pyvis_html(G, path, max_nodes=600):
    """Interactive network picture (needs `pip install pyvis`)."""
    from pyvis.network import Network
    net = Network(height="560px", width="100%", bgcolor="#ffffff", font_color="#222222")
    net.toggle_physics(False)
    nodes = sorted(G.nodes, key=lambda n: -G.degree[n])[:max_nodes]
    keep = set(nodes)
    for n in nodes:
        d = G.nodes[n]
        color = "#d1495b" if d["confirmed"] else "#edae49"
        title = (f"Lane {d['lane']}, {d['x']:.0f} m along lane<br>Linked neighbours: {d['degree']}"
                 f"<br>Centrality: {d['centrality']:.4f}<br>{'Confirmed' if d['confirmed'] else 'Unconfirmed (isolated)'}"
                 f"<br>Truth: {rs.CLASS_NAMES[d['true_class']]}")
        net.add_node(int(n), label=" ", title=title, x=d["x"] * 12, y=d["y"] * 12, color=color,
                     size=4 + 2 * d["degree"], physics=False)
    for a, b in G.edges:
        if a in keep and b in keep:
            net.add_edge(int(a), int(b), color="#999999")
    net.write_html(path, notebook=False)
    return path
