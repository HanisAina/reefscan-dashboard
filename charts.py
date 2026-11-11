"""charts.py: interactive Plotly charts for the dashboard."""
import numpy as np
import plotly.graph_objects as go
import reefsim as rs
import style as S

FONT = "Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"
CLASS_TXT = {rs.SAND: "Sand/rock", rs.HEALTHY: "Healthy coral", rs.BLEACHED: "Bleached coral"}
MODE_TXT = {0: "Fast", 1: "Normal", 2: "Detail"}


def base(fig, height=380, title=None, legend=True):
    fig.update_layout(
        height=height, template="plotly_white", font=dict(family=FONT, size=13, color=S.INK),
        margin=dict(l=10, r=10, t=50 if title else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, size=12), bordercolor=S.LINE),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=16, color=S.INK), x=0, xanchor="left",
                                     y=0.98, yref="container", yanchor="top"))
        if legend:   # leave room for the title above the legend
            fig.update_layout(margin=dict(l=10, r=10, t=96, b=10),
                              legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=S.LINE)
    fig.update_yaxes(gridcolor="#f0f0f2", zeroline=False, linecolor=S.LINE)
    return fig


def _discrete(colors):
    """Colour scale for whole numbers 0..n-1."""
    n = len(colors); out = []
    for i, c in enumerate(colors):
        out += [[i / n, c], [(i + 1) / n, c]]
    return out


def reef_heatmap(reef, rows, cols, seen, log, title="Hover over any metre of the reef"):
    """True reef map; hover shows what the robot saw and did in that metre."""
    lanes, n = reef.shape
    saw = np.full(reef.shape, -1); mode = np.full(reef.shape, -1); ph = np.zeros(reef.shape, dtype=int)
    saw[rows, cols] = seen; mode[rows, cols] = log["modes"]; ph[rows, cols] = log["photos"]
    custom = [[[CLASS_TXT[int(saw[r, c])], MODE_TXT[int(mode[r, c])], int(ph[r, c]),
                "correct" if saw[r, c] == reef[r, c] else "MISTAKE"] for c in range(n)] for r in range(lanes)]
    fig = go.Figure(go.Heatmap(
        z=reef, x=np.arange(n), y=np.arange(lanes), zmin=-0.5, zmax=2.5,
        colorscale=_discrete([S.SAND, S.HEALTHY, "#ffffff"]), showscale=False, customdata=custom,
        xgap=0.5, ygap=0.5,
        hovertemplate=("<b>Lane %{y}, %{x} m</b><br>Truth: %{text}<br>Robot saw: %{customdata[0]} "
                       "(%{customdata[3]})<br>Mode: %{customdata[1]}<br>Photos: %{customdata[2]}<extra></extra>"),
        text=[[CLASS_TXT[int(v)] for v in row] for row in reef]))
    fig.update_yaxes(autorange=False, range=[lanes - 0.5, -0.5], title="Lane", showgrid=False)
    fig.update_xaxes(autorange=False, range=[-0.5, n - 0.5], title="Distance along lane (m)")
    fig.update_layout(plot_bgcolor="#d1d1d6")
    return base(fig, 380, title, legend=False)


def playback(reef, rows, cols, modes, title="Press Play to watch the survey", step_lanes=2):
    """Animated mode map: the robot sweeps lane by lane."""
    lanes, n = reef.shape
    grid_all = np.full(reef.shape, np.nan)
    grid_all[rows, cols] = modes
    cs = _discrete([S.MODE_COLORS["fast"], S.MODE_COLORS["normal"], S.MODE_COLORS["detail"]])

    def grid_until(L):
        g = np.full(reef.shape, np.nan); g[:L] = grid_all[:L]
        return [[None if np.isnan(v) else float(v) for v in row] for row in g]

    def robot_at(L):
        if L <= 0:
            return [0], [0]
        lane = L - 1
        return ([n - 1] if lane % 2 == 0 else [0]), [lane]

    steps = list(range(step_lanes, lanes + 1, step_lanes))
    if steps[-1] != lanes:
        steps.append(lanes)
    hm = lambda L: go.Heatmap(z=grid_until(L), zmin=-0.5, zmax=2.5, colorscale=cs, showscale=False,
                             xgap=0.5, ygap=0.5, hoverinfo="skip")
    rb = lambda L: go.Scatter(x=robot_at(L)[0], y=robot_at(L)[1], mode="markers", hoverinfo="skip",
                             marker=dict(size=16, color="#ffd60a", line=dict(width=2, color=S.INK), symbol="diamond"))
    frames = [go.Frame(data=[hm(L), rb(L)], name=str(L)) for L in steps]
    fig = go.Figure(data=[hm(lanes), rb(lanes)], frames=frames)     # starts showing the finished survey
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", showactive=False, x=0, y=-0.16, xanchor="left", yanchor="top",
                          pad=dict(t=0, r=10), bgcolor=S.SOFT, bordercolor=S.LINE, font=dict(size=13),
                          buttons=[dict(label="▶  Play from start", method="animate",
                                        args=[None, dict(frame=dict(duration=180, redraw=True), fromcurrent=False,
                                                         transition=dict(duration=0), mode="immediate")]),
                                   dict(label="❚❚  Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                                           transition=dict(duration=0))])])],
        sliders=[dict(active=len(steps) - 1, x=0.25, len=0.75, y=-0.1, pad=dict(t=0), currentvalue=dict(prefix="Lanes surveyed: ", font=dict(size=12)),
                      steps=[dict(method="animate", label=str(L),
                                  args=[[str(L)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                       transition=dict(duration=0))]) for L in steps])])
    # fixed zoom: always the whole reef (a descending range puts lane 0 at the top)
    fig.update_yaxes(autorange=False, range=[lanes - 0.5, -0.5], title="Lane", showgrid=False)
    fig.update_xaxes(autorange=False, range=[-0.5, n - 0.5], title="Distance along lane (m)")
    fig.update_layout(plot_bgcolor="#ffffff")
    base(fig, 470, title, legend=False)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=90))
    return fig


def cumulative(logs, key, ylabel, title, scale=1.0, every=20):
    fig = go.Figure()
    for name, lg in logs.items():
        y = np.cumsum(lg[key]) * scale
        x = np.arange(1, len(y) + 1)
        fig.add_trace(go.Scatter(x=x[::every], y=y[::every], mode="lines", name=name,
                                 line=dict(width=3, color=S.VEHICLE_COLORS.get(name, S.MUTED)),
                                 hovertemplate=f"<b>{name}</b><br>%{{x}} m surveyed<br>{ylabel}: %{{y:,.1f}}<extra></extra>"))
    fig.update_xaxes(title="Metres surveyed")
    fig.update_yaxes(title=ylabel)
    return base(fig, 360, title)


def bars(names, values, title, fmt="{:,.1f}", suffix="", errors=None, height=320):
    fig = go.Figure(go.Bar(
        x=names, y=values, marker=dict(color=[S.VEHICLE_COLORS.get(n, S.MUTED) for n in names], cornerradius=8),
        text=[fmt.format(v) + suffix for v in values], textposition="outside", cliponaxis=False,
        error_y=dict(type="data", array=errors, visible=errors is not None, color="#8e8e93"),
        hovertemplate="<b>%{x}</b><br>%{y:,.2f}" + suffix + "<extra></extra>"))
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_xaxes(tickfont=dict(size=11))
    return base(fig, height, title, legend=False)


def tradeoff(df, perception, labels=None, title=None):
    """One line per robot through its three sand/rock shares, for one way of seeing."""
    labels = labels or {}
    d = df[df.classifier == perception]
    fig = go.Figure()
    order = ["Fixed standard", "Fixed detailed", "Adaptive (no linger)", "Adaptive", "RL agent"]
    for veh in order:
        g = d[d.strategy == veh].sort_values("sand_fraction")
        if g.empty:
            continue
        col = S.VEHICLE_COLORS.get(veh, S.MUTED)
        fixed = veh.startswith("Fixed")
        if fixed:                       # fixed robots do the same thing on every reef: one dot
            g = g.iloc[[0]]
        txt = ["" if fixed else f"{int(round(100 * sf))}%" for sf in g.sand_fraction]
        fig.add_trace(go.Scatter(
            x=g.energy_kj_mean, y=g.bleached_detail_pct_mean, name=veh,
            mode="markers+text" if fixed else "lines+markers+text", text=txt,
            textposition="top center", textfont=dict(size=11, color=col),
            line=dict(width=3, color=col), marker=dict(size=14 if fixed else 11, color=col, line=dict(width=2, color="white")),
            customdata=[[int(round(100 * sf)), int(round(p)), round(float(t), 1)]
                        for sf, p, t in zip(g.sand_fraction, g.photos_mean, g.time_min_mean)],
            hovertemplate=("<b>" + veh + "</b><br>" + ("Same on every reef" if fixed else "Sand/rock: %{customdata[0]}%") +
                           "<br>Energy: %{x:.0f} kJ<br>Bleached coral in detail: %{y:.1f}%"
                           "<br>Photos: %{customdata[1]:,}<br>Time: %{customdata[2]} min<extra></extra>")))
    fig.add_annotation(xref="paper", yref="paper", x=0.0, y=1.0, ax=0.30, ay=0.55, axref="paper", ayref="paper",
                       showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor="#00875a",
                       text="★ <b>Best corner</b>: more detail, less energy", align="left", xanchor="left",
                       font=dict(size=12, color="#00875a"), bgcolor="#e8f7f0", borderpad=6)
    fig.update_xaxes(title="Energy used (kJ)")
    fig.update_yaxes(title="Bleached coral photographed in detail (%)", range=[-5, 108])
    base(fig, 460, title)
    fig.update_layout(legend=dict(orientation="v", x=1.01, y=1, xanchor="left", yanchor="top",
                                  bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
                      margin=dict(l=10, r=10, t=16, b=10))
    return fig


def confusion(cm, labels, title, color="Blues", normalise=False):
    z = np.array(cm, dtype=float)
    txt = [[f"{int(v):,}" for v in row] for row in z]
    if normalise:
        z = z / z.sum(1, keepdims=True) * 100
        txt = [[f"{v:.1f}%" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, colorscale=color, showscale=False, text=txt,
                               texttemplate="%{text}", textfont=dict(size=14),
                               hovertemplate="True: %{y}<br>Predicted: %{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="True")
    fig.update_xaxes(title="Predicted", side="bottom")
    return base(fig, 340, title, legend=False)


def lines(x, series, xtitle, ytitle, title, log_x=False, height=340, colors=None):
    fig = go.Figure()
    for i, (name, ys) in enumerate(series.items()):
        col = (colors or {}).get(name, S.VEHICLE_COLORS.get(name, [S.ACCENT, S.INK, "#00a383", S.DANGER][i % 4]))
        fig.add_trace(go.Scatter(x=x, y=ys, mode="lines+markers", name=name, line=dict(width=3, color=col),
                                 marker=dict(size=8)))
    if log_x:
        fig.update_xaxes(type="log")
    fig.update_xaxes(title=xtitle); fig.update_yaxes(title=ytitle)
    return base(fig, height, title, legend=len(series) > 1)


def q_table(Q, labels, title="What the agent learned (value of each action)"):
    z = np.array(Q)
    best = z.argmax(1)
    txt = [[("★ " if j == best[i] else "") + f"{z[i, j]:+.3f}" for j in range(3)] for i in range(len(z))]
    fig = go.Figure(go.Heatmap(z=z, x=["Fast", "Normal", "Detail"], y=labels, colorscale="RdBu", zmid=0,
                               text=txt, texttemplate="%{text}", showscale=False,
                               hovertemplate="%{y}<br>Action: %{x}<br>Average reward: %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    return base(fig, 40 + 26 * len(labels), title, legend=False)


def kg_network(reef, G, title="Knowledge graph on the reef"):
    fig = go.Figure(go.Heatmap(z=reef, zmin=-0.5, zmax=2.5, colorscale=_discrete([S.SAND, S.HEALTHY, "#ffffff"]),
                               showscale=False, opacity=0.55, hoverinfo="skip"))
    ex, ey = [], []
    for a, b in G.edges:
        ex += [G.nodes[a]["cell"], G.nodes[b]["cell"], None]; ey += [G.nodes[a]["lane"], G.nodes[b]["lane"], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1, color="#636366"), hoverinfo="skip", name="Link"))
    for conf, col, name in [(True, S.DANGER, "Confirmed report"), (False, "#ff9f0a", "Unconfirmed (isolated)")]:
        ns = [k for k in G.nodes if G.nodes[k]["confirmed"] == conf]
        if not ns:
            continue
        fig.add_trace(go.Scatter(
            x=[G.nodes[k]["cell"] for k in ns], y=[G.nodes[k]["lane"] for k in ns], mode="markers", name=name,
            marker=dict(size=[7 + 2.5 * G.degree[k] for k in ns], color=col, line=dict(width=1, color="white")),
            customdata=[[G.degree[k], round(G.nodes[k]["centrality"], 5), CLASS_TXT[G.nodes[k]["true_class"]]] for k in ns],
            hovertemplate=("<b>Lane %{y}, %{x} m</b><br>Linked neighbours: %{customdata[0]}<br>"
                           "Centrality: %{customdata[1]}<br>Truth: %{customdata[2]}<extra></extra>")))
    fig.update_yaxes(autorange=False, range=[reef.shape[0] - 0.5, -0.5], title="Lane", showgrid=False)
    fig.update_xaxes(autorange=False, range=[-0.5, reef.shape[1] - 0.5], title="Distance along lane (m)")
    return base(fig, 460, title)


def energy_curve(hotel, drag, modes):
    """Energy per metre for travel at each speed, with the three modes marked."""
    v = np.linspace(0.2, 2.5, 120)
    e = (hotel + drag * v ** 3) / v
    vopt = (hotel / (2 * drag)) ** (1 / 3) if drag > 0 else float("nan")
    fig = go.Figure(go.Scatter(x=v, y=e, mode="lines", line=dict(width=3, color=S.INK), name="Energy per metre",
                               hovertemplate="Speed %{x:.2f} m/s<br>%{y:.1f} J per metre<extra></extra>"))
    for k, name in [(0, "fast"), (1, "normal"), (2, "detail")]:
        sp = modes[k]["speed"]
        fig.add_trace(go.Scatter(x=[sp], y=[(hotel + drag * sp ** 3) / sp], mode="markers+text", name=name.title(),
                                 text=[name.title()], textposition="top center",
                                 marker=dict(size=14, color=S.MODE_COLORS[name], line=dict(width=2, color=S.INK))))
    if np.isfinite(vopt):
        fig.add_vline(x=vopt, line_dash="dot", line_color=S.ACCENT,
                      annotation_text=f"Best speed ≈ {vopt:.2f} m/s", annotation_position="top")
    fig.update_xaxes(title="Speed (m/s)"); fig.update_yaxes(title="Travel energy (J per metre)")
    return base(fig, 360, "Travel energy per metre (without photos)", legend=False)


def grouped(categories, series, title, ytitle, height=340):
    fig = go.Figure()
    palette = [S.ACCENT, S.INK, "#00a383", "#ff9f0a"]
    for i, (name, vals) in enumerate(series.items()):
        fig.add_trace(go.Bar(x=categories, y=vals, name=name, marker=dict(color=palette[i % 4], cornerradius=6),
                             text=[f"{v:.2f}" for v in vals], textposition="outside", cliponaxis=False))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=ytitle)
    return base(fig, height, title)
