"""
ReefScan Research Console
A second dashboard for researchers and reef managers: raw data and statistics, custom experiments,
model diagnostics and a field hotspot planner.

FYP: Autonomous Resource-Aware Agent for Priority-Based Coral Reef Health Mapping
Muhammad Alif Najmi bin Mohammad Zamri (22005632), Universiti Teknologi PETRONAS

Run:   python -m pip install -r requirements.txt
       python -m streamlit run researcher.py
"""
import os, io, json, math
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.graph_objects as go
import reefsim as rs
import knowledge_graph as kg
import rl_agent as rl
import charts as C
import style as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
if not os.path.exists(os.path.join(DATA, "summary.csv")) and os.path.exists(os.path.join(HERE, "summary.csv")):
    DATA = HERE
UNET_M = rs.load_unet_perception(os.path.join(DATA, "unet_results.json"))
PERCEPTIONS = list(rs.CLASSIFIERS) + ([rs.UNET_NAME, rs.FUSED_NAME] if UNET_M is not None else [])
PLABEL = {"Perfect classifier": "Perfect (never wrong)",
          "Measured classifier": "Classifier + seabed sensor (measured error rates)",
          rs.UNET_NAME: "Camera only (Tiny U-Net)",
          rs.FUSED_NAME: "Camera + depth + position (fused)",
          "Custom classifier": "Custom error rates (set below)"}
RL_TAG = {rs.UNET_NAME: "_unet", rs.FUSED_NAME: "_fused"}
ROBOTS = ["Fixed standard", "Fixed detailed", "Adaptive", "Adaptive (no linger)", "RL agent"]
METRICS = {"energy_kj": "Energy (kJ)", "time_min": "Time (min)", "photos": "Photos", "storage_gb": "Storage (GB)",
           "bleached_detail_pct": "Bleached in detail (%)", "photos_on_sand_pct": "Photos over sand (%)",
           "photos_per_bleached_cell": "Photos per bleached metre", "reward": "Reward (Equation 6)"}
ROBOT_COLORS = {"Fixed standard": "#8a96a3", "Fixed detailed": "#14212e", "Adaptive": "#E8603C",
                "Adaptive (no linger)": "#F2A48C", "RL agent": "#1E8E78"}

st.set_page_config(page_title="ReefScan · Research Console", page_icon="🔬", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(S.CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def html(s):
    st.markdown(s, unsafe_allow_html=True)


def plot(fig, key=None):
    try:
        return st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False}, key=key)
    except Exception:
        return st.plotly_chart(fig, use_container_width=True, theme=None, config={"displaylogo": False}, key=key)


def table(df, **kw):
    try:
        return st.dataframe(df, hide_index=True, width="stretch", **kw)
    except Exception:
        return st.dataframe(df, hide_index=True, use_container_width=True, **kw)


def download(df, name, label="Download CSV", key=None):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv", key=key)


@st.cache_data
def load_csv(name):
    p = os.path.join(DATA, name)
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data
def load_json(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None


@st.cache_resource
def load_q(perception):
    p = os.path.join(DATA, f"rl_Q{RL_TAG.get(perception, '')}.npy")
    return np.load(p) if os.path.exists(p) else np.load(os.path.join(DATA, "rl_Q.npy"))


def ci95(x):
    """Mean and half-width of an approximate 95% confidence interval (normal approximation)."""
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan")), float("nan")
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def describe(df, by, cols):
    rows = []
    for g, d in df.groupby(by, sort=False):
        for c in cols:
            m, h = ci95(d[c])
            rows.append({**(dict(zip(by, g)) if isinstance(g, tuple) else {by[0]: g}), "Metric": METRICS.get(c, c),
                         "n": int(d[c].notna().sum()), "Mean": round(m, 3), "SD": round(float(d[c].std(ddof=1)), 3),
                         "95% CI low": round(m - h, 3), "95% CI high": round(m + h, 3),
                         "Min": round(float(d[c].min()), 3), "Max": round(float(d[c].max()), 3)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ header
html(S.brand().replace("Survey Agent", "Research Console") if "Survey Agent" in S.brand() else S.brand())
html(S.section("For researchers and reef managers", "ReefScan Research Console",
               "The full data behind the study, live experiments with your own settings, model diagnostics and a hotspot "
               "planner for field teams. Everything here is simulation-based: the models are real, the reefs and vehicle are simulated."))
st.caption("Looking for the guided version for examiners and the public? Open the main dashboard: https://reefscan-hanisaina.streamlit.app")
tabs = st.tabs(["📊 Data and statistics", "🧪 Run an experiment", "🔍 Model diagnostics", "🗺️ Hotspot planner"])

# ================================================================== 1. DATA AND STATISTICS
with tabs[0]:
    runs = load_csv("all_runs.csv")
    if runs is None:
        html(S.callout("data/all_runs.csv not found. Run <code>python run_experiments.py</code>.", "warn"))
    else:
        st.subheader("Survey results, one row per reef and robot")
        st.caption("90 test reefs (30 at each sand share) × 5 robots × each way of seeing. Every robot meets the same reefs "
                   "and the same recognition mistakes, so comparisons between robots are paired.")
        c = st.columns(3)
        percs = [p for p in PERCEPTIONS if p in runs.classifier.unique()]
        perc = c[0].selectbox("How the robot sees", percs, index=min(1, len(percs) - 1), format_func=lambda p: PLABEL.get(p, p), key="d_perc")
        sands = c[1].multiselect("Sand or rock share", sorted(runs.sand_fraction.unique()), default=sorted(runs.sand_fraction.unique()),
                                 format_func=lambda s: f"{int(s * 100)}%", key="d_sand")
        robots = c[2].multiselect("Robots", ROBOTS, default=ROBOTS, key="d_rob")
        sel = runs[(runs.classifier == perc) & runs.sand_fraction.isin(sands) & runs.strategy.isin(robots)]

        st.markdown("**Descriptive statistics**")
        mcols = st.multiselect("Metrics", list(METRICS), default=["energy_kj", "time_min", "photos", "bleached_detail_pct"],
                               format_func=lambda m: METRICS[m], key="d_met")
        if len(sel) and mcols:
            desc = describe(sel, ["sand_fraction", "strategy"], mcols)
            desc["sand_fraction"] = (desc.sand_fraction * 100).astype(int).astype(str) + "%"
            desc = desc.rename(columns={"sand_fraction": "Sand", "strategy": "Robot"})
            table(desc, height=320)
            download(desc, "reefscan_descriptive_stats.csv", key="dl_desc")

        st.markdown("**Distribution across reefs**")
        c = st.columns([1, 3])
        metric = c[0].selectbox("Metric", list(METRICS), index=0, format_func=lambda m: METRICS[m], key="d_box")
        fig = go.Figure()
        for r in robots:
            d = sel[sel.strategy == r]
            fig.add_trace(go.Box(y=d[metric], x=[f"{int(s * 100)}% sand" for s in d.sand_fraction], name=r,
                                 marker_color=ROBOT_COLORS[r], boxpoints="all", jitter=0.4, pointpos=0, marker_size=3))
        fig.update_layout(boxmode="group")
        fig.update_yaxes(title=METRICS[metric])
        with c[1]:
            plot(C.base(fig, 420, None), key="d_boxfig")

        st.markdown("**Paired comparison against a baseline**")
        st.caption("For each reef, the chosen robot's value minus the baseline's value on the same reef and mistakes. "
                   "A confidence interval that does not cross 0 means a consistent difference.")
        c = st.columns(3)
        base_r = c[0].selectbox("Baseline", ROBOTS, index=1, key="d_base")
        comp_r = c[1].selectbox("Compared robot", ROBOTS, index=2, key="d_comp")
        pm = c[2].selectbox("Metric ", list(METRICS), index=0, format_func=lambda m: METRICS[m], key="d_pm")
        rows = []
        for s in sands:
            a = runs[(runs.classifier == perc) & np.isclose(runs.sand_fraction, s) & (runs.strategy == comp_r)].set_index("map")[pm]
            b = runs[(runs.classifier == perc) & np.isclose(runs.sand_fraction, s) & (runs.strategy == base_r)].set_index("map")[pm]
            diff = (a - b).dropna()
            m, h = ci95(diff)
            bm = float(b.mean())
            rows.append({"Sand": f"{int(s * 100)}%", "Reefs": len(diff), "Mean difference": round(m, 2),
                         "95% CI": f"{m - h:.2f} to {m + h:.2f}",
                         "Change vs baseline": f"{100 * m / bm:+.1f}%" if bm else "n/a",
                         f"{comp_r} better on": f"{int(((diff < 0) if pm in ('energy_kj', 'time_min', 'photos', 'storage_gb', 'photos_on_sand_pct') else (diff > 0)).sum())} of {len(diff)} reefs"})
        if rows:
            table(pd.DataFrame(rows))

        st.markdown("**All result files**")
        files = sorted(f for f in os.listdir(DATA) if f.endswith((".csv", ".json")))
        c = st.columns([2, 1])
        f = c[0].selectbox("File", files, index=files.index("summary.csv") if "summary.csv" in files else 0, key="d_file")
        path = os.path.join(DATA, f)
        if f.endswith(".csv"):
            df = pd.read_csv(path)
            st.caption(f"{len(df):,} rows × {df.shape[1]} columns")
            table(df, height=320)
        else:
            st.json(json.load(open(path)), expanded=False)
        with open(path, "rb") as fh:
            c[1].download_button(f"Download {f}", fh.read(), file_name=f, key="dl_file")
        settings = load_json("settings.json")
        if settings:
            with st.expander("Experiment settings used for every result (settings.json)"):
                st.json(settings)

# ================================================================== 2. RUN AN EXPERIMENT
@st.cache_data(show_spinner=False)
def run_experiment(sand, n_reefs, seed0, perception, custom_clf, sensor_acc, linger, modes_t, robot_t, robots_t):
    modes = {i: dict(speed=modes_t[2 * i], spacing=modes_t[2 * i + 1]) for i in range(3)}
    robot = dict(rs.ROBOT); robot.update(dict(zip(["hotel_power_w", "drag_coeff", "photo_energy_j", "look_energy_j"], robot_t)))
    sensor = dict(sand_correct=sensor_acc, coral_correct=sensor_acc)
    Q = load_q(perception if perception in RL_TAG else "Measured classifier")
    rng = np.random.default_rng(seed0)
    out = []
    for k in range(n_reefs):
        seed = int(rng.integers(1 << 31))
        reef = rs.make_reef(np.random.default_rng(seed), sand)
        rows, cols, cells = rs.lawnmower_path(reef)
        if perception == "Custom classifier":
            seen = rs.observe_path(cells, np.random.default_rng(seed + 7),
                                   dict(p_bleached_found=custom_clf[0], p_healthy_called_bleached=custom_clf[1]), sensor)
        else:
            seen = rs.perceive(cells, np.random.default_rng(seed + 7), perception, UNET_M, sensor=sensor,
                               ctx=dict(reef=reef, rows=rows, cols=cols, seed=seed))
        nb = rs.neighbour_feature(rows, cols, seen)
        pols = rs.strategies(linger); pols["RL agent"] = rl.QPolicy(Q)
        for name in robots_t:
            log = rs.run_survey(cells, seen, pols[name], nb, mode_settings=modes, robot=robot)
            m = rs.metrics(cells, log, robot)
            m["reward"] = rl.total_reward(cells, log)
            out.append(dict(reef=k + 1, seed=seed, robot=name, **m))
    return pd.DataFrame(out)


with tabs[1]:
    st.subheader("Run your own experiment")
    st.caption("Change the reef, the recognition errors, the decision rule and the vehicle, then compare the robots on the "
               "same reefs. Reefs are generated from your seed, so the same settings always give the same result.")
    c = st.columns(4)
    sand = c[0].slider("Sand or rock share", 0.1, 0.9, 0.5, 0.05, key="e_sand")
    n_reefs = c[1].slider("Number of reefs", 1, 30, 10, key="e_n", help="Each reef is 40 lanes × 100 m. More reefs = tighter confidence intervals but slower.")
    seed0 = int(c[2].number_input("Random seed", 0, 10_000_000, 2026, step=1, key="e_seed"))
    eperc = c[3].selectbox("How the robot sees", PERCEPTIONS + ["Custom classifier"], index=1,
                           format_func=lambda p: PLABEL.get(p, p), key="e_perc")
    c = st.columns(4)
    custom = (rs.CLASSIFIERS["Measured classifier"]["p_bleached_found"], rs.CLASSIFIERS["Measured classifier"]["p_healthy_called_bleached"])
    if eperc == "Custom classifier":
        custom = (c[0].slider("Bleached found (recall)", 0.0, 1.0, 0.76, 0.01, key="e_rec"),
                  c[1].slider("Healthy called bleached", 0.0, 0.5, 0.05, 0.01, key="e_fa"))
    sensor_acc = c[2].slider("Seabed sensor accuracy", 0.5, 1.0, 0.97, 0.01, key="e_sens",
                             help="How often the coral-or-sand sensor is right (classifier perceptions only).")
    linger = c[3].slider("Linger after bleaching (m)", 0, 10, 3, key="e_ling")
    with st.expander("Vehicle and mode settings"):
        c = st.columns(3)
        mt = []
        for i, (nm, v, sp) in enumerate([("Fast", 1.5, 2.0), ("Normal", 1.0, 0.5), ("Detail", 0.5, 0.25)]):
            mt += [c[i].number_input(f"{nm} speed (m/s)", 0.1, 3.0, v, 0.1, key=f"e_v{i}"),
                   c[i].number_input(f"{nm} photo every (m)", 0.05, 5.0, sp, 0.05, key=f"e_s{i}")]
        c = st.columns(4)
        rt = (c[0].number_input("Hotel power (W)", 0.0, 500.0, 40.0, 5.0, key="e_h"),
              c[1].number_input("Drag constant", 0.0, 100.0, 12.0, 1.0, key="e_d"),
              c[2].number_input("Energy per photo (J)", 0.0, 100.0, 10.0, 1.0, key="e_p"),
              c[3].number_input("Energy per look (J)", 0.0, 50.0, 0.2, 0.1, key="e_l"))
        vstar = (rt[0] / (2 * rt[1])) ** (1 / 3) if rt[1] > 0 else float("inf")
        st.caption(f"Under these settings the most energy-efficient cruising speed is about {vstar:.2f} m/s "
                   "(energy per metre = hotel/v + drag·v²).")
    erob = st.multiselect("Robots to compare", ROBOTS, default=ROBOTS, key="e_rob")
    if st.button("Run experiment", type="primary", key="e_run") or "e_done" in st.session_state:
        st.session_state["e_done"] = True
        with st.spinner(f"Simulating {n_reefs} reefs × {len(erob)} robots..."):
            res = run_experiment(sand, n_reefs, seed0, eperc, tuple(custom), sensor_acc, linger, tuple(mt), tuple(rt), tuple(erob))
        if len(res):
            agg = describe(res, ["robot"], ["energy_kj", "time_min", "photos", "bleached_detail_pct", "photos_on_sand_pct", "reward"])
            wide = agg.pivot_table(index="robot", columns="Metric", values="Mean", sort=False).reset_index().rename(columns={"robot": "Robot"})
            if "Fixed detailed" in erob:
                ref = res[res.robot == "Fixed detailed"][["energy_kj", "photos"]].mean()
                wide["Energy vs fixed detailed"] = [f"{100 * (res[res.robot == r].energy_kj.mean() / ref.energy_kj - 1):+.1f}%" for r in wide.Robot]
            if "Fixed standard" in erob:
                ref = res[res.robot == "Fixed standard"].energy_kj.mean()
                wide["Energy vs fixed standard"] = [f"{100 * (res[res.robot == r].energy_kj.mean() / ref - 1):+.1f}%" for r in wide.Robot]
            table(wide.round(2))
            fig = go.Figure()
            for r in erob:
                d = res[res.robot == r]
                me, he = ci95(d.energy_kj); md, hd = ci95(d.bleached_detail_pct)
                fig.add_trace(go.Scatter(x=[me], y=[md], mode="markers+text", text=[r], textposition="top center",
                                         marker=dict(size=14, color=ROBOT_COLORS[r]), name=r,
                                         error_x=dict(type="data", array=[he] if not np.isnan(he) else [0], visible=True),
                                         error_y=dict(type="data", array=[hd] if not np.isnan(hd) else [0], visible=True)))
            fig.update_xaxes(title="Mission energy (kJ), mean ± 95% CI")
            fig.update_yaxes(title="Bleached coral in detail (%)", range=[-5, 108])
            plot(C.base(fig, 440, "Energy against detail for your settings", legend=False), key="e_fig")
            with st.expander("Every reef and robot (raw rows)"):
                table(res.round(3), height=300)
            download(res, "reefscan_custom_experiment.csv", key="dl_exp")
            download(agg, "reefscan_custom_experiment_stats.csv", label="Download statistics CSV", key="dl_exp2")

# ================================================================== 3. MODEL DIAGNOSTICS
def cm_fig(cm, labels, title, norm):
    cm = np.asarray(cm, dtype=float)
    z = cm / cm.sum(1, keepdims=True) if norm else cm
    txt = [[(f"{v:.0%}" if norm else f"{int(v):,}") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, text=txt, texttemplate="%{text}", colorscale="Blues", showscale=False,
                               hovertemplate="True %{y}<br>Predicted %{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="True")
    fig.update_xaxes(title="Predicted")
    return C.base(fig, 340, title, legend=False)


def metrics_from_cm(cm, labels):
    cm = np.asarray(cm, dtype=float)
    out = []
    for i, l in enumerate(labels):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * p * r / (p + r) if p + r else float("nan")
        out.append({"Class": l, "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f1, 3), "Support": int(cm[i].sum())})
    return pd.DataFrame(out)


with tabs[2]:
    clf, cr, unet, rob, ho = (load_json(n) for n in ["classifier.json", "clean_retrain.json", "unet_results.json",
                                                      "robustness.json", "holoocean_summary.json"])
    norm = st.toggle("Show confusion matrices as row percentages", value=False, key="m_norm")
    if clf:
        st.subheader("MobileNetV3-Small classifier")
        tr = clf.get("training", {})
        st.caption(f"{tr.get('pretrained', '')}-pretrained, {tr.get('epochs', '')} epochs on {tr.get('device', '')}, "
                   f"{tr.get('train_minutes', '')} min, {tr.get('model_size_mb', '')} MB, {tr.get('ms_per_image', '')} ms per photo. "
                   f"Near-copy check (dHash, ≤{clf['near_copy_check']['threshold_bits']} bits): "
                   f"{clf['near_copy_check']['test']['percent']}% of test and {clf['near_copy_check']['valid']['percent']}% of validation photos.")
        evals = {"Original model · own test set": clf["kaggle_test"]["confusion_matrix"],
                 "Original model · independent photos": clf["independent_test"]["confusion_matrix"]}
        if cr:
            evals["Clean retrain · clean test set"] = cr["clean_test"]["confusion_matrix"]
            evals["Clean retrain · independent photos"] = cr["independent_clean_model"]["confusion_matrix"]
        chosen = st.multiselect("Evaluations to compare", list(evals), default=list(evals)[1:3] if len(evals) > 2 else list(evals), key="m_ev")
        cols = st.columns(max(1, min(len(chosen), 4)))
        allm = []
        for i, name in enumerate(chosen):
            with cols[i % len(cols)]:
                plot(cm_fig(evals[name], clf["classes"], name, norm), key=f"m_cm{i}")
            mdf = metrics_from_cm(evals[name], clf["classes"]); mdf.insert(0, "Evaluation", name)
            acc = np.trace(np.asarray(evals[name])) / np.sum(evals[name]); mdf["Accuracy"] = round(float(acc), 4)
            allm.append(mdf)
        if allm:
            table(pd.concat(allm))
        if "accuracy_by_source" in clf.get("independent_test", {}):
            st.caption("Independent accuracy by original source: " +
                       ", ".join(f"{k} {100 * v:.1f}%" for k, v in clf["independent_test"]["accuracy_by_source"].items()))
        if cr:
            st.caption(f"Near-copy groups: {cr['groups']:,} groups from {cr['photos']:,} photos (largest {cr['largest_group']:,}, "
                       f"median {cr['median_group']}, {cr['groups_of_one']} single photos). Clean split: "
                       f"{cr['clean_split']['train']:,} / {cr['clean_split']['valid']:,} / {cr['clean_split']['test']:,}.")
    if unet:
        st.subheader("Tiny U-Net")
        st.caption(f"{unet['parameters']:,} parameters, {unet['model_size_mb']} MB, input {unet['input_size'][0]}×{unet['input_size'][1]}, "
                   f"{unet['epochs']} epochs in {unet['train_minutes']} min. {unet['ms_per_image_gpu']} ms per photo on GPU, "
                   f"{unet['ms_per_image_cpu']} ms on CPU. Pixel accuracy {100 * unet['test_pixel_accuracy']:.1f}%, mean IoU {100 * unet['test_mean_iou']:.1f}%.")
        c = st.columns(2)
        per = pd.DataFrame({"Class": unet["classes"], "IoU": [unet["test_iou"][k] for k in unet["classes"]],
                            "Dice": [unet["test_dice"][k] for k in unet["classes"]],
                            "Share of training pixels": [unet["train_pixel_share"][k] for k in unet["classes"]]})
        with c[0]:
            table(per.round(4))
            plot(cm_fig(unet["photo_confusion"], unet["photo_classes"], "Photo-level decisions", norm), key="m_ucm2")
        with c[1]:
            plot(cm_fig(unet["pixel_confusion"], unet["classes"], "Pixel confusion (test set)", True), key="m_ucm")
            st.caption(f"Photo rule: bleached if ≥{100 * unet['photo_rule']['bleached_min_share']:.0f}% of pixels are bleached, "
                       f"else coral if ≥{100 * unet['photo_rule']['coral_min_share']:.0f}% coral. Photo accuracy {100 * unet['photo_accuracy']:.1f}%.")
    if rob:
        st.subheader("Robustness to murky water")
        names = rob["level_names"]
        c = st.columns([1, 3])
        series = c[0].multiselect("Curves", ["Original classifier", "Original, colour-corrected", "Clean classifier",
                                             "Clean, colour-corrected", "Tiny U-Net (bleached Dice)", "Tiny U-Net, colour-corrected"],
                                  default=["Original classifier", "Original, colour-corrected", "Tiny U-Net (bleached Dice)"], key="m_mk")
        src = {"Original classifier": [100 * d["bleached_recall"] for d in rob["classifier"]["original_raw"]],
               "Original, colour-corrected": [100 * d["bleached_recall"] for d in rob["classifier"]["original_corrected"]],
               "Clean classifier": [100 * d["bleached_recall"] for d in rob["classifier"]["clean_raw"]],
               "Clean, colour-corrected": [100 * d["bleached_recall"] for d in rob["classifier"]["clean_corrected"]],
               "Tiny U-Net (bleached Dice)": [100 * d["bleached_dice"] for d in rob["unet"]["raw"]],
               "Tiny U-Net, colour-corrected": [100 * d["bleached_dice"] for d in rob["unet"]["corrected"]]}
        with c[1]:
            plot(C.lines(names, {k: src[k] for k in series}, "", "%", "Bleached coral found (classifier recall) or outlined (U-Net Dice)",
                         height=380), key="m_mkfig")
        rows = []
        for i, n in enumerate(names):
            rows.append({"Water": n,
                         "Classifier accuracy": rob["classifier"]["original_raw"][i]["accuracy"],
                         "Classifier bleached recall": rob["classifier"]["original_raw"][i]["bleached_recall"],
                         "Corrected recall": rob["classifier"]["original_corrected"][i]["bleached_recall"],
                         "U-Net mean IoU": rob["unet"]["raw"][i]["mean_iou"],
                         "U-Net photo accuracy": rob["unet"]["raw"][i]["photo_accuracy"]})
        mt = pd.DataFrame(rows)
        table(mt.round(4)); download(mt, "reefscan_murky_water.csv", key="dl_mk")
    if ho:
        st.subheader("HoloOcean vehicle check")
        rows = []
        for r, v in ho["robots"].items():
            h, d = v["holoocean"], v["simulation_2d"]
            rows.append({"Robot": r, "Energy 2-D (kJ)": d["energy_kj"], "Energy HoloOcean (kJ)": h["energy_kj"],
                         "Time 2-D (min)": d["time_min"], "Time HoloOcean (min)": h["time_min"],
                         "Detail 2-D (%)": d["bleached_detail_pct"], "Detail HoloOcean (%)": h["bleached_detail_pct"],
                         "Slow-down mean (m)": h.get("slowdown_metres_mean"), "Slow-down max (m)": h.get("slowdown_metres_max"),
                         "Max drift (m)": h.get("max_path_error_m"), "Turns (s)": h.get("manoeuvre_time_s")})
        hdf = pd.DataFrame(rows)
        table(hdf); download(hdf, "reefscan_holoocean.csv", key="dl_ho")
        st.caption(f"{ho['world']} world, {ho['reef_metres']} m of one {int(ho['sand_fraction'] * 100)}% sand test reef, "
                   f"{ho['ticks_per_sec']} physics steps per second, one run per robot.")

# ================================================================== 4. HOTSPOT PLANNER
def graph_from_points(df, radius, min_degree):
    G = nx.Graph()
    pts = df[["x_m", "y_m"]].to_numpy(dtype=float)
    for i, (x, y) in enumerate(pts):
        G.add_node(i, x=float(x), y=float(y))
    for i in range(len(pts)):
        d = np.hypot(pts[i + 1:, 0] - pts[i, 0], pts[i + 1:, 1] - pts[i, 1])
        for j in np.where(d <= radius + 1e-9)[0]:
            G.add_edge(i, i + 1 + int(j))
    for n in G.nodes:
        G.nodes[n]["confirmed"] = G.degree[n] >= min_degree
    return G


def rank_hotspots(G, truth=None):
    H = G.subgraph([n for n, d in G.nodes(data=True) if d["confirmed"]])
    rows = []
    for comp in nx.connected_components(H):
        xs = [G.nodes[n]["x"] for n in comp]; ys = [G.nodes[n]["y"] for n in comp]
        row = {"Confirmed reports": len(comp), "Centre x (m)": round(float(np.mean(xs)), 1), "Centre y (m)": round(float(np.mean(ys)), 1),
               "Spans x (m)": f"{min(xs):.0f} to {max(xs):.0f}", "Spans y (m)": f"{min(ys):.0f} to {max(ys):.0f}",
               "Mean links": round(float(np.mean([G.degree[n] for n in comp])), 2)}
        if truth is not None:
            row["Truly bleached (check)"] = f"{100 * np.mean([truth[n] for n in comp]):.0f}%"
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["Confirmed reports", "Mean links"], ascending=False).reset_index(drop=True)
        out.insert(0, "Priority", range(1, len(out) + 1))
    return out


with tabs[3]:
    st.subheader("Hotspot planner for field teams")
    st.caption("Turns bleaching reports into a ranked list of places for divers to check first. Reports close to other reports are "
               "confirmed; isolated reports are probably false alarms. Use a simulated survey, or upload your own reports.")
    c = st.columns(3)
    source = c[0].radio("Reports from", ["Simulated survey", "Upload a CSV"], key="h_src", horizontal=True)
    radius = c[1].slider("Link distance (m)", 1.0, 6.0, kg.LINK_RADIUS_M, 0.1, key="h_rad",
                         help="Reports closer than this are linked. 2.3 m reaches the next lane (lanes are 2 m apart).")
    min_deg = c[2].slider("Links needed to confirm", 1, 5, kg.MIN_DEGREE, key="h_deg")
    pts, truth = None, None
    if source == "Simulated survey":
        c = st.columns(3)
        hs = c[0].selectbox("Sand or rock share", [0.3, 0.5, 0.7], index=1, format_func=lambda s: f"{int(s * 100)}%", key="h_sand")
        hk = c[1].number_input("Test reef number (1 to 30)", 1, 30, 1, key="h_reef")
        hp = c[2].selectbox("How the robot sees", PERCEPTIONS, index=min(1, len(PERCEPTIONS) - 1), format_func=lambda p: PLABEL.get(p, p), key="h_perc")
        seed = [s for s in rs.map_seeds([0.3, 0.5, 0.7], 30) if np.isclose(s[0], hs)][int(hk) - 1][2]
        reef = rs.make_reef(np.random.default_rng(seed), hs)
        rows_, cols_, cells = rs.lawnmower_path(reef)
        seen = rs.perceive(cells, np.random.default_rng(seed + 7), hp, UNET_M, ctx=dict(reef=reef, rows=rows_, cols=cols_, seed=seed))
        idx = np.where(seen == rs.BLEACHED)[0]
        pts = pd.DataFrame({"x_m": cols_[idx] * rs.MAP["cell_len_m"], "y_m": rows_[idx] * rs.MAP["lane_width_m"]})
        truth = (cells[idx] == rs.BLEACHED)
    else:
        tmpl = pd.DataFrame({"x_m": [12.0, 13.0, 13.0, 40.0, 41.0, 75.0], "y_m": [4.0, 4.0, 6.0, 10.0, 10.0, 22.0]})
        st.download_button("Download a template CSV", tmpl.to_csv(index=False).encode("utf-8"), file_name="bleaching_reports_template.csv",
                           mime="text/csv", key="h_tmpl")
        up = st.file_uploader("CSV with columns x_m and y_m (position of each bleaching report in metres)", type="csv", key="h_up")
        if up is not None:
            try:
                d = pd.read_csv(up)
                d.columns = [c_.strip().lower() for c_ in d.columns]
                if not {"x_m", "y_m"} <= set(d.columns):
                    html(S.callout("The file needs columns named <b>x_m</b> and <b>y_m</b>.", "warn"))
                else:
                    pts = d[["x_m", "y_m"]].apply(pd.to_numeric, errors="coerce").dropna().head(20000)
            except Exception as e:
                html(S.callout(f"Could not read the file: {S.esc(str(e))}", "warn"))
    if pts is not None and len(pts):
        G = graph_from_points(pts, radius, min_deg)
        conf = np.array([G.nodes[n]["confirmed"] for n in G.nodes])
        hot = rank_hotspots(G, truth)
        items = [(f"{len(pts):,}", "bleaching reports"), (f"{int(conf.sum()):,}", "confirmed", f"{100 * conf.mean():.0f}% of reports"),
                 (f"{len(hot)}", "hotspots to visit")]
        if truth is not None:
            fa = ~truth
            items.append((f"{100 * (~conf[fa]).mean():.0f}%" if fa.any() else "n/a", "false alarms filtered out",
                          f"{int(fa.sum())} false alarms in total"))
        html(S.cards(items, cols=len(items)))
        fig = go.Figure()
        for flag, nm, col in [(False, "Unconfirmed (likely false alarm)", "#c7ced6"), (True, "Confirmed", "#E8603C")]:
            sel = conf == flag
            fig.add_trace(go.Scattergl(x=pts.x_m[sel], y=pts.y_m[sel], mode="markers", name=nm,
                                       marker=dict(size=6, color=col), hovertemplate="x %{x} m, y %{y} m<extra></extra>"))
        if len(hot):
            top = hot.head(15)
            fig.add_trace(go.Scatter(x=top["Centre x (m)"], y=top["Centre y (m)"], mode="markers+text", text=top.Priority.astype(str),
                                     textposition="middle center", name="Hotspot (priority)",
                                     marker=dict(size=24, color="rgba(20,33,46,0.85)", line=dict(color="white", width=1.5)),
                                     textfont=dict(color="white", size=11), hovertemplate="Hotspot %{text}<extra></extra>"))
        fig.update_xaxes(title="x (m along the lanes)")
        fig.update_yaxes(title="y (m across the lanes)", autorange="reversed", scaleanchor="x", scaleratio=1)
        plot(C.base(fig, 520, "Bleaching reports and ranked hotspots"), key="h_map")
        if len(hot):
            table(hot)
            download(hot, "reefscan_hotspots.csv", label="Download hotspot list", key="dl_hot")
        if truth is not None:
            st.caption("'Truly bleached (check)' is only known in simulation, where the true reef is available. It shows how "
                       "reliable each hotspot is; field data would not have this column.")
    elif source == "Upload a CSV":
        st.info("Upload a CSV to see the hotspot map, or download the template to see the format.")

html(S.footer())
