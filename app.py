"""
ReefScan dashboard
FYP: Autonomous Resource-Aware Agent for Priority-Based Coral Reef Health Mapping
Muhammad Alif Najmi bin Mohammad Zamri (22005632), Universiti Teknologi PETRONAS

Run:   python -m pip install -r requirements.txt
       python -m streamlit run app.py
"""
import os, json
import numpy as np
import pandas as pd
import streamlit as st
import reefsim as rs
import knowledge_graph as kg
import rl_agent as rl
import charts as C
import style as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
if not os.path.exists(os.path.join(DATA, "summary.csv")) and os.path.exists(os.path.join(HERE, "summary.csv")):
    DATA = HERE                     # data files were uploaded next to app.py instead of inside data/
MODELS = os.path.join(HERE, "models")
FIGS = os.path.join(HERE, "figs")
_fb = os.path.join(HERE, "feedback_link.txt")          # paste the UAT form link into this file
FEEDBACK_URL = open(_fb, encoding="utf-8").read().strip().split()[0] if os.path.exists(_fb) and open(_fb, encoding="utf-8").read().strip() else ""
import importlib.util
HAS_TORCH = importlib.util.find_spec("torch") is not None
UNET_M = rs.load_unet_perception(os.path.join(DATA, "unet_results.json"))
PERCEPTIONS = list(rs.CLASSIFIERS) + ([rs.UNET_NAME, rs.FUSED_NAME] if UNET_M is not None else [])
PLABEL = {"Perfect classifier": "Perfect (never wrong)",
          "Measured classifier": "Classifier + seabed sensor (real error rates)",
          rs.UNET_NAME: "Camera only (Tiny U-Net)",
          rs.FUSED_NAME: "Camera + depth + position (fused)"}
RL_TAG = {rs.UNET_NAME: "_unet", rs.FUSED_NAME: "_fused"}
FU_VARIANTS = ["Camera only", "Depth only", "Camera + depth", "Camera + depth + position"]
VEHICLES = ["Fixed standard", "Fixed detailed", "Adaptive", "Adaptive (no linger)", "RL agent"]
ROBOT_INFO = [
    ("Fixed standard", "Normal speed and photo rate everywhere. Today's usual survey."),
    ("Fixed detailed", "Slow, with many photos everywhere. Catches all bleaching but costs the most."),
    ("Adaptive", "Our rule: fast over sand, normal over healthy coral, slow over bleaching, then stays slow 3 m."),
    ("Adaptive (no linger)", "Same rule without staying slow afterwards. Shows why lingering matters."),
    ("RL agent", "Learned its own rule from rewards instead of being told."),
]
PAGES = ["Home", "Classifier", "Tiny U-Net", "Simulation", "Results", "Knowledge Graph", "RL Agent", "Robot Lab", "Glossary & Limits"]
N_STEPS = 8

st.set_page_config(page_title="ReefScan · Coral Reef Survey Agent", page_icon="🪸", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(S.CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def html(s):
    st.markdown(s, unsafe_allow_html=True)


def wide(fn, *args, **kw):
    try:
        return fn(*args, width="stretch", **kw)
    except Exception:
        return fn(*args, use_container_width=True, **kw)


def plot(fig, key=None):
    try:
        return st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False}, key=key)
    except Exception:
        return st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False}, key=key)


def table(df, **kw):
    return wide(st.dataframe, df, hide_index=True, **kw)


def go_to(page):
    st.session_state["nav"] = page


def header(eyebrow, title, text=""):
    html(S.strip())
    i = PAGES.index(page) + 1
    if i <= N_STEPS:
        html(S.step(i, N_STEPS))
    html(S.section(eyebrow, title, text))


def next_button():
    i = PAGES.index(page)
    if i + 1 < len(PAGES):
        st.write("")
        c = st.columns([1, 3])
        c[0].button(f"Next: {PAGES[i + 1]}  →", on_click=go_to, args=(PAGES[i + 1],), key="next_btn")


def perception_box(col, key, options=None, index=1):
    options = options or PERCEPTIONS
    return col.selectbox("How the robot sees", options, index=min(index, len(options) - 1),
                         format_func=lambda p: PLABEL.get(p, p), key=key,
                         help="Perfect = never makes mistakes. Classifier + sensor = the trained MobileNetV3 with its "
                              "real error rates plus a simulated coral/sand sensor. Camera only = the Tiny U-Net alone.")


@st.cache_data
def load_csv(name):
    p = os.path.join(DATA, name)
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data
def load_json(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None


@st.cache_resource
def load_rl(perception="Measured classifier"):
    tag = RL_TAG.get(perception, "")
    q, n = os.path.join(DATA, f"rl_Q{tag}.npy"), os.path.join(DATA, f"rl_N{tag}.npy")
    if os.path.exists(q):
        return np.load(q), np.load(n)
    p = perception if perception in PERCEPTIONS and perception != "Perfect classifier" else "Measured classifier"
    Q, N, _ = rl.train(n_reefs=20, perception=p, unet_M=UNET_M)
    return Q, N


@st.cache_data
def simulate(sand, perception, seed, linger, sensor_acc):
    reef = rs.make_reef(np.random.default_rng(seed), sand)
    rows, cols, cells = rs.lawnmower_path(reef)
    seen = rs.perceive(cells, np.random.default_rng(seed + 7), perception, UNET_M,
                       sensor=dict(sand_correct=sensor_acc, coral_correct=sensor_acc),
                       ctx=dict(reef=reef, rows=rows, cols=cols, seed=seed))
    nb = rs.neighbour_feature(rows, cols, seen)
    Q, _ = load_rl(perception)
    pols = rs.strategies(linger); pols["RL agent"] = rl.QPolicy(Q)
    logs, mets = {}, {}
    for name, p in pols.items():
        lg = rs.run_survey(cells, seen, p, nb)
        logs[name] = lg
        m = rs.metrics(cells, lg); m["reward"] = rl.total_reward(cells, lg)
        mets[name] = m
    return reef, rows, cols, cells, seen, logs, mets


@st.cache_data
def robot_lab(mode_tuple, robot_tuple, sand, n_reefs, linger):
    modes = {i: dict(speed=mode_tuple[2 * i], spacing=mode_tuple[2 * i + 1]) for i in range(3)}
    robot = dict(rs.ROBOT); robot.update(dict(zip(["hotel_power_w", "drag_coeff", "photo_energy_j"], robot_tuple)))
    out = {}
    for _, k, seed in rs.map_seeds([sand], n_reefs):
        reef = rs.make_reef(np.random.default_rng(seed), sand)
        rows, cols, cells = rs.lawnmower_path(reef)
        seen = rs.perceive(cells, np.random.default_rng(seed + 7), "Measured classifier")
        for name, p in [("Fixed standard", rs.FixedPolicy(rs.NORMAL)), ("Fixed detailed", rs.FixedPolicy(rs.DETAIL)),
                        ("Adaptive", rs.RulePolicy(linger))]:
            m = rs.metrics(cells, rs.run_survey(cells, seen, p, None, mode_settings=modes, robot=robot), robot)
            out.setdefault(name, []).append(m)
    return {k: {m: float(np.nanmean([d[m] for d in v])) for m in v[0]} for k, v in out.items()}


summary = load_csv("summary.csv")
clf = load_json("classifier.json")
unet = load_json("unet_results.json")
kg_eval = load_csv("kg_eval.csv")


def pick(perc, sand, strat):
    r = summary[(summary.classifier == perc) & (np.isclose(summary.sand_fraction, sand)) & (summary.strategy == strat)]
    return r.iloc[0]


# headline numbers used in several places
if summary is not None:
    A50, D50, S50, R50 = (pick("Measured classifier", 0.5, v) for v in ["Adaptive", "Fixed detailed", "Fixed standard", "RL agent"])
    NL50 = pick("Measured classifier", 0.5, "Adaptive (no linger)")
    SAVE_E = 100 * (1 - A50.energy_kj_mean / D50.energy_kj_mean)
    SAVE_T = 100 * (1 - A50.time_min_mean / D50.time_min_mean)
    SAVE_P = 100 * (1 - A50.photos_mean / D50.photos_mean)
else:
    SAVE_E = SAVE_T = SAVE_P = float("nan")

# ------------------------------------------------------------------ top bar
html(S.brand())
if "nav" not in st.session_state:
    st.session_state["nav"] = "Home"
if "_page" not in st.session_state:
    st.session_state["_page"] = "Home"


def _keep_page():
    """Clicking the page you are already on would leave nothing selected; keep it selected."""
    if st.session_state.get("nav") is None:
        st.session_state["nav"] = st.session_state["_page"]


if hasattr(st, "pills"):          # Streamlit 1.40+: real pill buttons
    page = st.pills("Navigation", PAGES, key="nav", label_visibility="collapsed", on_change=_keep_page) or st.session_state["_page"]
else:
    page = st.radio("Navigation", PAGES, horizontal=True, key="nav", label_visibility="collapsed")
st.session_state["_page"] = page

# ================================================================== 1 HOME
if page == "Home":
    html(S.hero("Final Year Project · Coral reef monitoring",
                title="A survey robot that spends its battery", accent="where the reef needs it.",
                text="Over bleached coral it slows down and photographs in detail. Over healthy coral it cruises. "
                     "Over sand and rock it speeds up and saves power. Tested against fixed 'mow-the-lawn' surveys "
                     "on 90 simulated reefs, using the real error rates of the trained models.",
                chips=[("#ff4d6d", "Bleached coral"), ("#2dd4bf", "Healthy coral"), ("#8ea2d6", "Sand/rock"),
                       ("#7aa7ff", "Lawnmower survey path")]))
    if summary is not None:
        html(S.cards([(f"{A50.bleached_detail_pct_mean:.1f}%", "of bleached coral photographed in detail", "adaptive robot, 50% sand/rock"),
                      (f"−{SAVE_E:.0f}%", "energy", "vs a fixed survey with the same detail", "up"),
                      (f"−{SAVE_T:.0f}%", "survey time", "vs the same fixed survey", "up"),
                      (f"{100 * clf['independent_test']['accuracy']:.1f}%" if clf else "–", "classifier accuracy",
                       "on photos from a different source")], cols=4))
    c = st.columns([1.4, 1.3, 1.6, 2.4])
    c[0].button("Start the tour  →", on_click=go_to, args=("Classifier",))
    c[1].button("Jump to results", on_click=go_to, args=("Results",))
    if FEEDBACK_URL:
        c[2].link_button("Give feedback (UAT form)", FEEDBACK_URL)
        html(S.callout("<b>Taking part in testing?</b> Follow the suggested tour at the bottom of this page (about 10 minutes), "
                       "then open the <b>Give feedback</b> form and work through its tasks.", "good"))

    html(S.section("In one minute", "The problem, the idea and the result"))
    html(S.features([
        ("Problem", "Surveys waste effort on empty seabed",
         "Underwater robots usually move and photograph at one fixed rate, so as much battery and storage go on bare sand as on bleached coral."),
        ("Idea", "Look first, then decide",
         "A small on-board model recognises what is below the robot every metre, and the robot changes its speed and photo rate to match."),
        ("Result", "Detail where it matters, for less",
         (f"The adaptive robot captured {A50.bleached_detail_pct_mean:.0f}% of bleached coral in detail using {SAVE_E:.0f}% less energy "
          f"and {SAVE_T:.0f}% less time than a fixed survey with the same detail.") if summary is not None else ""),
    ], cols=3))

    html(S.section("Research questions", "Where each question is answered"))
    html(S.features([
        ("RQ1 · Steps 2 and 3", "How well can small models recognise reef health?",
         "Classifier: 87.1% on new photos, misses 1 in 4 bleached corals. Tiny U-Net: outlines sand, coral and bleaching pixel by pixel."),
        ("RQ2 · Steps 4, 6, 7", "How can that be turned into robot decisions?",
         "A three-mode rule with a linger step, camera + depth + position combined in one decision, a knowledge graph that "
         "double-checks reports, and an RL agent that learns its own rule."),
        ("RQ3 · Steps 5, 8", "How much time, energy and storage does it save?",
         "Results over 90 reefs, and a Robot Lab to test how the answer changes with different robot designs."),
    ], cols=3))

    html(S.section("The five robots", "Compared on every page"))
    html(S.robots([(n, S.VEHICLE_COLORS[n], d) for n, d in ROBOT_INFO]))

    html(S.section("Suggested tour", "About five minutes, in this order"))
    html(S.tour(["<b>Classifier</b>: why 100% was too good to be true, and the honest 87.1%",
                 "<b>Tiny U-Net</b>: the camera that tells sand from coral",
                 "<b>Simulation</b>: press Play and watch the robot work",
                 "<b>Results</b>: the trade-off chart over 90 reefs",
                 "<b>Knowledge Graph</b>: turning reports into hotspots",
                 "<b>RL Agent</b>: what the robot learned by itself",
                 "<b>Robot Lab</b>: change the robot and see what happens",
                 "<b>Glossary & Limits</b>: terms, assumptions and honest limits"]))

# ================================================================== 2 CLASSIFIER
elif page == "Classifier":
    header("Seeing the reef · RQ1", "MobileNetV3-Small: healthy or bleached?",
           "A small image model, trained on 9,662 coral photos, then checked honestly in two steps.")
    if not clf:
        st.warning("data/classifier.json not found.")
    else:
        t = clf["training"]; it = clf["independent_test"]; nc = clf["near_copy_check"]; pc = it["per_class"]
        html(S.explain(
            "A model that looks at one photo and says <b>healthy</b> or <b>bleached</b>. Each tab below is one step of how it was checked.",
            [f"It scored <b>100%</b> on its own test photos, which looked too good.",
             f"<b>{nc['test']['percent']}%</b> of those test photos were near-copies of training photos (frames from the same videos).",
             f"On photos from a different source it scores <b>{100 * it['accuracy']:.1f}%</b> and misses about <b>1 in 4</b> bleached corals.",
             "The simulation uses these real error rates, not the 100%."]))
        html(S.cards([(f"{clf['images']['train']:,}", "Training photos"), (f"{t['train_minutes']} min", "Training time", "on a T4 GPU"),
                      (f"{t['model_size_mb']} MB", "Model size", "small enough for a robot"), (f"{t['ms_per_image']} ms", "Time per photo", "on a GPU")], cols=4))
        tab1, tab2, tab3, tab3b, tab4 = st.tabs(["Step 1 · Looked perfect", "Step 2 · Why: near-copies", "Step 3 · The honest number",
                                                  "Step 4 · Cleaned and retrained", "Try your own photo"])
        with tab1:
            c = st.columns(2)
            with c[0]:
                plot(C.confusion(clf["kaggle_test"]["confusion_matrix"], clf["classes"], "Original test set: 257 of 257 correct"), key="c1")
                st.caption("How to read: rows are the true answer, columns are the model's answer. Everything on the diagonal is correct.")
            with c[1]:
                rj = load_json("results.json")
                if rj and rj.get("history"):
                    h = rj["history"]
                    plot(C.lines([x["epoch"] for x in h], {"Training": [x["train_acc"] for x in h], "Validation": [x["valid_acc"] for x in h]},
                                 "Training round (epoch)", "Accuracy", "Training curve"), key="c2")
                    st.caption("Training accuracy reaches almost 100% while validation stays near 98%: a sign of memorising.")
            html(S.callout("100% on real underwater photos is unusual, so Step 2 checks the test photos themselves.", "warn"))
        with tab2:
            html(S.cards([(f"{nc['test']['percent']}%", "of test photos had a near-copy in training", f"{nc['test']['near_copies']} of {nc['test']['total']}", "down"),
                          (f"{nc['valid']['percent']}%", "of validation photos had one", f"{nc['valid']['near_copies']} of {nc['valid']['total']}", "down"),
                          ("8", "versions checked per photo", "flipped and rotated"), (f"{nc['threshold_bits']} of 64", "fingerprint bits allowed to differ")], cols=4))
            p = os.path.join(FIGS, "near_copies.png")
            if os.path.exists(p):
                wide(st.image, p, caption="Top: a test or validation photo. Bottom: its near-copy found in the training photos.")
                html(S.callout("The pairs are frames from the same videos (note the TV watermarks and subtitles). Neighbouring frames "
                               "ended up in both training and test, so the 100% partly measures memory, not skill.", "bad"))
            else:
                html(S.callout("The matching pairs turned out to be <b>frames from the same videos</b>. Neighbouring frames ended up in "
                               "both training and test, so the 100% partly measures memory, not skill. (Example pairs are shown in the "
                               "dissertation; they are not published online because the frames come from TV broadcasts.)", "bad"))
        with tab3:
            c = st.columns(2)
            with c[0]:
                plot(C.confusion(it["confusion_matrix"], clf["classes"], f"A different dataset: {100 * it['accuracy']:.1f}% correct", "Oranges"), key="c3")
                st.caption("1,169 photos from another public dataset that the model never saw during training.")
            with c[1]:
                html(S.cards([(f"{100 * it['accuracy']:.1f}%", "Overall accuracy"),
                              (f"{100 * pc['Bleached']['recall']:.1f}%", "of bleached coral found", "about 1 in 4 missed", "down"),
                              (f"{100 * pc['Healthy']['recall']:.1f}%", "of healthy coral found", "", "up"),
                              (f"{100 * pc['Bleached']['precision']:.1f}%", "of its 'bleached' answers were right", "", "up")], cols=2))
        with tab3b:
            cr = load_json("clean_retrain.json")
            if not cr:
                html(S.callout("data/clean_retrain.json not found (run colab/clean_and_murky.py).", "info"))
            else:
                ct, ic, io = cr["clean_test"], cr["independent_clean_model"], cr["independent_original_model"]
                html(S.explain(
                    "To remove the shortcut (Gap 4), every photo was grouped with its near-copies, and each whole group was put in "
                    "<b>only one</b> of training, validation or test. Then the model was trained again from scratch.",
                    [f"The {cr['photos']:,} photos form only <b>{cr['groups']:,} groups</b>; the largest group alone has <b>{cr['largest_group']:,}</b> photos (one long video).",
                     f"With the stricter grouping, <b>{cr['original_test_sharing_group_with_train']['percent']}%</b> of the original test photos shared a group with training.",
                     f"On a clean test set the new model scores <b>{100 * ct['accuracy']:.1f}%</b>: an honest score for this dataset, instead of 100%.",
                     f"On the independent photos it scores <b>{100 * ic['accuracy']:.1f}%</b>, lower than the original model's <b>{100 * io['accuracy']:.1f}%</b>. "
                     "Removing the shortcut fixes the <i>measurement</i>, but cannot add the variety the dataset lacks."]))
                html(S.cards([(f"{cr['groups']:,}", "groups of near-copies", f"from {cr['photos']:,} photos"),
                              (f"{100 * ct['accuracy']:.1f}%", "clean test accuracy", "was 100% with shortcuts", "down"),
                              (f"{100 * ic['accuracy']:.1f}%", "on independent photos", f"original model: {100 * io['accuracy']:.1f}%", "down"),
                              (f"{100 * ic['per_class']['Bleached']['recall']:.1f}%", "of bleached found (independent)",
                               f"original model: {100 * io['per_class']['Bleached']['recall']:.1f}%", "down")], cols=4))
                c = st.columns(2)
                with c[0]:
                    plot(C.grouped(["Own test set", "Independent photos"],
                                   {"Original model": [100.0, 100 * io["accuracy"]], "Cleaned and retrained": [100 * ct["accuracy"], 100 * ic["accuracy"]]},
                                   "Accuracy: own test set vs new photos", "%", fmt="{:.1f}%"), key="cr1")
                with c[1]:
                    plot(C.confusion(ic["confusion_matrix"], clf["classes"], "Retrained model on independent photos", "Purples"), key="cr2")
                html(S.callout("<b>What this means:</b> the 100% was inflated by the shortcut; the honest in-dataset score is 97.4%. "
                               "Both models do worse on photos from other places, so the dataset needs more variety, not more of the same "
                               "videos. The simulation keeps the original model's error rates on independent photos (76% of bleaching "
                               "found), which are the ones measured on photos neither model was trained on.", "info"))
        with tab4:
            up = None
            if not HAS_TORCH or not os.path.exists(os.path.join(MODELS, "best_model.pt")):
                html(S.callout("Photo upload runs on the <b>offline (laptop) version</b> of this dashboard. The model needs PyTorch, "
                               "which is too large for the free online host.", "info"))
            else:
                up = st.file_uploader("Upload an underwater coral photo", type=["jpg", "jpeg", "png"])
            if up is not None:
                c = st.columns(2)
                c[0].image(up)
                try:
                    import torch, torch.nn as nn
                    from torchvision import models, transforms
                    from PIL import Image
                    mp = os.path.join(MODELS, "best_model.pt")
                    if not os.path.exists(mp):
                        c[1].warning("Copy best_model.pt into the models folder to enable this.")
                    else:
                        ck = torch.load(mp, map_location="cpu")
                        m = models.mobilenet_v3_small(weights=None)
                        m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(ck["classes"]))
                        m.load_state_dict(ck["state_dict"]); m.eval()
                        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                        with torch.no_grad():
                            pr = torch.softmax(m(tf(Image.open(up).convert("RGB")).unsqueeze(0)), 1)[0].numpy()
                        b = int(pr.argmax())
                        with c[1]:
                            html(S.cards([(ck["classes"][b], "Prediction", f"{100 * pr[b]:.1f}% confident")], cols=1))
                            plot(C.bars(ck["classes"], [100 * float(v) for v in pr], "Confidence", "{:.1f}", "%", height=280), key="c5")
                except ImportError:
                    c[1].warning("Install PyTorch first: python -m pip install torch torchvision pillow")
    next_button()

# ================================================================== 3 U-NET
elif page == "Tiny U-Net":
    header("Seeing the reef · RQ1", "Tiny U-Net: colouring every pixel",
           "Trained on Coralscapes, 2,075 Red Sea reef photos where every pixel is labelled.")
    if not unet:
        html(S.callout("The Tiny U-Net results are not in data/ yet.", "warn"))
    else:
        html(S.explain(
            "Instead of one label per photo, this model colours <b>every pixel</b> as sand/rock, healthy coral, bleached coral or other. "
            "The robot then turns each photo into one decision.",
            [f"It lets the robot tell <b>sand/rock from coral with the camera alone</b>, with no extra sensor.",
             f"Overall it overlaps the true outlines by <b>{100 * unet['test_mean_iou']:.0f}%</b> (mean IoU).",
             f"Bleached coral is the hardest class (Dice <b>{unet['test_dice']['Bleached coral']:.2f}</b>) because only 1% of pixels are bleached.",
             f"It is tiny: <b>{unet['model_size_mb']} MB</b>, {unet['ms_per_image_gpu']} ms per photo on a GPU."]))
        p = os.path.join(FIGS, "unet_examples.png")
        tabs = st.tabs(["Examples", "Scores per class", "Where pixels go wrong", "One decision per photo"])
        with tabs[0]:
            html(S.legend([("#4d4d4d", "Other"), (S.SAND, "Sand/rock"), (S.HEALTHY, "Healthy coral"), ("#ffffff", "Bleached coral")]))
            if os.path.exists(p):
                wide(st.image, p, caption="Left: photo. Middle: the true outline drawn by people. Right: what the Tiny U-Net drew.")
            else:
                html(S.callout("Example outlines are shown in the dissertation.", "info"))
        with tabs[1]:
            cl = unet["classes"]
            plot(C.grouped(cl, {"IoU": [unet["test_iou"][k] for k in cl], "Dice": [unet["test_dice"][k] for k in cl]},
                           "Test scores per class (1.0 = perfect outline)", "Score"), key="u1")
            st.caption("IoU and Dice both measure how much the drawn outline overlaps the true one. See the Glossary for details.")
        with tabs[2]:
            plot(C.confusion(unet["pixel_confusion"], unet["classes"], "Where each true class ends up (% of its pixels)", "Greens", normalise=True), key="u2")
            st.caption("How to read: each row adds up to 100%. The diagonal is correct; the rest shows what it was confused with.")
        with tabs[3]:
            c = st.columns(2)
            with c[0]:
                plot(C.confusion(unet["photo_confusion"], ["Sand/rock", "Healthy", "Bleached"],
                                 f"Photo decisions: {100 * unet['photo_accuracy']:.1f}% correct", "Purples"), key="u3")
            with c[1]:
                rule = unet["photo_rule"]
                html(S.callout(f"<b>How a photo becomes a decision:</b> <i>bleached</i> if at least {int(100 * rule['bleached_min_share'])}% of its "
                               f"seabed is bleached; otherwise <i>healthy</i> if at least {int(100 * rule['coral_min_share'])}% is coral; "
                               "otherwise <i>sand/rock</i>. This table drives the camera-only robot in the simulation.", "info"))
                html(S.callout("Only 27 test photos count as bleached, so the bleached row is uncertain.", "warn"))
    next_button()

# ================================================================== 4 SIMULATION
elif page == "Simulation":
    header("Deciding · RQ2", "Watch the robot survey a reef",
           "A 100 m × 80 m reef surveyed in 40 back-and-forth lanes. Change the settings to make a new reef.")
    html(S.explain(
        "One random reef and one robot. <b>Survey playback</b> shows the speed mode it chose in each metre; "
        "<b>Explore the reef</b> shows the true seabed, and hovering shows what the robot thought it saw.",
        ["Choose <b>Adaptive</b> or <b>RL agent</b>: red (slow, detailed) appears over the bleached patches, grey (fast) over sand.",
         "The fixed robots look the same everywhere, which is exactly their weakness.",
         "Every robot sees the same reef and makes the same perception mistakes, so the comparison is fair."]))
    c = st.columns([1.1, 1.7, 0.9, 0.9, 1])
    sand = c[0].slider("Sand/rock share", 0.1, 0.9, 0.5, 0.05, help="How much of the seabed is empty (no coral)")
    perc = perception_box(c[1], "sim_p")
    seed = int(c[2].number_input("Reef number", 0, 10_000, 7, 1, help="Change to generate a different random reef"))
    linger = c[3].slider("Linger (m)", 0, 8, 3, help="How many metres the adaptive robot stays slow after seeing bleaching")
    sensor = c[4].slider("Sensor accuracy", 0.80, 1.00, 0.97, 0.01, disabled=(perc in (rs.UNET_NAME, rs.FUSED_NAME)),
                         help="Simulated sensor that says coral or not. Only used with the classifier; the camera-only "
                              "and fused robots work out coral vs sand themselves.")
    reef, rows, cols, cells, seen, logs, mets = simulate(sand, perc, seed, linger, sensor)
    vehicle = st.radio("Robot", VEHICLES, index=2, horizontal=True, key="veh")
    st.caption(dict(ROBOT_INFO)[vehicle])
    m = mets[vehicle]
    html(S.cards([(f"{m['time_min']:.0f} min", "Survey time"), (f"{m['energy_kj']:.0f} kJ", "Energy used"),
                  (f"{m['photos']:,}", "Photos taken", f"{m['storage_gb']:.1f} GB"),
                  (f"{m['bleached_detail_pct']:.1f}%", "Bleached coral in detail", "at least 4 photos per metre"),
                  (f"{m['photos_on_sand_pct']:.0f}%", "Photos spent on sand/rock", "lower is better")], cols=5))
    t1, t2 = st.tabs(["▶  Survey playback", "🔍  Explore the reef"])
    with t1:
        html(S.legend([(S.MODE_COLORS["fast"], "Fast (1.5 m/s, few photos)"), (S.MODE_COLORS["normal"], "Normal (1.0 m/s)"),
                       (S.MODE_COLORS["detail"], "Detail (0.5 m/s, many photos)"), ("#ffd60a", "Robot")]))
        plot(C.playback(reef, rows, cols, logs[vehicle]["modes"], f"{vehicle}: speed mode chosen in each metre"), key="pb")
    with t2:
        html(S.legend([(S.SAND, "Sand/rock"), (S.HEALTHY, "Healthy coral"), ("#ffffff", "Bleached coral")]))
        plot(C.reef_heatmap(reef, rows, cols, seen, logs[vehicle], "The true reef: hover over any square"), key="rh")
        true_b = int((cells == rs.BLEACHED).sum()); found = int(((seen == rs.BLEACHED) & (cells == rs.BLEACHED)).sum())
        html(S.callout(f"This reef has <b>{true_b}</b> bleached metres. The robot's perception spotted <b>{found}</b> of them "
                       f"(<b>{100 * found / max(true_b, 1):.1f}%</b>). Lingering and the RL agent recover many of the rest.", "info"))
    html(S.section("Race", "All five robots on this reef", "Pick what to compare. The lines grow as each robot covers the reef."))
    metric = st.radio("Compare", ["Energy", "Time", "Photos"], horizontal=True, key="race")
    key, lab, scale = {"Energy": ("energy_j", "Energy (kJ)", 1 / 1000), "Time": ("time_s", "Time (min)", 1 / 60),
                       "Photos": ("photos", "Photos", 1)}[metric]
    c = st.columns([1.6, 1])
    with c[0]:
        plot(C.cumulative(logs, key, lab, f"{lab} used so far", scale), key="cum")
    with c[1]:
        plot(C.bars(VEHICLES, [mets[v]["bleached_detail_pct"] for v in VEHICLES], "Bleached coral in detail", "{:.0f}", "%", height=360), key="race2")
    next_button()

# ================================================================== 5 RESULTS
elif page == "Results":
    header("Results · RQ3", "90 reefs, five robots, three ways of seeing",
           "30 random reefs at each of three sand/rock shares. Every number is an average over 30 reefs.")
    if summary is None:
        st.warning("Run `python run_experiments.py` first.")
    else:
        html(S.explain(
            "Averages over many reefs, so the conclusions do not depend on one lucky map.",
            [f"The adaptive robot photographs <b>{A50.bleached_detail_pct_mean:.1f}%</b> of bleached coral in detail, close to the slow survey's 100%, "
             f"with <b>{SAVE_E:.0f}% less energy</b>, <b>{SAVE_T:.0f}% less time</b> and <b>{SAVE_P:.0f}% fewer photos</b>.",
             f"Compared with the standard survey it uses <b>{A50.energy_kj_vs_standard_pct:+.0f}%</b> energy, but the standard survey photographs no bleaching in detail.",
             f"Without lingering, detail coverage drops to <b>{NL50.bleached_detail_pct_mean:.0f}%</b>.",
             f"The RL agent finds the best balance: <b>{R50.bleached_detail_pct_mean:.0f}%</b> detail at <b>{R50.energy_kj_vs_standard_pct:+.0f}%</b> energy vs standard."]))
        perc = perception_box(st.columns([1, 1])[0], "res_p", list(summary.classifier.unique()))
        html(S.section("Trade-off", "Energy against detail",
                       "Each line is one robot, with a dot for each sand/rock share (30%, 50%, 70%). "
                       "The fixed robots behave the same on every reef, so they are single dots. Up and to the left is best."))
        plot(C.tradeoff(summary, perc, labels=PLABEL), key="to")
        st.caption("Tip: the names on the right are the key. Click a name to hide or show that robot; double-click to show only that one. "
                   "Hover over a dot for its numbers.")
        html(S.section("Compare", "All five robots at one sand/rock share", "Choose how much of the seabed is sand/rock."))
        sand = st.columns([1, 1])[0].select_slider("Sand/rock share", options=[0.3, 0.5, 0.7], value=0.5,
                                                   format_func=lambda v: f"{int(v * 100)}%")
        rr = {v: pick(perc, sand, v) for v in VEHICLES}
        c = st.columns(4)
        for col, (mk, lab, fmt, suf) in zip(c, [("time_min", "Survey time (min)", "{:,.0f}", ""), ("energy_kj", "Energy (kJ)", "{:,.0f}", ""),
                                                ("photos", "Photos", "{:,.0f}", ""), ("bleached_detail_pct", "Bleached in detail", "{:.1f}", "%")]):
            with col:
                plot(C.bars(VEHICLES, [rr[v][mk + "_mean"] for v in VEHICLES], lab, fmt, suf,
                            errors=[rr[v][mk + "_sd"] for v in VEHICLES]), key="r" + mk)
        st.caption("Thin lines on the bars show how much results vary between reefs (one standard deviation).")
        with st.expander("Show the numbers as a table"):
            table(pd.DataFrame([{"Robot": v, "Time (min)": f"{r.time_min_mean:.1f} ± {r.time_min_sd:.1f}",
                                 "Energy (kJ)": f"{r.energy_kj_mean:.1f} ± {r.energy_kj_sd:.1f}", "Photos": f"{r.photos_mean:,.0f}",
                                 "Bleached in detail": f"{r.bleached_detail_pct_mean:.1f}%", "Photos on sand": f"{r.photos_on_sand_pct_mean:.1f}%",
                                 "Energy vs standard": f"{r.energy_kj_vs_standard_pct:+.1f}%", "Reward": f"{r.reward_mean:.0f}"} for v, r in rr.items()]))
        html(S.section("Sand effect", "The emptier the seabed, the more the adaptive robots save"))
        xs = [30, 50, 70]
        c = st.columns(2)
        with c[0]:
            plot(C.lines(xs, {v: [pick(perc, s, v).energy_kj_vs_standard_pct for s in (0.3, 0.5, 0.7)] for v in
                              ["Fixed detailed", "Adaptive", "RL agent"]}, "Sand/rock share (%)", "% compared with standard",
                         "Energy compared with the standard survey (0 = same)"), key="se")
        with c[1]:
            plot(C.lines(xs, {v: [pick(perc, s, v).bleached_detail_pct_mean for s in (0.3, 0.5, 0.7)] for v in
                              ["Fixed detailed", "Adaptive", "Adaptive (no linger)", "RL agent"]}, "Sand/rock share (%)", "%",
                         "Bleached coral photographed in detail"), key="sd")
        fa = load_csv("fusion_ablation.csv")
        if fa is not None:
            html(S.section("Combining sensors · Gap 2", "Camera, depth and position in one decision",
                           "Each metre, the robot combines three clues with Bayes' rule: what the camera (Tiny U-Net) sees, how high the "
                           "seabed is (a depth sensor: coral stands up, sand is flat, but rock stands up too), and what it already "
                           "decided next door, because reef types come in patches."))
            g = fa[np.isclose(fa.sand_fraction, sand)].groupby("variant").mean(numeric_only=True).reindex(FU_VARIANTS)
            cam, full = g.loc["Camera only"], g.loc["Camera + depth + position"]
            html(S.explain(
                f"Four versions of the robot on the same {int(sand * 100)}% sand/rock reefs, with identical camera mistakes. "
                "Only the clues they combine differ.",
                [f"Every metre is labelled correctly <b>{100 * cam.accuracy:.0f}% → {100 * full.accuracy:.0f}%</b> of the time when all three clues are combined.",
                 f"Sand is told apart from coral <b>{100 * cam.coral_vs_sand_accuracy:.0f}% → {100 * full.coral_vs_sand_accuracy:.0f}%</b> of the time.",
                 f"Bleached coral found: <b>{100 * cam.bleached_recall:.0f}% → {100 * full.bleached_recall:.0f}%</b>, mainly thanks to the position clue.",
                 "Depth alone cannot see bleaching at all: colour is essential, depth and position sharpen it."]))
            labels = list(g.index)
            colors = {"Camera only": "#8e8e93", "Depth only": "#c7c7cc", "Camera + depth": "#7aa7ff", "Camera + depth + position": "#1259ff"}
            for k_, v_ in colors.items():
                S.VEHICLE_COLORS.setdefault(k_, v_)
            c = st.columns(3)
            with c[0]:
                plot(C.bars(labels, list(100 * g.accuracy), "Metres labelled correctly", "{:.0f}", "%"), key="fu1")
            with c[1]:
                plot(C.bars(labels, list(100 * g.bleached_recall), "Bleached coral found", "{:.0f}", "%"), key="fu2")
            with c[2]:
                plot(C.bars(labels, list(g.false_alarms_per_100m), "False bleaching alarms per 100 m", "{:.1f}"), key="fu3")
            with st.expander("Survey results for each version (adaptive robot)"):
                table(pd.DataFrame({"Robot sees with": labels,
                                    "Energy (kJ)": [f"{v:.0f}" for v in g.adaptive_energy_kj],
                                    "Bleached in detail": [f"{v:.1f}%" for v in g.adaptive_bleached_detail_pct],
                                    "Photos": [f"{v:,.0f}" for v in g.adaptive_photos],
                                    "Bleached in detail (no linger)": [f"{v:.1f}%" for v in g.no_linger_bleached_detail_pct]}))
            st.caption("The depth sensor, rock heights and the strength of the position clue are assumptions; the position "
                       "strength was tuned on 18 separate practice reefs, not on these test reefs.")
        rob, mk = load_json("robustness.json"), load_csv("murky_survey.csv")
        if rob and mk is not None:
            html(S.section("Murky water · Problem 3", "What happens when the water gets cloudy",
                           "The test photos were made murky step by step (red light lost, blue-green haze, darker, grainy, blurred), "
                           "then both models were tested again. Their new error rates were then given to the robot."))
            names = rob["level_names"]
            m_ = mk[np.isclose(mk.sand_fraction, sand)].groupby(["variant", "level"]).mean(numeric_only=True)
            cam_d = [m_.loc[("Camera only", l)].bleached_detail_pct for l in rob["levels"]]
            fus_d = [m_.loc[("Camera + depth + position", l)].bleached_detail_pct for l in rob["levels"]]
            html(S.explain(
                "Murky water hurts both models, and simple colour correction makes it <b>worse</b>, not better.",
                [f"The classifier finds <b>{100 * rob['classifier']['original_raw'][0]['bleached_recall']:.0f}%</b> of bleached coral in clear water, "
                 f"<b>{100 * rob['classifier']['original_raw'][2]['bleached_recall']:.0f}%</b> when murky and <b>{100 * rob['classifier']['original_raw'][4]['bleached_recall']:.0f}%</b> when extremely murky.",
                 "Colour correction (grey-world white balance) removes the white-blue look that marks bleaching, so the models miss even more.",
                 f"In the survey, combining sensors keeps the robot useful up to 'Murky': <b>{fus_d[2]:.0f}%</b> of bleached coral in detail, against "
                 f"<b>{cam_d[2]:.0f}%</b> with the camera alone.",
                 f"Beyond that both fail ({fus_d[3]:.0f}% and {cam_d[3]:.0f}%): the robot should report that the water is too murky rather than trust its camera."]))
            lev = st.select_slider("Water clarity", names, value=names[2], key="murk")
            i = names.index(lev)
            co, u = rob["classifier"]["original_raw"][i], rob["unet"]["raw"][i]
            html(S.cards([(f"{100 * co['bleached_recall']:.0f}%", "Classifier: bleached found", f"accuracy {100 * co['accuracy']:.0f}%"),
                          (f"{u['mean_iou']:.2f}", "Tiny U-Net mean IoU", f"bleached Dice {u['bleached_dice']:.2f}"),
                          (f"{cam_d[i]:.0f}%", "Bleached in detail, camera only", "adaptive robot", "down" if cam_d[i] < 80 else "up"),
                          (f"{fus_d[i]:.0f}%", "Bleached in detail, fused", "adaptive robot", "down" if fus_d[i] < 80 else "up")], cols=4))
            c = st.columns(2)
            with c[0]:
                plot(C.lines(names, {"Classifier": [100 * d["bleached_recall"] for d in rob["classifier"]["original_raw"]],
                                     "Classifier, colour-corrected": [100 * d["bleached_recall"] for d in rob["classifier"]["original_corrected"]],
                                     "Tiny U-Net": [100 * d["bleached_dice"] for d in rob["unet"]["raw"]],
                                     "Tiny U-Net, colour-corrected": [100 * d["bleached_dice"] for d in rob["unet"]["corrected"]]},
                             "", "%", "Models: bleached coral found (classifier) and outlined (U-Net Dice)", height=380,
                             colors={"Classifier": "#1259ff", "Classifier, colour-corrected": "#9db8ff",
                                     "Tiny U-Net": "#ff3b30", "Tiny U-Net, colour-corrected": "#ffaaa5"}), key="mk1")
            with c[1]:
                plot(C.lines(names, {"Camera only": cam_d, "Camera + depth + position": fus_d}, "", "%",
                             f"Survey: bleached coral photographed in detail ({int(sand * 100)}% sand)", height=380,
                             colors={"Camera only": "#8e8e93", "Camera + depth + position": "#1259ff"}), key="mk2")
            with st.expander("Survey numbers at every clarity level"):
                table(pd.DataFrame([{"Water": names[j], "Robot sees with": v,
                                     "Metres labelled correctly": f"{100 * m_.loc[(v, l)].accuracy:.0f}%",
                                     "Bleached found": f"{100 * m_.loc[(v, l)].bleached_recall:.0f}%",
                                     "False alarms / 100 m": f"{m_.loc[(v, l)].false_alarms_per_100m:.1f}",
                                     "Energy (kJ)": f"{m_.loc[(v, l)].energy_kj:.0f}",
                                     "Bleached in detail": f"{m_.loc[(v, l)].bleached_detail_pct:.0f}%"}
                                    for j, l in enumerate(rob["levels"]) for v in ["Camera only", "Camera + depth + position"]]))
            st.caption("Murky water is simulated on real test photos (classifier: 1,169 independent photos; U-Net: Coralscapes test photos). "
                       "Real turbidity varies more than this. The slight rise for the classifier at 'Slightly murky' is probably because "
                       "its training videos were themselves blue-green.")
    ho = load_json("holoocean_summary.json")
    if ho:
        R_ = ho["robots"]
        html(S.section("HoloOcean check · Objective iii", "Does it still work with a real-feeling vehicle?",
                       "The 2-D simulation assumes the robot changes speed instantly. Here a HoloOcean HoveringAUV, with mass, drag, "
                       f"buoyancy and thrusters, flew {ho['reef_metres']} m of one test reef ({ho['lanes']} lanes, {int(ho['sand_fraction'] * 100)}% sand) "
                       "with the same rule and the same recognition mistakes."))
        ad, nl = R_["Adaptive"], R_["Adaptive (no linger)"]
        html(S.explain(
            "The main result holds: the adaptive robot photographs almost the same share of bleached coral in detail as in the 2-D simulation.",
            [f"Bleached coral in detail: <b>{ad['holoocean']['bleached_detail_pct']:.1f}%</b> in HoloOcean vs {ad['simulation_2d']['bleached_detail_pct']:.1f}% in 2-D "
             f"(no linger: {nl['holoocean']['bleached_detail_pct']:.1f}% vs {nl['simulation_2d']['bleached_detail_pct']:.1f}%).",
             f"Slowing down is not instant: on average <b>{ad['holoocean']['slowdown_metres_mean']:.1f} m</b> ({ad['holoocean']['slowdown_seconds_mean']:.1f} s) "
             f"to drop to detail speed, up to {ad['holoocean']['slowdown_metres_max']:.1f} m. Photos are still taken at detail spacing, so the photo count holds, "
             "but the vehicle is moving faster than planned for the first few metres.",
             f"Against the fixed detailed survey it uses <b>{100 * (1 - ad['holoocean']['energy_kj'] / R_['Fixed detailed']['holoocean']['energy_kj']):.0f}% less energy</b> and "
             f"{100 * (1 - ad['holoocean']['photos'] / R_['Fixed detailed']['holoocean']['photos']):.0f}% fewer photos (2-D: "
             f"{100 * (1 - ad['simulation_2d']['energy_kj'] / R_['Fixed detailed']['simulation_2d']['energy_kj']):.0f}% and "
             f"{100 * (1 - ad['simulation_2d']['photos'] / R_['Fixed detailed']['simulation_2d']['photos']):.0f}%).",
             f"The vehicle cruised at about {ho['reef_metres'] / R_['Fixed standard']['holoocean']['time_min'] / 60:.2f} m/s when asked for 1.0 m/s, and stayed within "
             f"{max(r['holoocean']['max_path_error_m'] for r in R_.values()):.1f} m of its line (lanes are 2 m apart)."]))
        names_ = list(R_)
        c = st.columns(2)
        with c[0]:
            plot(C.grouped(names_, {"2-D simulation": [R_[n]["simulation_2d"]["bleached_detail_pct"] for n in names_],
                                    "HoloOcean": [R_[n]["holoocean"]["bleached_detail_pct"] for n in names_]},
                           "Bleached coral photographed in detail", "%", fmt="{:.0f}%"), key="ho1")
        with c[1]:
            plot(C.grouped(names_, {"2-D simulation": [R_[n]["simulation_2d"]["energy_kj"] for n in names_],
                                    "HoloOcean": [R_[n]["holoocean"]["energy_kj"] for n in names_]},
                           "Energy over the reef lanes", "kJ", fmt="{:.0f}"), key="ho2")
        with st.expander("All HoloOcean numbers"):
            table(pd.DataFrame([{"Robot": n, "Where": w, "Time (min)": f"{R_[n][k]['time_min']:.1f}", "Energy (kJ)": f"{R_[n][k]['energy_kj']:.1f}",
                                 "Photos": f"{R_[n][k]['photos']:,}", "Bleached in detail": f"{R_[n][k]['bleached_detail_pct']:.1f}%",
                                 "Photos over sand": f"{R_[n][k]['photos_on_sand_pct']:.1f}%"}
                                for n in names_ for w, k in [("2-D", "simulation_2d"), ("HoloOcean", "holoocean")]]))
        st.caption(f"One reef, one run per robot (HoloOcean {ho['world']} world, {ho['ticks_per_sec']} physics steps per second). Time and energy are over the "
                   "reef lanes; turns added about 8 to 25 s per robot. Energy uses the same model as the 2-D simulation, applied to the speeds HoloOcean "
                   "actually produced. Recognition mistakes still come from the measured error rates, not from HoloOcean's camera.")
    next_button()

# ================================================================== 6 KNOWLEDGE GRAPH
elif page == "Knowledge Graph":
    header("Deciding · RQ2", "Knowledge graph: turning reports into hotspots",
           "Each bleaching report is a dot; reports close together are linked.")
    html(S.explain(
        "Every place the robot reported bleaching becomes a dot. Dots within a set distance are joined by lines. "
        "A dot with enough neighbours is <b>confirmed</b> (red); an isolated one is <b>unconfirmed</b> (orange) and is probably a false alarm.",
        ["Real bleaching comes in patches, so real reports have neighbours. False alarms are scattered and isolated.",
         "This simple check raises the share of correct reports from about <b>87% to 97%</b> over 90 reefs.",
         "Groups of linked dots become a ranked <b>hotspot list</b>: where to send a diver first.",
         "Move the two sliders on the right to see how the rule changes the result."]))
    c = st.columns([1, 0.8, 1.6, 1, 1])
    sand = c[0].slider("Sand/rock share", 0.1, 0.9, 0.5, 0.05, key="kg_s")
    seed = int(c[1].number_input("Reef number", 0, 10_000, 7, 1, key="kg_seed"))
    perc = perception_box(c[2], "kg_p", [p for p in PERCEPTIONS if p != "Perfect classifier"], index=0)
    radius = c[3].slider("Link distance (m)", 1.0, 4.0, kg.LINK_RADIUS_M, 0.1, help="Reports closer than this are linked")
    mindeg = c[4].slider("Links needed to confirm", 1, 5, kg.MIN_DEGREE, help="A report needs this many linked neighbours to be trusted")
    reef, rows, cols, cells, seen, logs, mets = simulate(sand, perc, seed, 3, 0.97)
    G = kg.build_graph(reef, rows, cols, seen, radius=radius, min_degree=mindeg)
    ev = kg.evaluate(reef, G)
    if ev:
        html(S.cards([(ev["detections"], "Bleaching reports"),
                      (f"{100 * ev['precision_all']:.1f}%", "Correct before checking"),
                      (f"{100 * ev['precision_confirmed']:.1f}%", "Correct after checking", f"{100 * (ev['precision_confirmed'] - ev['precision_all']):+.1f} points", "up"),
                      (f"{100 * ev['false_alarms_flagged']:.0f}%" if ev["false_alarms"] else "–", "False alarms caught"),
                      (f"{100 * ev['true_detections_kept']:.0f}%", "Real reports kept"),
                      (f"{ev['hotspots_confirmed']} / {ev['true_patches']}", "Hotspots found / true patches")], cols=6))
        plot(C.kg_network(reef, G, "Bleaching reports and their links (hover over a dot)"), key="kgn")
        hs = kg.hotspots(G.subgraph([n for n in G.nodes if G.nodes[n]["confirmed"]]))
        if hs:
            html(S.section("Hotspot list", "Where to send a diver first", "Ranked by number of linked reports."))
            table(pd.DataFrame(hs).rename(columns={"hotspot": "Hotspot", "detections": "Reports", "centre_x_m": "Along lane (m)",
                                                   "centre_y_m": "Across (m)", "lanes": "Lanes", "mean_degree": "Average links",
                                                   "share_truly_bleached": "Truly bleached"}).head(15))
    if kg_eval is not None:
        with st.expander("Results over all 90 reefs (default rule: 2.3 m links, 2 links to confirm)"):
            g = kg_eval.groupby("sand_fraction").mean(numeric_only=True)
            table(pd.DataFrame({"Sand/rock": [f"{int(s * 100)}%" for s in g.index],
                                "Correct before": [f"{100 * v:.1f}%" for v in g.precision_all],
                                "Correct after": [f"{100 * v:.1f}%" for v in g.precision_confirmed],
                                "False alarms caught": [f"{100 * v:.1f}%" for v in g.false_alarms_flagged],
                                "Real reports kept": [f"{100 * v:.1f}%" for v in g.true_detections_kept],
                                "Bleached patches found": [f"{v:.1f}%" for v in g.patches_found_pct],
                                "Patches of 5 m or more found": [f"{v:.1f}%" for v in g.patches_5plus_found_pct]}))
    next_button()

# ================================================================== 7 RL AGENT
elif page == "RL Agent":
    header("Deciding · RQ2", "An agent that learns when to slow down",
           "Instead of following our hand-written rule, this robot works out its own rule by trial and reward.")
    html(S.explain(
        "After every metre the agent gets a score: <b>value of what it mapped − energy used − photos taken</b> "
        "(R = α·A − β·E − γ·S). Over many practice reefs it learns which speed earns the best score in each situation.",
        ["It learns quickly: its score stops improving after about <b>1,500 metres</b> of practice.",
         "It slows down for bleaching, but only <b>stays slow</b> when the previous lane also had bleaching nearby. Nobody told it to do this.",
         f"It earns the <b>highest score</b> of all five robots, using less energy than the hand-written rule."]))
    opts = [p for p in PERCEPTIONS if p != "Perfect classifier"]
    perc = perception_box(st.columns([1.5, 2])[0], "rl_p", opts, index=0)
    tag = RL_TAG.get(perc, "")
    Q, N = load_rl(perc)
    curve = load_csv(f"rl_curve{tag}.csv")
    pt = rl.policy_table(Q, N)
    labels = [f"{r['situation']} · {'bleaching on neighbour lane' if 'no bleaching' not in r['previous_lane'] else 'neighbour lane clear'}" for r in pt]
    c = st.columns(2)
    with c[0]:
        if curve is not None:
            plot(C.lines(curve.steps_trained.clip(lower=1), {"Score per metre": curve.validation_reward_per_m},
                         "Practice so far (metres, log scale)", "Score per metre", "Learning curve: higher is better", log_x=True,
                         colors={"Score per metre": "#00a383"}), key="rl1")
    with c[1]:
        if summary is not None and perc in set(summary.classifier):
            rr = {v: pick(perc, 0.5, v) for v in VEHICLES}
            plot(C.bars(VEHICLES, [rr[v].reward_mean for v in VEHICLES], "Total score on test reefs (higher is better)", "{:,.0f}"), key="rl2")
            st.caption("Every metre costs energy and photos, so all scores are below zero. Closer to zero is better.")
    seen_rows = [i for i, r in enumerate(pt) if r["times_seen"] > 0]
    plot(C.q_table(Q[seen_rows], [labels[i] for i in seen_rows], "What the agent learned: average score of each action"), key="rl3")
    st.caption("How to read: one row per situation. Blue = good average score, red = bad. ★ marks the action the agent chooses.")
    html(S.section("Try it", "Train a new agent with your own priorities",
                   "Make bleached coral more valuable, or energy more expensive, and see how the learned behaviour changes."))
    c = st.columns(4)
    w = dict(rl.REWARD)
    w["value_bleached"] = c[0].slider("Value of bleached coral in detail", 0.0, 3.0, 1.0, 0.1)
    w["value_healthy"] = c[1].slider("Value of healthy coral mapped", 0.0, 1.0, 0.2, 0.05)
    w["beta"] = c[2].slider("Cost of energy (per kJ)", 0.0, 10.0, 3.0, 0.5)
    w["gamma"] = c[3].slider("Cost per photo", 0.0, 0.1, 0.01, 0.005)
    if st.button("Train my agent  →"):
        bar = st.progress(0.0)
        Q2, N2, _ = rl.train(n_reefs=12, w=w, perception=perc, unet_M=UNET_M, progress=lambda i, n: bar.progress(i / n))
        res = {}
        for sf, k, seed in rs.map_seeds([0.5], 10):
            reef = rs.make_reef(np.random.default_rng(seed), sf)
            rows, cols, cells = rs.lawnmower_path(reef)
            seen = rs.perceive(cells, np.random.default_rng(seed + 7), perc, UNET_M,
                               ctx=dict(reef=reef, rows=rows, cols=cols, seed=seed))
            nb = rs.neighbour_feature(rows, cols, seen)
            for name, p in [("Your agent", rl.QPolicy(Q2)), ("Adaptive", rs.RulePolicy(3)), ("Fixed standard", rs.FixedPolicy(rs.NORMAL))]:
                res.setdefault(name, []).append(rs.metrics(cells, rs.run_survey(cells, seen, p, nb)))
        mean = {k: {m: np.mean([d[m] for d in v]) for m in v[0]} for k, v in res.items()}
        S.VEHICLE_COLORS.setdefault("Your agent", "#ff9f0a")
        names = list(mean)
        c = st.columns(3)
        with c[0]:
            plot(C.bars(names, [mean[n]["energy_kj"] for n in names], "Energy (kJ)", "{:,.0f}"), key="t1")
        with c[1]:
            plot(C.bars(names, [mean[n]["photos"] for n in names], "Photos", "{:,.0f}"), key="t2")
        with c[2]:
            plot(C.bars(names, [mean[n]["bleached_detail_pct"] for n in names], "Bleached in detail", "{:.0f}", "%"), key="t3")
        pt2 = rl.policy_table(Q2, N2)
        sr = [i for i, r in enumerate(pt2) if r["times_seen"] > 0]
        plot(C.q_table(Q2[sr], [labels[i] for i in sr], "What your agent learned"), key="t4")
    next_button()

# ================================================================== 8 ROBOT LAB
elif page == "Robot Lab":
    header("What if? · RQ3", "Design your own survey robot",
           "Change the robot and the three robots are re-run on 5 reefs straight away.")
    html(S.explain(
        "The study's robot settings are assumptions, not measurements from a real vehicle. This page lets you test whether the "
        "conclusions still hold if those assumptions change.",
        ["The curve shows why slowing down is expensive: the computer and sensors draw power every second, "
         "so a slow robot pays that 'hotel' cost for longer on each metre.",
         "Try lowering <b>hotel power</b> to 10 W: slowing down becomes cheap and the adaptive robot saves more.",
         "Try raising <b>energy per photo</b>: skipping photos over sand becomes the biggest saving."]))
    c = st.columns(3)
    with c[0]:
        st.markdown("**Fast mode** · over sand/rock")
        fs = st.slider("Speed (m/s)", 0.3, 2.5, 1.5, 0.1, key="fs")
        fp = st.slider("Metres between photos", 0.25, 4.0, 2.0, 0.25, key="fp")
    with c[1]:
        st.markdown("**Normal mode** · over healthy coral")
        ns = st.slider("Speed (m/s)", 0.3, 2.5, 1.0, 0.1, key="ns")
        npp = st.slider("Metres between photos", 0.25, 2.0, 0.5, 0.25, key="np")
    with c[2]:
        st.markdown("**Detail mode** · over bleached coral")
        ds = st.slider("Speed (m/s)", 0.2, 2.0, 0.5, 0.1, key="ds")
        dp = st.slider("Metres between photos", 0.125, 1.0, 0.25, 0.125, key="dp")
    c = st.columns(5)
    hotel = c[0].slider("Hotel power (W)", 5.0, 150.0, 40.0, 5.0, help="Power for the computer and sensors, used every second")
    drag = c[1].slider("Drag coefficient", 2.0, 40.0, 12.0, 1.0, help="Power to push through water = drag × speed³")
    pe = c[2].slider("Energy per photo (J)", 0.0, 40.0, 10.0, 1.0, help="Strobe light, capture and saving one photo")
    sand = c[3].slider("Sand/rock share", 0.1, 0.9, 0.5, 0.1, key="lab_sand")
    linger = c[4].slider("Linger (m)", 0, 8, 3, key="lab_linger")
    if dp > npp or npp > fp:
        html(S.callout("Tip: detail mode normally takes photos <b>closer together</b> than normal mode, and normal closer than fast.", "warn"))
    if ds > ns or ns > fs:
        html(S.callout("Tip: detail mode is normally the <b>slowest</b> and fast mode the quickest.", "warn"))
    res = robot_lab((fs, fp, ns, npp, ds, dp), (hotel, drag, pe), sand, 5, linger)
    names = list(res.keys())
    a, s_ = res["Adaptive"], res["Fixed standard"]
    tone = lambda x, y: "up" if x <= y else "down"
    html(S.cards([(f"{a['energy_kj']:.0f} kJ", "Adaptive robot energy", f"{100 * (a['energy_kj'] / s_['energy_kj'] - 1):+.1f}% vs standard", tone(a["energy_kj"], s_["energy_kj"])),
                  (f"{a['time_min']:.0f} min", "Adaptive robot time", f"{100 * (a['time_min'] / s_['time_min'] - 1):+.1f}% vs standard", tone(a["time_min"], s_["time_min"])),
                  (f"{a['photos']:,.0f}", "Adaptive robot photos", f"{100 * (a['photos'] / s_['photos'] - 1):+.1f}% vs standard", tone(a["photos"], s_["photos"])),
                  (f"{a['bleached_detail_pct']:.1f}%", "Bleached coral in detail", "needs at least 4 photos per metre")], cols=4))
    c = st.columns([1.1, 1, 1])
    with c[0]:
        plot(C.energy_curve(hotel, drag, {0: dict(speed=fs), 1: dict(speed=ns), 2: dict(speed=ds)}), key="ec")
    with c[1]:
        plot(C.bars(names, [res[n]["energy_kj"] for n in names], "Energy (kJ)", "{:,.0f}", height=360), key="le")
    with c[2]:
        plot(C.bars(names, [res[n]["bleached_detail_pct"] for n in names], "Bleached in detail", "{:.0f}", "%", height=360), key="lb")
    st.caption("Uses the classifier's real error rates and 5 reefs, so numbers differ slightly from the 90-reef results.")
    next_button()

# ================================================================== GLOSSARY & LIMITS
else:
    html(S.section("Scope & limits", "What the interim report promised, and what was delivered",
                   "Each problem, objective and research gap from the interim report, checked against the evidence in this dashboard."))
    fa_ = load_csv("fusion_ablation.csv")
    FUSION_TXT = ""
    if fa_ is not None:
        g_ = fa_[np.isclose(fa_.sand_fraction, 0.5)].groupby("variant").mean(numeric_only=True)
        FUSION_TXT = (f"Camera, depth and position are combined with Bayes' rule. Per-metre accuracy rises from "
                      f"<b>{100 * g_.loc['Camera only'].accuracy:.0f}%</b> (camera only) to <b>{100 * g_.loc['Camera + depth + position'].accuracy:.0f}%</b>, "
                      f"and bleached coral found from <b>{100 * g_.loc['Camera only'].bleached_recall:.0f}%</b> to "
                      f"<b>{100 * g_.loc['Camera + depth + position'].bleached_recall:.0f}%</b>. <i>The depth sensor is simulated.</i>")
    cr_, rob_, mk_ = load_json("clean_retrain.json"), load_json("robustness.json"), load_csv("murky_survey.csv")
    MURKY_TXT = ""
    if rob_ and mk_ is not None:
        q_ = mk_[np.isclose(mk_.sand_fraction, 0.5)].groupby(["variant", "level"]).bleached_detail_pct.mean()
        MURKY_TXT = (f"<b>Murky water was tested</b> on real photos made murky in 5 steps: the classifier's bleached recall falls from "
                     f"{100 * rob_['classifier']['original_raw'][0]['bleached_recall']:.0f}% to {100 * rob_['classifier']['original_raw'][4]['bleached_recall']:.0f}%, "
                     f"and colour correction makes it worse. Combining sensors keeps <b>{q_[('Camera + depth + position', 0.5)]:.0f}%</b> of bleached "
                     f"coral in detail in murky water (camera alone {q_[('Camera only', 0.5)]:.0f}%); in very murky water both fail.")
    ho_ = load_json("holoocean_summary.json"); HO_TXT = ""
    if ho_:
        h_ = ho_["robots"]["Adaptive"]
        HO_TXT = (f"In HoloOcean (one reef, {ho_['reef_metres']} m) it photographed <b>{h_['holoocean']['bleached_detail_pct']:.1f}%</b> "
                  f"(2-D on the same reef: {h_['simulation_2d']['bleached_detail_pct']:.1f}%) and used "
                  f"{100 * (1 - h_['holoocean']['energy_kj'] / ho_['robots']['Fixed detailed']['holoocean']['energy_kj']):.0f}% less energy than the fixed detailed survey.")
    look_share = (100 * 4000 * rs.ROBOT["look_energy_j"] / 1000 / A50.energy_kj_mean) if summary is not None else float("nan")
    sand_ph = (f"{A50.photos_on_sand_pct_mean:.0f}%" if summary is not None else "–")
    nc = clf["near_copy_check"]["test"]["percent"] if clf else "–"
    rows = [
        ("Problem 1", "Robots scan blindly, wasting battery and storage on empty sand and rock.", "ok",
         f"Only <b>{sand_ph}</b> of the adaptive robot's photos are taken over sand/rock, against 50% for fixed surveys "
         f"(50% sand reef). It uses <b>{SAVE_P:.0f}% fewer photos</b> and <b>{SAVE_E:.0f}% less energy</b> than a fixed survey with the "
         "same detail. Against the standard survey it only saves energy on mostly-sand reefs. <i>Simulation only.</i>"),
        ("Problem 2", "Running picture recognition all the time drains the battery.", "found",
         f"Under the energy model, classifying every metre costs about <b>{look_share:.2f}%</b> of the mission's energy. "
         "Moving and keeping the electronics on cost far more. <i>Rests on an assumed 0.2 J per classification; "
         "not measured on embedded hardware.</i>"),
        ("Problem 3", "Murky water confuses recognition, causing false alarms and missed bleaching.", "ok" if MURKY_TXT else "part",
         "Both kinds of mistake were measured on an independent dataset. The knowledge graph removes about <b>82%</b> of false "
         "alarms and lingering recovers most missed bleaching. " + (MURKY_TXT or "<b>Murky water itself was not tested.</b>")),
        ("Objective i", "Lightweight models (MobileNetV3, Tiny U-Net) that sort healthy, bleached and sand/rock.", "ok",
         "Both built: MobileNetV3-Small (6.2 MB) and Tiny U-Net (2.0 MB). The classifier covers healthy vs bleached only, "
         "because no sand/rock photos were available; the Tiny U-Net covers all three."),
        ("Objective ii", "A control programme in ROS 2 that slows down for coral and speeds up over sand.", "part",
         "The control logic is built and tested (a rule and an RL agent), but in a Python simulation, <b>not inside ROS 2</b>."),
        ("Objective iii", "Test in HoloOcean and measure battery and storage savings without losing accuracy.", "ok" if HO_TXT else "part",
         (f"Savings measured on 90 reefs in a 2-D Python simulation (the adaptive robot photographs <b>{A50.bleached_detail_pct_mean:.1f}%</b> "
          f"of bleached coral in detail), then checked in HoloOcean. " + HO_TXT) if summary is not None else HO_TXT),
        ("Gap 1", "Balance missed detections against false alarms.", "ok",
         "Both measured separately (76.4% of bleached found; 92.6% of 'bleached' calls correct). Lingering and the knowledge graph "
         "handle each side."),
        ("Gap 2", "Combine depth, colour and position in one decision.", "ok" if FUSION_TXT else "no",
         FUSION_TXT or "No depth data was used."),
        ("Gap 3", "Double-check automatic results before they affect decisions.", "ok",
         "The knowledge graph's rule check raises the share of correct bleaching reports from about <b>87% to 97%</b>."),
        ("Gap 4", "Remove hidden shortcuts (repeated photos) before training.", "ok" if cr_ else "part",
         (f"Near-copies were grouped ({cr_['groups']:,} groups from {cr_['photos']:,} photos), each group kept in one split, and the "
          f"model retrained. Honest test accuracy: <b>{100 * cr_['clean_test']['accuracy']:.1f}%</b> (was 100%). On independent photos it "
          f"scores {100 * cr_['independent_clean_model']['accuracy']:.1f}%, so the dataset also lacks variety.") if cr_ else
         f"Detected, not removed: <b>{nc}%</b> of test photos were near-copies of training photos."),
    ]
    html(S.promise_table(rows))
    html(S.section("Ground rules", "What is real, measured or assumed"))
    html(S.features([
        ("Real", "Trained models", "MobileNetV3-Small classifier and Tiny U-Net, trained and tested on real reef photographs."),
        ("Measured", "Error rates", "The robot's mistakes follow the error rates measured on photos the models never saw."),
        ("Assumed", "Reef and robot", "A 2-D grid reef and a lawnmower path. Speeds, power use and photo energy are assumptions."),
    ], cols=3))
    html(S.section("Glossary", "Terms used in this dashboard"))
    table(pd.DataFrame([
        ("Bleaching", "Coral turning white under heat stress. Still alive, but at risk of dying."),
        ("Lawnmower survey", "Covering an area in straight back-and-forth lanes at one fixed speed and photo rate."),
        ("Detail", "A metre counts as photographed 'in detail' if it got at least 4 photos."),
        ("Linger", "After seeing bleaching, the adaptive robot stays slow for a few more metres in case it missed some."),
        ("Accuracy", "Share of all answers that were correct."),
        ("Recall", "Of everything that really is bleached, the share the model found."),
        ("Precision", "Of everything the model called bleached, the share that really was."),
        ("Near-copy", "A flipped, rotated or brightened version of the same photo (here, nearby frames of one video)."),
        ("IoU", "Overlap between the drawn outline and the true outline, divided by their combined area (1 = perfect)."),
        ("Dice", "Similar to IoU but counts the overlap twice; always a bit higher than IoU (1 = perfect)."),
        ("Hotel power", "Power used every second by the computer and sensors, whether moving or not."),
        ("Drag", "Power needed to push through water; rises with the cube of speed."),
        ("Knowledge graph", "Dots (bleaching reports) joined by lines when close together, used to confirm reports and find hotspots."),
        ("Degree centrality", "How many links a dot has, divided by the most it could have: C(v) = deg(v) / (N − 1)."),
        ("Reinforcement learning", "Learning by trial and reward instead of being given rules."),
        ("Q-learning", "Keeping a table of the average reward for each action in each situation, and picking the best."),
    ], columns=["Term", "Meaning"]))
    html(S.section("Limitations", "What this study does not show"))
    st.markdown("""
- **HoloOcean was used for one reef only** (800 m, one run per robot), without currents; the 90-reef results come from the 2-D simulation. ROS 2 was not used.
- **Slowing down takes about 4 m in HoloOcean**, longer than the 3 m linger, so on short bleached patches the vehicle takes detail-spaced photos before reaching detail speed. Motion blur is not modelled.
- **Robot figures are assumptions.** Speeds, power use and photo energy were not measured on a real vehicle (try them on the Robot Lab page).
- **Murky water is simulated** by changing real photos (haze, colour loss, noise, blur); real turbidity varies more. Above "murky", both models fail.
- **The depth sensor is simulated**: rock and coral heights and the sensor's noise are assumed values (Gap 2).
- **The dataset lacks variety.** After removing shortcut photos (Gap 4) the classifier is honest on its own data (97.4%) but weaker on photos from elsewhere.
- **The classifier has two classes.** Sand/rock comes from a simulated sensor (97% assumed accuracy) or from the Tiny U-Net.
- **Mistakes are drawn independently per metre**; in real water they may come in runs, which could weaken the linger rule.
- **Dataset caveats:** the classifier's training and independent datasets may share original sources; the Tiny U-Net was trained
  on Red Sea reefs and only 27 of its test photos count as bleached.
- **Speeds were measured on a GPU**, not on low-power hardware like a robot would carry.
""")
    s = load_json("settings.json")
    if s:
        with st.expander("All simulation settings"):
            st.json(s)

if FEEDBACK_URL and page != "Home":
    st.write("")
    st.columns([1, 3])[0].link_button("Give feedback (UAT form)", FEEDBACK_URL)
html(S.footer())
