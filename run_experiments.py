"""
run_experiments.py: runs every experiment and saves the results to data/
for the dashboard and the report.

    python run_experiments.py

Takes well under a minute on a laptop. Results are identical every run.
"""
import os, csv, json
import numpy as np
import reefsim as rs
import knowledge_graph as kg
import rl_agent as rl
import fusion as fu

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
SAND = [0.3, 0.5, 0.7]
N_MAPS = 30
UNET_M = rs.load_unet_perception(os.path.join(OUT, "unet_results.json"))
PERCEPTIONS = list(rs.CLASSIFIERS) + ([rs.UNET_NAME, rs.FUSED_NAME] if UNET_M is not None else [])
TAGS = {"Measured classifier": "", rs.UNET_NAME: "_unet", rs.FUSED_NAME: "_fused"}
print("Perception settings:", PERCEPTIONS)

# ---------------- 1. train the RL agent (on separate training reefs), one per kind of perception
print("Training RL agents...")
QS = {}
for p in PERCEPTIONS:
    if p == "Perfect classifier":
        continue
    Qp, Np, curve_p = rl.train(n_reefs=120, perception=p, unet_M=UNET_M)
    QS[p] = Qp
    tag = TAGS[p]
    np.save(os.path.join(OUT, f"rl_Q{tag}.npy"), Qp); np.save(os.path.join(OUT, f"rl_N{tag}.npy"), Np)
    with open(os.path.join(OUT, f"rl_curve{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve_p[0].keys())); w.writeheader(); w.writerows(curve_p)
    with open(os.path.join(OUT, f"rl_policy{tag}.csv"), "w", newline="") as fh:
        rows = rl.policy_table(Qp, Np)
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
QS["Perfect classifier"] = QS["Measured classifier"]
Q = QS["Measured classifier"]

# ---------------- 2. run all vehicles on the 90 test reefs
print("Running surveys...")
STRATS = ["Fixed standard", "Fixed detailed", "Adaptive", "Adaptive (no linger)", "RL agent"]
runs, kg_rows = [], []
for sf, k, seed in rs.map_seeds(SAND, N_MAPS):
    reef = rs.make_reef(np.random.default_rng(seed), sf)
    rows_, cols_, cells = rs.lawnmower_path(reef)
    for clf_name in PERCEPTIONS:
        seen = rs.perceive(cells, np.random.default_rng(seed + 7), clf_name, UNET_M,
                           ctx=dict(reef=reef, rows=rows_, cols=cols_, seed=seed))
        nb = rs.neighbour_feature(rows_, cols_, seen)
        pols = dict(rs.strategies()); pols["RL agent"] = rl.QPolicy(QS[clf_name])
        for name in STRATS:
            log = rs.run_survey(cells, seen, pols[name], nb)
            m = rs.metrics(cells, log)
            m["reward"] = rl.total_reward(cells, log)
            runs.append(dict(classifier=clf_name, sand_fraction=sf, map=k, strategy=name, **m))
        if clf_name == "Measured classifier":
            G = kg.build_graph(reef, rows_, cols_, seen)
            kg_rows.append(dict(sand_fraction=sf, map=k, **kg.evaluate(reef, G)))

with open(os.path.join(OUT, "all_runs.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(runs[0].keys())); w.writeheader(); w.writerows(runs)

METRICS = ["time_min", "energy_kj", "photos", "storage_gb", "bleached_detail_pct",
           "photos_per_bleached_cell", "photos_on_sand_pct", "reward"]
summary = []
for clf_name in PERCEPTIONS:
    for sf in SAND:
        ref = [r for r in runs if r["classifier"] == clf_name and r["sand_fraction"] == sf and r["strategy"] == "Fixed standard"]
        refm = {m: np.mean([r[m] for r in ref]) for m in METRICS}
        for st in STRATS:
            sel = [r for r in runs if r["classifier"] == clf_name and r["sand_fraction"] == sf and r["strategy"] == st]
            s = dict(classifier=clf_name, sand_fraction=sf, strategy=st)
            for m in METRICS:
                v = np.array([r[m] for r in sel])
                s[m + "_mean"] = round(float(v.mean()), 3); s[m + "_sd"] = round(float(v.std(ddof=1)), 3)
            for m in ["time_min", "energy_kj", "photos"]:
                s[m + "_vs_standard_pct"] = round(100 * (s[m + "_mean"] / refm[m] - 1), 1)
            summary.append(s)
with open(os.path.join(OUT, "summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

# ---------------- sensor fusion ablation (Gap 2)
if UNET_M is not None:
    print("Sensor fusion ablation...")
    abl = []
    for sf, k, seed in rs.map_seeds(SAND, N_MAPS):
        reef = rs.make_reef(np.random.default_rng(seed), sf)
        rows_, cols_, cells = rs.lawnmower_path(reef)
        for v in fu.VARIANTS:
            seen = fu.perceive_fused(reef, rows_, cols_, cells, seed, UNET_M, v)
            row = dict(sand_fraction=sf, map=k, variant=v, **fu.perception_scores(cells, seen))
            for lab, pol in [("adaptive", rs.RulePolicy(3)), ("no_linger", rs.RulePolicy(0))]:
                m = rs.metrics(cells, rs.run_survey(cells, seen, pol))
                row[f"{lab}_energy_kj"] = m["energy_kj"]; row[f"{lab}_bleached_detail_pct"] = m["bleached_detail_pct"]
                row[f"{lab}_photos"] = m["photos"]
            abl.append(row)
    with open(os.path.join(OUT, "fusion_ablation.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(abl[0].keys())); w.writeheader(); w.writerows(abl)

with open(os.path.join(OUT, "kg_eval.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(kg_rows[0].keys())); w.writeheader(); w.writerows(kg_rows)

json.dump(dict(MAP=rs.MAP, ROBOT=rs.ROBOT, MODES={rs.MODE_NAMES[k]: v for k, v in rs.MODES.items()},
               SENSOR=rs.SENSOR, CLASSIFIERS=rs.CLASSIFIERS, SAND=SAND, N_MAPS=N_MAPS,
               KG=dict(link_radius_m=kg.LINK_RADIUS_M, min_degree=kg.MIN_DEGREE), RL_REWARD=rl.REWARD,
               FUSION=dict(depth=fu.DEPTH, position_weight=fu.POSITION_WEIGHT,
                           position_weight_tuned_on="18 practice reefs with seeds separate from the 90 test reefs")),
          open(os.path.join(OUT, "settings.json"), "w"), indent=2)

# ---------------- 3. short printout
print("\nMeasured classifier, 50% sand/rock:")
for s in summary:
    if s["classifier"] == "Measured classifier" and s["sand_fraction"] == 0.5:
        print(f"  {s['strategy']:22s} energy {s['energy_kj_mean']:6.1f} kJ  time {s['time_min_mean']:5.1f} min  "
              f"photos {s['photos_mean']:6.0f}  bleached detail {s['bleached_detail_pct_mean']:5.1f}%  reward {s['reward_mean']:7.1f}")
print("\nKnowledge graph (means over reefs):")
for sf in SAND:
    L = [r for r in kg_rows if r["sand_fraction"] == sf]
    print(f"  sand {int(sf*100)}%: precision all {np.mean([r['precision_all'] for r in L]):.3f} -> confirmed "
          f"{np.mean([r['precision_confirmed'] for r in L]):.3f}; false alarms flagged {np.nanmean([r['false_alarms_flagged'] for r in L]):.3f}; "
          f"patches found {np.mean([r['patches_found_pct'] for r in L]):.1f}%; hotspots {np.mean([r['hotspots_confirmed'] for r in L]):.1f} "
          f"vs true patches {np.mean([r['true_patches'] for r in L]):.1f}")
print("\nSaved to", OUT)
