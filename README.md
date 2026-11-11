# ReefScan: Autonomous Resource-Aware Agent for Priority-Based Coral Reef Health Mapping

Final Year Project, Bachelor of Computer Science (Hons), Universiti Teknologi PETRONAS
**Author:** Muhammad Alif Najmi bin Mohammad Zamri (22005632)
**Supervisor:** Ts. Dr. Manoranjitham A/P Muniandy · **Co-supervisor:** Ts. Dr. Mazeyanti Bt Mohd Ariffin

**Live dashboard:** _add your Streamlit link here_

## What this is

A simulated underwater survey robot that changes its speed and photo rate based on what it sees below it:
slow and detailed over **bleached coral**, normal over **healthy coral**, fast over **sand and rock**.
It is compared with fixed "mow-the-lawn" surveys on 90 simulated reefs, using the measured error rates
of real trained models.

The dashboard walks through the whole study in eight steps:

1. **Classifier**: MobileNetV3-Small (healthy vs bleached), including a near-copy check and a test on an independent dataset
2. **Tiny U-Net**: pixel-level segmentation of sand/rock, healthy and bleached coral
3. **Simulation**: watch a survey, lane by lane
4. **Results**: time, energy, photos and bleached-coral coverage over 90 reefs, plus the sensor-fusion comparison
5. **Knowledge graph**: linking bleaching reports into hotspots and filtering false alarms
6. **RL agent**: a Q-learning agent that learns its own speed rule
7. **Robot Lab**: change the robot's assumptions and re-run
8. **Glossary & Limits**: what was promised, what was delivered, and honest limitations

> This is a **simulation-only** study. The models are real and trained on real photographs; the reef,
> the robot's movement and its energy use are simulated with stated assumptions.

## Run it on your own computer

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

All results are already in `data/`. To regenerate them: `python run_experiments.py` (under a minute).

## Files

| File | Purpose |
|---|---|
| `app.py` | The dashboard |
| `reefsim.py` | Reef generation, robot, energy and photo accounting |
| `fusion.py` | Combines camera, depth and position in one decision (Bayes' rule) |
| `knowledge_graph.py` | Bleaching-report graph, hotspots, false-alarm check |
| `rl_agent.py` | Q-learning agent |
| `run_experiments.py` | Runs all experiments and saves `data/` |
| `charts.py`, `style.py`, `dotmap.py` | Charts and visual design |
| `colab/` | Google Colab scripts used to train and test the models |

## Data and credits

- Classifier training data: *Coral Reefs Images*, Kaggle (asfarhossainsitab). Only results are published here, not the photos.
- Independent test data: *Coral health classification*, Hugging Face (esahit), combining datasets by Jamil et al. (2021) and Shihavuddin et al. (2013).
- Segmentation data and example photos: *Coralscapes*, Sauder et al. (2025), Apache-2.0 licence.
- Models: MobileNetV3 (Howard et al., 2019) via PyTorch/torchvision; Tiny U-Net written for this project.
