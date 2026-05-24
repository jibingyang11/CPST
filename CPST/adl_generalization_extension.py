from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper" / "medication_extension.csv"


STAGES = [
    "receive reminder",
    "retrieve medication",
    "verify dose/person",
    "take medication",
    "caregiver confirmation",
    "log completion",
]


def simulate(seed: int, episodes: int = 260, windows_per_stage: int = 5):
    rng = np.random.default_rng(seed)
    xs = {"P": [], "C": [], "S": [], "T": []}
    y = []
    episode_id = []
    for ep in range(episodes):
        caregiver_present = rng.binomial(1, 0.42)
        med_time = rng.choice([0, 1, 2], p=[0.35, 0.40, 0.25])
        adherence_risk = rng.beta(2, 6)
        for stage, stage_name in enumerate(STAGES):
            for _ in range(windows_per_stage):
                p = rng.normal(0, 0.55, 18)
                c = rng.normal(0, 0.45, 10)
                s = rng.normal(0, 0.35, 8)
                t = rng.normal(0, 0.35, 9)

                movement = 1.0 if stage in [1, 3] else 0.25
                cabinet = 1.0 if stage == 1 else 0.15
                hand_to_mouth = 1.0 if stage == 3 else 0.10
                phone_or_clock = 1.0 if stage in [0, 5] else 0.20
                waiting = 1.0 if stage in [0, 2, 4] else 0.15
                p[:6] += [movement, cabinet, hand_to_mouth, phone_or_clock, waiting, rng.normal(0, 0.1)]

                reminder_log = 1.0 if stage == 0 else 0.10
                pillbox_open = 1.0 if stage in [1, 2] else 0.05
                dose_scan = 1.0 if stage == 2 else 0.05
                ingestion_log = 1.0 if stage in [3, 5] else 0.05
                c[:6] += [reminder_log, pillbox_open, dose_scan, ingestion_log, med_time / 2.0, adherence_risk]

                caregiver_speech = caregiver_present if stage in [2, 4] else 0.05 * caregiver_present
                identity_check = caregiver_present if stage == 2 else 0.05
                assistance = caregiver_present if stage in [3, 4] else 0.05
                s[:5] += [caregiver_present, caregiver_speech, identity_check, assistance, stage == 4]

                schedule_prior = 1.0 if stage == 0 else 0.25
                intent_to_take = 1.0 if stage in [0, 1, 2, 3] else 0.35
                dose_plan = 1.0 if stage == 2 else 0.10
                completion_goal = 1.0 if stage in [4, 5] else 0.10
                t[:6] += [schedule_prior, intent_to_take, dose_plan, completion_goal, med_time / 2.0, adherence_risk]

                xs["P"].append(p)
                xs["C"].append(c)
                xs["S"].append(s)
                xs["T"].append(t)
                y.append(stage)
                episode_id.append(ep)
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in xs.items()}
    return arrays, np.asarray(y), np.asarray(episode_id)


def run_once(seed: int, spaces: tuple[str, ...]):
    arrays, y, episodes = simulate(seed)
    train_eps, test_eps = train_test_split(np.unique(episodes), test_size=0.30, random_state=seed)
    train_mask = np.isin(episodes, train_eps)
    test_mask = ~train_mask
    x = np.hstack([arrays[s] for s in spaces])
    clf = ExtraTreesClassifier(n_estimators=220, min_samples_leaf=2, class_weight="balanced", random_state=seed)
    clf.fit(x[train_mask], y[train_mask])
    pred = clf.predict(x[test_mask])
    acc = accuracy_score(y[test_mask], pred)
    f1 = f1_score(y[test_mask], pred, average="macro")
    early = accuracy_score(y[test_mask][y[test_mask] <= 2], pred[y[test_mask] <= 2])
    return acc, f1, early


def fmt(vals, col):
    vals = np.asarray(vals, dtype=float)
    return f"{vals[:, col].mean():.4f}\\pm{vals[:, col].std(ddof=1):.4f}"


def fmt_p(vals, ref):
    p = wilcoxon(ref[:, 0], vals[:, 0], alternative="greater").pvalue
    if p < 0.001:
        return "<0.001^{***}"
    if p < 0.01:
        return f"{p:.3f}^{{**}}"
    if p < 0.05:
        return f"{p:.3f}^{{*}}"
    return f"{p:.3f}"


def main():
    configs = [
        ("Medication-P", ("P",)),
        ("Medication-P+C", ("P", "C")),
        ("Medication-CPST", ("P", "C", "S", "T")),
    ]
    scores = {name: np.array([run_once(seed, spaces) for seed in range(10)]) for name, spaces in configs}
    ref = scores["Medication-CPST"]
    rows = []
    for name, spaces in configs:
        vals = scores[name]
        rows.append(
            {
                "Activity": "Medication management",
                "Model": name,
                "Spaces": "+".join(spaces),
                "Stages": str(len(STAGES)),
                "Accuracy": fmt(vals, 0),
                "Macro F1": fmt(vals, 1),
                "Early Acc.": fmt(vals, 2),
                "$p$ vs CPST": "Ref." if name.endswith("CPST") else fmt_p(vals, ref),
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
