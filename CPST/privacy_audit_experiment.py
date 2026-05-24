from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper" / "privacy_audit.csv"


def simulate_cpst(seed: int, n: int = 6000):
    rng = np.random.default_rng(seed)
    labels = rng.choice(
        ["tea", "serve_guest_tea", "shower", "cook", "medication", "relax"],
        size=n,
        p=[0.20, 0.12, 0.18, 0.20, 0.10, 0.20],
    )
    label_id = {name: i for i, name in enumerate(sorted(set(labels)))}
    y = np.array([label_id[v] for v in labels], dtype=np.int64)

    guest = (labels == "serve_guest_tea").astype(float)
    care_need = (labels == "medication").astype(float)
    post_meal_habit = ((labels == "tea") | (labels == "serve_guest_tea") | (labels == "cook")).astype(float)
    post_meal_habit = np.clip(post_meal_habit + rng.binomial(1, 0.08, n), 0, 1)
    sensitive = np.vstack([guest, care_need, post_meal_habit]).T.astype(int)

    p = rng.normal(0, 0.45, (n, 12))
    c = rng.normal(0, 0.45, (n, 9))
    s = rng.normal(0, 0.35, (n, 8))
    t = rng.normal(0, 0.35, (n, 8))

    kitchen = np.isin(labels, ["tea", "serve_guest_tea", "cook"]).astype(float)
    water = np.isin(labels, ["tea", "serve_guest_tea", "shower"]).astype(float)
    stove = (labels == "cook").astype(float)
    cup = np.isin(labels, ["tea", "serve_guest_tea"]).astype(float)
    shower = (labels == "shower").astype(float)
    med = (labels == "medication").astype(float)

    p[:, 0] += kitchen
    p[:, 1] += water
    p[:, 2] += stove
    p[:, 3] += cup
    p[:, 4] += shower
    p[:, 5] += med
    c[:, 0] += water
    c[:, 1] += stove
    c[:, 2] += cup
    c[:, 3] += med

    s[:, 0] += guest
    s[:, 1] += care_need
    s[:, 2] += np.isin(labels, ["serve_guest_tea", "medication"]).astype(float)
    s[:, 3] += guest * 0.8 + rng.normal(0, 0.08, n)
    t[:, 0] += post_meal_habit
    t[:, 1] += care_need
    t[:, 2] += (labels == "tea").astype(float)
    t[:, 3] += (labels == "cook").astype(float)
    return p, c, s, t, y, sensitive


def transform_spaces(p, c, s, t, mitigation: str, seed: int):
    rng = np.random.default_rng(seed + 911)
    s2 = s.copy()
    t2 = t.copy()
    if mitigation == "Raw CPST":
        return [p, c, s2, t2], np.hstack([s2, t2])
    if mitigation == "Encrypted local CPST":
        return [p, c, s2, t2], np.hstack([s2, t2])
    if mitigation == "Minimized S/T":
        s2[:, [0, 1, 3]] = 0.0
        t2[:, [0, 1, 2]] = 0.0
        return [p, c, s2, t2], np.hstack([s2, t2])
    if mitigation == "Minimized + noise/trust":
        s2[:, [0, 1, 3]] = 0.0
        t2[:, [0, 1, 2]] = 0.0
        st = np.hstack([s2, t2])
        mask = rng.random(st.shape) > 0.18
        st = st * mask + rng.normal(0, 0.25, st.shape)
        return [p, c, st[:, : s.shape[1]], st[:, s.shape[1] :]], st
    raise ValueError(mitigation)


def evaluate(seed: int, mitigation: str):
    p, c, s, t, y, sensitive = simulate_cpst(seed)
    spaces, exposed_st = transform_spaces(p, c, s, t, mitigation, seed)
    x = np.hstack(spaces)
    x_train, x_test, y_train, y_test, st_train, st_test, z_train, z_test = train_test_split(
        x, y, exposed_st, sensitive, test_size=0.30, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    utility = ExtraTreesClassifier(n_estimators=180, min_samples_leaf=2, random_state=seed, class_weight="balanced")
    utility.fit(x_train, y_train)
    pred = utility.predict(x_test)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    attack_scaler = StandardScaler()
    st_train = attack_scaler.fit_transform(st_train)
    st_test = attack_scaler.transform(st_test)
    attacker = MultiOutputClassifier(ExtraTreesClassifier(n_estimators=120, min_samples_leaf=2, random_state=seed))
    attacker.fit(st_train, z_train)
    attack_pred = attacker.predict(st_test)
    attack_acc = (attack_pred == z_test).mean()
    aucs = []
    for j, est in enumerate(attacker.estimators_):
        prob = est.predict_proba(st_test)[:, 1]
        aucs.append(roc_auc_score(z_test[:, j], prob))
    return acc, macro_f1, attack_acc, float(np.mean(aucs))


def fmt(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.4f}\\pm{values.std(ddof=1):.4f}"


def fmt_p(values, ref):
    p = wilcoxon(ref, values, alternative="greater").pvalue
    if p < 0.001:
        return "<0.001^{***}"
    if p < 0.01:
        return f"{p:.3f}^{{**}}"
    if p < 0.05:
        return f"{p:.3f}^{{*}}"
    return f"{p:.3f}"


def main():
    mitigations = ["Raw CPST", "Encrypted local CPST", "Minimized S/T", "Minimized + noise/trust"]
    metrics = {m: np.array([evaluate(seed, m) for seed in range(10)]) for m in mitigations}
    raw = metrics["Raw CPST"]
    rows = []
    for m in mitigations:
        vals = metrics[m]
        rows.append(
            {
                "Mitigation": m,
                "ADL Accuracy": fmt(vals[:, 0]),
                "Macro F1": fmt(vals[:, 1]),
                "Utility Cost": f"{(raw[:, 0].mean() - vals[:, 0].mean()):.4f}",
                "Attack Acc.": fmt(vals[:, 2]),
                "Attack AUC": fmt(vals[:, 3]),
                "$p$ Utility": "Ref." if m == "Raw CPST" else fmt_p(vals[:, 0], raw[:, 0]),
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
