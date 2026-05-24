from __future__ import annotations

import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
CASAS_ARUBA = ROOT / "data" / "casas" / "new_labeled_data" / "aruba.txt"
OPPORTUNITY_DIR = ROOT / "data" / "public_adl" / "opportunity" / "OpportunityUCIDataset" / "dataset"
UCI_DIR = ROOT / "data" / "public_adl" / "uci_har" / "UCI HAR Dataset"
OUT = ROOT / "paper" / "public_neural_validation.csv"


@dataclass
class Event:
    time: datetime
    sensor: str
    value: str
    label: str | None


def parse_aruba(path: Path) -> list[Event]:
    events: list[Event] = []
    active: str | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        time_part = parts[1]
        if "." in time_part:
            head, frac = time_part.split(".", 1)
            time_part = f"{head}.{(frac + '000000')[:6]}"
        stamp = datetime.fromisoformat(f"{parts[0]} {time_part}")
        sensor, value = parts[2], parts[3]
        event_label = active
        if len(parts) >= 6 and parts[-1] in {"begin", "end"}:
            activity = " ".join(parts[4:-1])
            if parts[-1] == "begin":
                active = activity
                event_label = activity
            else:
                event_label = activity
                active = None
        events.append(Event(stamp, sensor, value, event_label))
    return events


def build_casas(window: int = 30, stride: int = 20):
    events = parse_aruba(CASAS_ARUBA)
    motion_sensors = [s for s, _ in Counter(e.sensor for e in events if e.sensor.startswith("M")).most_common(36)]
    door_sensors = [s for s, _ in Counter(e.sensor for e in events if e.sensor.startswith("D")).most_common(10)]
    temp_sensors = [s for s, _ in Counter(e.sensor for e in events if e.sensor.startswith("T")).most_common(8)]
    motion_idx = {s: i for i, s in enumerate(motion_sensors)}
    door_idx = {s: i for i, s in enumerate(door_sensors)}
    temp_idx = {s: i for i, s in enumerate(temp_sensors)}

    xp, xc, xt, y = [], [], [], []
    for start in range(0, len(events) - window + 1, stride):
        chunk = events[start : start + window]
        labels = [e.label for e in chunk if e.label]
        if not labels:
            continue
        label, count = Counter(labels).most_common(1)[0]
        if count < max(3, window // 5):
            continue

        p = np.zeros(len(motion_sensors) + len(door_sensors) + 4, dtype=np.float32)
        c = np.zeros(len(temp_sensors) + 7, dtype=np.float32)
        temp_values: list[float] = []
        on_count = off_count = numeric_count = 0
        unique_sensors = set()
        for e in chunk:
            unique_sensors.add(e.sensor)
            if e.sensor in motion_idx:
                p[motion_idx[e.sensor]] += 1.0
            if e.sensor in door_idx:
                p[len(motion_sensors) + door_idx[e.sensor]] += 1.0
            if e.value.upper() == "ON":
                on_count += 1
            elif e.value.upper() == "OFF":
                off_count += 1
            else:
                try:
                    val = float(e.value)
                    numeric_count += 1
                    if e.sensor in temp_idx:
                        c[temp_idx[e.sensor]] = val
                        temp_values.append(val)
                except ValueError:
                    pass
        p[-4:] = [
            sum(1 for e in chunk if e.sensor.startswith("M")),
            sum(1 for e in chunk if e.sensor.startswith("D")),
            len(unique_sensors),
            sum(chunk[i].sensor != chunk[i - 1].sensor for i in range(1, len(chunk))),
        ]
        if temp_values:
            c[len(temp_sensors) : len(temp_sensors) + 4] = [
                float(np.mean(temp_values)),
                float(np.std(temp_values)),
                float(np.min(temp_values)),
                float(np.max(temp_values)),
            ]
        c[-3:] = [on_count / window, off_count / window, numeric_count / window]

        t0 = chunk[-1].time
        hour = t0.hour + t0.minute / 60.0
        dow = t0.weekday()
        t = np.array(
            [
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
                math.sin(2 * math.pi * dow / 7),
                math.cos(2 * math.pi * dow / 7),
                1.0 if dow >= 5 else 0.0,
            ],
            dtype=np.float32,
        )
        xp.append(p)
        xc.append(c)
        xt.append(t)
        y.append(label)

    split = int(len(y) * 0.70)
    spaces = {"P": np.vstack(xp), "C": np.vstack(xc), "T": np.vstack(xt)}
    labels = np.asarray(y)
    return {k: (v[:split], v[split:]) for k, v in spaces.items()}, labels[:split], labels[split:]


def window_stats(block: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(block)
    counts = valid.sum(axis=0)
    safe = np.nan_to_num(block, nan=0.0)
    mean = np.divide(safe.sum(axis=0), counts, out=np.zeros(block.shape[1], dtype=np.float32), where=counts > 0)
    centered = np.where(valid, block - mean, 0.0)
    var = np.divide((centered * centered).sum(axis=0), counts, out=np.zeros(block.shape[1], dtype=np.float32), where=counts > 0)
    std = np.sqrt(var)
    return np.nan_to_num(np.concatenate([mean, std]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def opportunity_file_windows(path: Path, window: int = 90, stride: int = 90):
    data = np.loadtxt(path, dtype=np.float32)
    label = data[:, 244].astype(np.int64)  # HL_Activity, one-based column 245.
    p_idx = np.r_[1:134, 231:243]  # body/shoe inertial plus location tags.
    c_idx = np.r_[134:231]  # object and ambient sensors.
    xp, xc, y = [], [], []
    for start in range(0, len(data) - window + 1, stride):
        end = start + window
        labels = label[start:end]
        labels = labels[labels > 0]
        if len(labels) < window * 0.60:
            continue
        cls = Counter(labels.tolist()).most_common(1)[0][0]
        xp.append(window_stats(data[start:end, p_idx]))
        xc.append(window_stats(data[start:end, c_idx]))
        y.append(str(cls))
    return xp, xc, y


def build_opportunity():
    train_files = sorted(OPPORTUNITY_DIR.glob("S[123]-ADL*.dat"))
    test_files = sorted(OPPORTUNITY_DIR.glob("S4-ADL*.dat"))
    xptr, xctr, ytr = [], [], []
    xpte, xcte, yte = [], [], []
    for path in train_files:
        p, c, y = opportunity_file_windows(path)
        xptr.extend(p)
        xctr.extend(c)
        ytr.extend(y)
    for path in test_files:
        p, c, y = opportunity_file_windows(path)
        xpte.extend(p)
        xcte.extend(c)
        yte.extend(y)
    spaces = {
        "P": (np.vstack(xptr), np.vstack(xpte)),
        "C": (np.vstack(xctr), np.vstack(xcte)),
    }
    return spaces, np.asarray(ytr), np.asarray(yte)


def build_uci_har():
    x_train = np.loadtxt(UCI_DIR / "train" / "X_train.txt", dtype=np.float32)
    y_train = np.loadtxt(UCI_DIR / "train" / "y_train.txt", dtype=np.int64).astype(str)
    x_test = np.loadtxt(UCI_DIR / "test" / "X_test.txt", dtype=np.float32)
    y_test = np.loadtxt(UCI_DIR / "test" / "y_test.txt", dtype=np.int64).astype(str)
    feature_names = []
    for line in (UCI_DIR / "features.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        _, name = line.split(maxsplit=1)
        feature_names.append(name)
    p_idx = np.array([i for i, name in enumerate(feature_names) if name.startswith("t")])
    c_idx = np.array([i for i, name in enumerate(feature_names) if name.startswith("f")])
    return {"P": (x_train[:, p_idx], x_test[:, p_idx]), "C": (x_train[:, c_idx], x_test[:, c_idx])}, y_train, y_test


class SpaceAttentionNet(nn.Module):
    def __init__(self, dims: list[int], n_classes: int, hidden: int = 48):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.10),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                )
                for d in dims
            ]
        )
        self.score = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, xs: list[torch.Tensor]):
        hs = torch.stack([enc(x) for enc, x in zip(self.encoders, xs)], dim=1)
        attn = torch.softmax(self.score(hs).squeeze(-1), dim=1).unsqueeze(-1)
        fused = (hs * attn).sum(dim=1)
        return self.classifier(fused)


def train_once(spaces_train, spaces_test, y_train_raw, y_test_raw, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    enc = LabelEncoder()
    y_train = enc.fit_transform(y_train_raw)
    y_test = enc.transform(y_test_raw)

    scaled_train, scaled_test = [], []
    for xtr, xte in zip(spaces_train, spaces_test):
        scaler = StandardScaler()
        xtr = scaler.fit_transform(xtr)
        xte = scaler.transform(xte)
        scaled_train.append(torch.tensor(xtr.astype(np.float32)))
        scaled_test.append(torch.tensor(xte.astype(np.float32)))

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    loader = DataLoader(TensorDataset(*scaled_train, y_train_t), batch_size=256, shuffle=True)

    model = SpaceAttentionNet([x.shape[1] for x in spaces_train], len(enc.classes_))
    counts = np.bincount(y_train, minlength=len(enc.classes_)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    model.train()
    for _ in range(12):
        for batch in loader:
            *xb, yb = batch
            opt.zero_grad()
            loss = criterion(model(list(xb)), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(scaled_test).argmax(dim=1).cpu().numpy()
    return accuracy_score(y_test_t.numpy(), pred), f1_score(y_test_t.numpy(), pred, average="macro")


def run_scores(space_pairs, y_train, y_test, seeds=range(10)):
    spaces_train = [pair[0] for pair in space_pairs]
    spaces_test = [pair[1] for pair in space_pairs]
    return np.array([train_once(spaces_train, spaces_test, y_train, y_test, seed) for seed in seeds], dtype=np.float64)


def fmt_score(scores: np.ndarray, col: int):
    return f"{scores[:, col].mean():.4f}\\pm{scores[:, col].std(ddof=1):.4f}"


def fmt_p(scores: np.ndarray, ref_scores: np.ndarray):
    if np.array_equal(scores, ref_scores):
        return "Ref."
    p = wilcoxon(ref_scores[:, 0], scores[:, 0], alternative="greater").pvalue
    if p < 0.001:
        return "<0.001^{***}"
    if p < 0.01:
        return f"{p:.3f}^{{**}}"
    if p < 0.05:
        return f"{p:.3f}^{{*}}"
    return f"{p:.3f}"


def dataset_rows(dataset: str, spaces, y_train, y_test, configs):
    cache = {}
    for model, space_keys in configs:
        cache[model] = run_scores([spaces[k] for k in space_keys], y_train, y_test)
    ref_scores = cache[configs[-1][0]]
    rows = []
    for model, space_keys in configs:
        scores = cache[model]
        rows.append(
            {
                "Dataset": dataset,
                "Model": model,
                "Spaces": "+".join(space_keys),
                "Train/Test": f"{len(y_train)}/{len(y_test)}",
                "Classes": str(len(set(y_train) | set(y_test))),
                "Accuracy": fmt_score(scores, 0),
                "Macro F1": fmt_score(scores, 1),
                "$p$ vs Ref.": fmt_p(scores, ref_scores),
            }
        )
    return rows


def main():
    rows = []
    casas_spaces, casas_y_train, casas_y_test = build_casas()
    rows.extend(
        dataset_rows(
            "CASAS Aruba",
            casas_spaces,
            casas_y_train,
            casas_y_test,
            [("Neural-P", ("P",)), ("Neural-P+C", ("P", "C")), ("Neural-P+C+T", ("P", "C", "T"))],
        )
    )
    opp_spaces, opp_y_train, opp_y_test = build_opportunity()
    rows.extend(
        dataset_rows(
            "OPPORTUNITY",
            opp_spaces,
            opp_y_train,
            opp_y_test,
            [("Neural-P", ("P",)), ("Neural-C", ("C",)), ("Neural-P+C", ("P", "C"))],
        )
    )
    uci_spaces, uci_y_train, uci_y_test = build_uci_har()
    rows.extend(
        dataset_rows(
            "UCI HAR",
            uci_spaces,
            uci_y_train,
            uci_y_test,
            [("Neural-P", ("P",)), ("Neural-C", ("C",)), ("Neural-P+C", ("P", "C"))],
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
