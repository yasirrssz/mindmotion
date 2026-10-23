"""
Export real EEG motor-imagery trials plus a small trained model for MindMotion.

Data: PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB), downloaded by MNE.
Setup:  pip install mne scikit-learn numpy scipy
Run:    python export_trials.py        (writes data.js next to this file)

Classes: 0 = left hand, 1 = right hand, 2 = both feet, 3 = rest.
Only held-out (unseen) trials are exported, so the demo is an honest test.
The filter and band-power code below mirrors the JavaScript in index.html,
so features match between training (here) and prediction (browser).
"""
import json
from pathlib import Path

import numpy as np
from mne.datasets import eegbci
from mne.io import read_raw_edf
from scipy.signal import lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

SUBJECTS = [1]  # avoid 88, 89, 92, 100, 104, 106 (known timing problems)
CH = ["C3", "Cz", "C4", "Pz"]
FS, N = 160, 640  # 4 seconds at 160 Hz
NOTCH = 60  # mains frequency in the US recordings
MU = [8, 9, 10, 11, 12, 13]
BETA = [14, 16, 18, 20, 22, 24, 26, 28]
rng = np.random.default_rng(0)


def bq(kind, f0, Q):
    w = 2 * np.pi * f0 / FS
    al, cw = np.sin(w) / (2 * Q), np.cos(w)
    a0, a1, a2 = 1 + al, -2 * cw, 1 - al
    if kind == "lp":
        b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = b0
    elif kind == "hp":
        b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = b0
    else:
        b0, b1, b2 = 1, -2 * cw, 1
    return np.array([b0, b1, b2]) / a0, np.array([1, a1 / a0, a2 / a0])


FILT = [bq("notch", NOTCH, 8), bq("hp", 8, .707), bq("hp", 8, .707),
        bq("lp", 30, .707), bq("lp", 30, .707)]
T = np.arange(64, N)
WIN = .5 - .5 * np.cos(2 * np.pi * (T - 64) / (N - 65))


def clean(x):
    for b, a in FILT:
        x = lfilter(b, a, x)
    return x


def band(x, freqs):
    p = 0.0
    for f in freqs:
        ang = 2 * np.pi * f * T / FS
        re = (x[T] * WIN * np.cos(ang)).sum()
        im = (x[T] * WIN * np.sin(ang)).sum()
        p += (re * re + im * im) / ((N - 64) ** 2)
    return p


def features(trial):
    c = [clean(x) for x in trial]
    return [np.log(band(x, MU) + 1e-9) for x in c] + [np.log(band(x, BETA) + 1e-9) for x in c]


def load_subject(subject):
    out = {0: [], 1: [], 2: [], 3: []}

    def read(runs, mapping):
        for path in eegbci.load_data(subject, runs, verbose=False):
            raw = read_raw_edf(path, preload=True, verbose=False)
            eegbci.standardize(raw)
            raw.pick(CH)
            raw.reorder_channels(CH)
            assert int(raw.info["sfreq"]) == FS
            data = raw.get_data() * 1e6  # volts -> microvolts
            for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
                s = int(round(onset * FS))
                if desc in mapping and s + N <= data.shape[1]:
                    seg = data[:, s:s + N]
                    out[mapping[desc]].append(seg - seg.mean(axis=1, keepdims=True))

    read([4, 8, 12], {"T1": 0, "T2": 1, "T0": 3})  # imagine left / right fist, rest
    read([6, 10, 14], {"T2": 2})  # imagine both feet
    n = min(len(v) for v in out.values())
    for k in out:  # same number of trials per class
        out[k] = [out[k][i] for i in rng.choice(len(out[k]), n, replace=False)]
    return out


def main():
    subjects = {}
    for sid in SUBJECTS:
        print(f"Subject {sid}: downloading and processing...")
        data = load_subject(sid)
        X_raw = np.array([t for k in range(4) for t in data[k]])
        y = np.array([k for k in range(4) for _ in data[k]])
        idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3,
                                          stratify=y, random_state=0)
        F = np.array([features(t) for t in X_raw])
        sc = StandardScaler().fit(F[idx_tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(F[idx_tr]), y[idx_tr])
        acc = clf.score(sc.transform(F[idx_te]), y[idx_te])
        print(f"  trials per class: {len(data[0])}, accuracy on unseen trials: {acc:.0%}")
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        metrics = {}
        for name, m in {"Logistic Regression": LogisticRegression(max_iter=2000), "SVM (RBF)": SVC(),
                        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0)}.items():
            pred = cross_val_predict(make_pipeline(StandardScaler(), m), F, y, cv=cv)
            pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
            metrics[name] = {"acc": round(accuracy_score(y, pred), 3), "prec": round(pr, 3),
                             "rec": round(rc, 3), "f1": round(f1, 3)}
            print(f"  {name}: accuracy {metrics[name]['acc']:.0%}, precision {pr:.2f}, recall {rc:.2f}, F1 {f1:.2f}")
            if name == "Logistic Regression":
                print("  Confusion matrix (rows = true, columns = predicted; left, right, feet, rest):")
                print(confusion_matrix(y, pred))
        trials = {str(k): [np.round(X_raw[i], 1).tolist() for i in idx_te if y[i] == k]
                  for k in range(4)}
        subjects[f"S{sid}"] = {
            "acc": round(float(acc), 3), "metrics": metrics, "trials": trials,
            "mean": sc.mean_.tolist(), "std": sc.scale_.tolist(),
            "W": clf.coef_.tolist(), "b": clf.intercept_.tolist(),
        }
    payload = {"fs": FS, "n": N, "ch": CH, "notch": NOTCH, "mu": MU, "beta": BETA,
               "subjects": subjects}
    Path("data.js").write_text("window.MM_DATA=" + json.dumps(payload, separators=(",", ":")) + ";")
    print("Wrote data.js")


if __name__ == "__main__":
    main()