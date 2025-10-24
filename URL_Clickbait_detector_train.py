from __future__ import annotations
import json
import logging
from pathlib import Path
import hashlib
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    GroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"
REPORTS_DIR = PROJECT_DIR / "reports"
DEFAULT_CSV_1 = DATA_DIR / "clickbait_dataset-2.csv"
DEFAULT_CSV_2 = DATA_DIR / "20000_turkish_news_title.csv"
DEFAULT_CSV = DEFAULT_CSV_1
DEFAULT_TEXT_COL = "text"
DEFAULT_LABEL_COL = "label"
DEFAULT_MODEL_PATH = MODEL_DIR / "clickbait_model.joblib"
RANDOM_STATE = 42

def load_df(csv_path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8")
    try:
        df.columns = df.columns.str.strip()
    except Exception:
        pass
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Beklenen kolonlar yok. Var olan: {list(df.columns)}; aranan: {text_col}, {label_col}"
        )
    df = df[[text_col, label_col]].dropna()
    df[label_col] = df[label_col].astype(int)
    return df


def build_pipeline(model_type: str = "logreg", max_features: int = 50000) -> Pipeline:
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        max_df=0.9,
        strip_accents="unicode",
        lowercase=True,
        stop_words=stopwords.words('turkish'),
    )
    if model_type == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
    elif model_type == "linearsvc":
        clf = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
    else:
        raise ValueError("model_type 'logreg' veya 'linearsvc' olmalı")
    pipe = Pipeline([
        ("tfidf", tfidf),
        ("clf", clf),
    ])
    return pipe


def train(
    csv_path: Path = DEFAULT_CSV,
    text_col: str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
    model_out: Path = DEFAULT_MODEL_PATH,
    model_type: str = "logreg",
    do_grid: bool = False,
):
    if DEFAULT_CSV_1.exists() and DEFAULT_CSV_2.exists():
        df1 = load_df(DEFAULT_CSV_1, text_col, label_col)
        df2 = load_df(DEFAULT_CSV_2, text_col, label_col)
        df = pd.concat([df1, df2], axis=0, ignore_index=True)
    else:
        df = load_df(csv_path, text_col, label_col)
    X_series = df[text_col].astype(str)
    y = df[label_col].astype(int).to_numpy()

    hash_func = lambda s: hashlib.md5(s.strip().encode('utf-8')).hexdigest()
    full_hashes = X_series.map(hash_func).tolist()
    hash_counts = Counter(full_hashes)
    dup_total = sum(1 for _, c in hash_counts.items() if c > 1)
    dup_ratio = dup_total / max(1, len(hash_counts))

    groups = np.array(full_hashes)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_series, y, groups=groups))

    X_train = X_series.iloc[train_idx].tolist()
    y_train = y[train_idx]
    X_test = X_series.iloc[test_idx].tolist()
    y_test = y[test_idx]

    train_hashes = {full_hashes[i] for i in train_idx}
    test_hashes = {full_hashes[i] for i in test_idx}
    overlap = train_hashes & test_hashes

    pipe = build_pipeline(model_type=model_type)

    if do_grid:
        param_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
        }
        if model_type in ("logreg", "linearsvc"):
            param_grid.update({"clf__C": [0.5, 1.0, 2.0]})
        try:
            from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            cv_method = "StratifiedGroupKFold"
            grid = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring="f1",
                cv=cv,
                n_jobs=-1,
                verbose=1,
            )
            grid.fit(X_train, y_train, groups=groups[train_idx])
        except Exception:
            cv = GroupKFold(n_splits=5)
            cv_method = "GroupKFold"
            grid = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring="f1",
                cv=cv,
                n_jobs=-1,
                verbose=1,
            )
            grid.fit(X_train, y_train, groups=groups[train_idx])
        best = grid.best_estimator_
        pipe = best
    else:
        cv_method = None
        pipe.fit(X_train, y_train)

    # Değerlendirme
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred).tolist()
    class_dist = Counter(map(int, y))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    tfidf_step = pipe.named_steps["tfidf"]
    vocab_size = len(getattr(tfidf_step, "vocabulary_", {}) or {})
    summary = {
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "samples": int(len(X_series)),
        "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
        "confusion_matrix": cm,
        "duplicates_unique_hashes": int(len(hash_counts)),
        "duplicates_repeated_hash_count": int(dup_total),
        "duplicates_ratio": round(float(dup_ratio), 6),
        "model_type": model_type,
        "random_state": RANDOM_STATE,
        "vocab_size": vocab_size,
        "cv_method": cv_method,
    }
    (REPORTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    split_payload = {
        "train_indices": list(map(int, train_idx)),
        "test_indices": list(map(int, test_idx)),
        "overlap_count": len(overlap),
        "duplicate_unique_hashes": len(hash_counts),
        "duplicate_repeated_hash_count": dup_total,
        "split_method": "GroupShuffleSplit(md5_hash)"
    }
    (REPORTS_DIR / "split_indices.json").write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    joblib.dump(pipe, model_out)
    logging.info("Model kaydedildi.")
    #print("\n=== Özet ===")
    #print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Model eğitimi başlıyor...")
    train(
        csv_path=DEFAULT_CSV,
        text_col=DEFAULT_TEXT_COL,
        label_col=DEFAULT_LABEL_COL,
        model_out=DEFAULT_MODEL_PATH,
        model_type="logreg",
        do_grid=False,
    )


if __name__ == "__main__":
    main()