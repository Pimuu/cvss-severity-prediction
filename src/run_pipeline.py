"""End-to-end reproducible pipeline: data -> split -> tune/train -> evaluate -> figures.

Usage:  python -m src.run_pipeline
All outputs are written to data/ and results/. Deterministic given config.SEED.
"""
import json
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config, evaluate, plots
from .features import FEATURE_SETS
from .generate_data import generate
from .models import build_models


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data ---------------------------------------------------------------
    df = generate()
    df.to_csv(config.DATASET_CSV, index=False)
    log(f"Generated {len(df)} records -> {config.DATASET_CSV.relative_to(config.ROOT)}")
    plots.class_distribution(df)

    # 2. Stratified train/test split ----------------------------------------
    train, test = train_test_split(df, test_size=config.TEST_SIZE, stratify=df["severity"],
                                   random_state=config.SEED)
    X_tr, y_tr, X_te, y_te = train["description"], train["severity"], test["description"], test["severity"]
    log(f"Train {len(train)} / test {len(test)}")
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)

    # 3. Tune (CV on train only) + evaluate on held-out test ----------------
    rows, per_class_rows, cms, preds, fitted = [], [], {}, {}, {}
    for name, (pipe, grid) in build_models().items():
        t = time.time()
        gs = GridSearchCV(pipe, grid or {"clf__strategy": ["most_frequent"]}, scoring="f1_macro", cv=cv,
                          n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        cv_std = gs.cv_results_["std_test_score"][gs.best_index_]
        y_hat = gs.predict(X_te)
        m = evaluate.summary_metrics(y_te, y_hat)
        ci = evaluate.bootstrap_ci(y_te, y_hat)
        rows.append({"model": name, "best_params": json.dumps(gs.best_params_),
                     "cv_macro_f1_mean": gs.best_score_, "cv_macro_f1_std": cv_std, **m,
                     "accuracy_ci_low": ci["accuracy_ci"][0], "accuracy_ci_high": ci["accuracy_ci"][1],
                     "macro_f1_ci_low": ci["macro_f1_ci"][0], "macro_f1_ci_high": ci["macro_f1_ci"][1],
                     "train_and_tune_seconds": time.time() - t})
        pc = evaluate.per_class(y_te, y_hat)
        pc.insert(0, "model", name)
        per_class_rows.append(pc)
        cms[name] = evaluate.confusion(y_te, y_hat)
        preds[name], fitted[name] = y_hat, gs.best_estimator_
        log(f"{name:20s} CV macro-F1 {gs.best_score_:.3f}±{cv_std:.3f} | test acc {m['accuracy']:.3f} "
            f"macro-F1 {m['macro_f1']:.3f} | {gs.best_params_} | {time.time() - t:.1f}s")

    summary = pd.DataFrame(rows)
    summary.to_csv(config.RESULTS_DIR / "metrics_summary.csv", index=False, float_format="%.4f")
    pd.concat(per_class_rows).to_csv(config.RESULTS_DIR / "per_class_metrics.csv", index=False, float_format="%.4f")
    cm_rows = []
    for name, cm in cms.items():
        for i, t_lab in enumerate(config.LABELS):
            cm_rows.append({"model": name, "true": t_lab, **{f"pred_{p}": int(cm[i, j])
                                                              for j, p in enumerate(config.LABELS)}})
    pd.DataFrame(cm_rows).to_csv(config.RESULTS_DIR / "confusion_matrices.csv", index=False)

    # Model selection uses CV score (never the test set).
    supervised = summary[summary["model"] != "Majority baseline"]
    best = supervised.sort_values("cv_macro_f1_mean", ascending=False).iloc[0]["model"]
    log(f"Best model by CV macro-F1: {best}")
    for name, cm in cms.items():
        if name == "Majority baseline":
            continue
        slug = name.lower().replace(" ", "_")
        plots.confusion_matrix_plot(cm, f"Confusion matrix - {name} (test)", f"confusion_matrix_{slug}.png")
    plots.confusion_matrix_plot(cms[best], f"Confusion matrix - {best} (best, test)", "confusion_matrix_best.png")
    plots.model_comparison(summary)

    # Paired bootstrap: is the best model reliably better than each other model?
    paired = {other: evaluate.paired_bootstrap(y_te, preds[best], preds[other])
              for other in preds if other != best}

    # 4. Interpretability: top LR terms per class ----------------------------
    lr = fitted["Logistic Regression"]
    vocab = lr.named_steps["tfidf"].get_feature_names_out()
    coef = pd.DataFrame(lr.named_steps["clf"].coef_.T, index=vocab, columns=lr.named_steps["clf"].classes_)
    coef = coef[config.LABELS]
    # Document frequency of each n-gram in the training set. Only n-grams seen in
    # >= MIN_DF_TERMS training descriptions are reported, so that rare n-grams
    # with large but unstable weights do not crowd out the general patterns.
    MIN_DF_TERMS = 30
    doc_freq = pd.Series(
        (lr.named_steps["tfidf"].transform(X_tr) > 0).sum(axis=0).A1, index=vocab)
    coef = coef[doc_freq >= MIN_DF_TERMS]
    plots.top_terms(coef)
    top_rows = []
    for lab in config.LABELS:
        for rank, (term, w) in enumerate(coef[lab].sort_values(ascending=False).head(20).items(), 1):
            top_rows.append({"class": lab, "rank": rank, "term": term, "coefficient": w,
                             "train_doc_freq": int(doc_freq[term])})
    pd.DataFrame(top_rows).to_csv(config.RESULTS_DIR / "top_terms_per_class.csv", index=False, float_format="%.4f")

    # 5. Feature ablation (Logistic Regression, tuned C) ---------------------
    best_c = json.loads(summary.set_index("model").loc["Logistic Regression", "best_params"])["clf__C"]
    abl = []
    for fname, make in FEATURE_SETS.items():
        t = time.time()
        p = Pipeline([("tfidf", make()), ("clf", LogisticRegression(C=best_c, max_iter=3000,
                                                                    class_weight="balanced",
                                                                    random_state=config.SEED))])
        cvs = cross_val_score(p, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
        p.fit(X_tr, y_tr)
        yh = p.predict(X_te)
        n_feat = len(p.named_steps["tfidf"].get_feature_names_out())
        abl.append({"features": fname, "n_features": n_feat, "cv_macro_f1_mean": cvs.mean(),
                    "cv_macro_f1_std": cvs.std(), "test_accuracy": evaluate.summary_metrics(y_te, yh)["accuracy"],
                    "test_macro_f1": evaluate.summary_metrics(y_te, yh)["macro_f1"], "seconds": time.time() - t})
        log(f"Ablation {fname:20s} CV {cvs.mean():.3f} test macro-F1 {abl[-1]['test_macro_f1']:.3f}")
    pd.DataFrame(abl).to_csv(config.RESULTS_DIR / "ablation_features.csv", index=False, float_format="%.4f")

    # 6. Information ceiling: classifier given the TRUE CVSS metrics ---------
    def metric_frame(d):
        f = d["cvss_vector"].str.replace("CVSS:3.1/", "").str.split("/", expand=True)
        return f.set_axis(["AV", "AC", "PR", "UI", "S", "C", "I", "A"], axis=1)
    oracle = Pipeline([("oh", OneHotEncoder(handle_unknown="ignore")),
                       ("clf", LogisticRegression(C=100, max_iter=5000, random_state=config.SEED))])
    # Interactions matter for CVSS, so use all pairwise metric combinations as features.
    def pairwise(d):
        f = metric_frame(d)
        cols = {f"{a}|{b}": f[a] + "|" + f[b] for i, a in enumerate(f.columns) for b in f.columns[i + 1:]}
        return pd.concat([f, pd.DataFrame(cols)], axis=1)
    oracle.fit(pairwise(train), y_tr)
    oracle_m = evaluate.summary_metrics(y_te, oracle.predict(pairwise(test)))

    # 7. Error analysis of the best model ------------------------------------
    te = test.assign(pred=preds[best], correct=preds[best] == test["severity"].values)
    by_class = te.groupby("vuln_class").agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                            majority_label_share=("severity",
                                                                  lambda s: s.value_counts(normalize=True).iloc[0]))
    by_class.sort_values("accuracy").to_csv(config.RESULTS_DIR / "error_by_vuln_class.csv", float_format="%.4f")
    clean = te[~te["label_noise"]]
    vague = te["description"].str.startswith("A vulnerability in ")
    extra = {
        "best_model": best,
        "best_model_test_accuracy_noise_free_subset": float(clean["correct"].mean()),
        "best_model_test_accuracy_noisy_labels_subset": float(te[te["label_noise"]]["correct"].mean()),
        "best_model_test_accuracy_vague_template": float(te[vague]["correct"].mean()),
        "best_model_test_accuracy_informative_templates": float(te[~vague]["correct"].mean()),
        "n_test": int(len(te)), "n_test_vague": int(vague.sum()), "n_test_label_noise": int(te["label_noise"].sum()),
        "oracle_true_cvss_metrics": {k: float(v) for k, v in oracle_m.items()},
        "paired_bootstrap_p_best_beats": paired,
        "class_distribution": df["severity"].value_counts().reindex(config.LABELS).to_dict(),
        "label_noise_rate_dataset": float(df["label_noise"].mean()),
        "runtime_seconds": time.time() - t0,
    }
    te[~te["correct"]][["cve_id", "description", "vuln_class", "cvss_vector", "severity", "pred"]].head(40) \
        .to_csv(config.RESULTS_DIR / "misclassified_examples.csv", index=False)
    with open(config.RESULTS_DIR / "summary.json", "w") as fh:
        json.dump(extra, fh, indent=2)

    pd.set_option("display.width", 200)
    print(summary[["model", "cv_macro_f1_mean", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                   "macro_f1_ci_low", "macro_f1_ci_high"]].round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in extra.items() if k != "class_distribution"}, indent=2))
    log(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
