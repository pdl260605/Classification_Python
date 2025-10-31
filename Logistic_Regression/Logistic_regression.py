
"""
Logistic Regression từ đầu bằng NumPy + so sánh với sklearn.
- Đọc CSV trực tiếp từ URL (mặc định: Titanic dataset).
- Tự lọc cột "bất lợi" (ID-like, missing cao, quasi-constant, trùng, high-card, collinearity)
  + có thể TỰ DROP cột rò rỉ (leakage).
- Tự nhận diện cột số / cột phân loại
- Tự chuẩn hóa nhãn về nhị phân (0/1)
- Đánh giá: Accuracy, Precision, Recall, F1, ROC-AUC, AP
- Trực quan hóa: learning curve, confusion matrix, ROC, PR, top coefficients
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression


# 1) Logistic Regression from scratch (NumPy)

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iters=2000, l2=0.0, tol=1e-7, verbose=False, random_state=42):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.w = None
        self.loss_history_ = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _add_intercept(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _loss(self, Xb, y):
        z = Xb @ self.w
        p = self._sigmoid(z)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        ce = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        l2_term = 0.5 * self.l2 * np.sum(self.w[1:] ** 2)  # không phạt intercept
        return ce + l2_term

    def fit(self, X, y):
        Xb = self._add_intercept(X)
        rng = np.random.RandomState(self.random_state)
        self.w = rng.normal(scale=0.01, size=(Xb.shape[1],))

        prev_loss = np.inf
        for i in range(self.n_iters):
            z = Xb @ self.w
            p = self._sigmoid(z)

            grad = (Xb.T @ (p - y)) / y.shape[0]
            grad[1:] += self.l2 * self.w[1:]

            self.w -= self.lr * grad

            curr_loss = self._loss(Xb, y)
            self.loss_history_.append(curr_loss)
            if self.verbose and (i % 200 == 0 or i == self.n_iters - 1):
                print(f"[Iter {i:4d}] loss={curr_loss:.5f}")
            if abs(prev_loss - curr_loss) < self.tol:
                break
            prev_loss = curr_loss
        return self

    def predict_proba(self, X):
        Xb = self._add_intercept(X)
        return self._sigmoid(Xb @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def coef_(self):
        return self.w[1:].reshape(1, -1)

    @property
    def intercept_(self):
        return np.array([self.w[0]])


# 2) Bộ lọc cột bất lợi (auto prune)

def auto_prune_columns(
    df: pd.DataFrame,
    target_col: str,
    *,
    id_ratio=0.9,
    missing_thresh=0.6,
    quasi_const_thresh=0.98,
    high_card_ratio=0.5,
    high_card_min=100,
    collinear_corr=0.98,
    leakage_corr=0.995,
    leakage_determinism=0.995,
    drop_high_card=True,
    drop_flagged_leakage=True,  
    verbose=True
):
    assert target_col in df.columns, f"'{target_col}' không có trong df"
    n = len(df)

    report = {
        "dropped_id_like": [],
        "dropped_missing": [],
        "dropped_quasi_constant": [],
        "dropped_duplicates": [],
        "dropped_collinear": [],
        "dropped_high_card": [],
        "dropped_leakage": [],
        "flagged_high_card": [],
        "flagged_leakage": [],
        "kept": []
    }

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col]

    # ID-like
    id_like = [c for c in X.columns if X[c].nunique(dropna=True) >= id_ratio * n]
    X.drop(columns=id_like, inplace=True, errors="ignore")
    report["dropped_id_like"] = id_like

    # Missing collum quá nhiều
    miss_cols = [c for c in X.columns if X[c].isna().mean() >= missing_thresh]
    X.drop(columns=miss_cols, inplace=True, errors="ignore")
    report["dropped_missing"] = miss_cols

    # Quasi-constant
    qc_cols = []
    for c in X.columns:
        vc = X[c].value_counts(dropna=True, normalize=True)
        if not vc.empty and vc.iloc[0] >= quasi_const_thresh:
            qc_cols.append(c)
    X.drop(columns=qc_cols, inplace=True, errors="ignore")
    report["dropped_quasi_constant"] = qc_cols

    # Duplicate columns (trùng hệt)
    duplicated_cols = []
    sig = {}
    for c in X.columns:
        sig[c] = pd.util.hash_pandas_object(X[c], index=False).sum()
    seen = {}
    for c, h in sig.items():
        if h in seen:
            duplicated_cols.append(c)
        else:
            seen[h] = c
    X.drop(columns=duplicated_cols, inplace=True, errors="ignore")
    report["dropped_duplicates"] = duplicated_cols

    # High-cardinality categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    high_card_cols = []
    for c in cat_cols:
        nunq = X[c].nunique(dropna=True)
        if nunq > max(high_card_min, int(high_card_ratio * n)):
            high_card_cols.append(c)
    if drop_high_card and high_card_cols:
        X.drop(columns=high_card_cols, inplace=True, errors="ignore")
        report["dropped_high_card"] = high_card_cols
    else:
        report["flagged_high_card"] = high_card_cols  # gợi ý xử lý riêng (hashing/target encoding)

    # Collinearity (numeric)
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    drop_collinear = set()
    if len(num_cols) >= 2:
        corr = X[num_cols].corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        for c in upper.columns:
            high = upper[c][upper[c] >= collinear_corr].index.tolist()
            for h in high:
                c1, c2 = c, h
                miss1 = X[c1].isna().mean()
                miss2 = X[c2].isna().mean()
                drop = c1 if miss1 > miss2 else c2  # bỏ cột "xấu" hơn
                drop_collinear.add(drop)
        if drop_collinear:
            X.drop(columns=list(drop_collinear), inplace=True, errors="ignore")
            report["dropped_collinear"] = sorted(list(drop_collinear))

    # Leakage detector
    flagged = set()
    if pd.api.types.is_numeric_dtype(y):
        y_num = pd.to_numeric(y, errors="coerce")
        for c in X.select_dtypes(include=["number"]).columns:
            x_num = pd.to_numeric(X[c], errors="coerce")
            r = np.corrcoef(x_num.fillna(x_num.median()), y_num.fillna(y_num.median()))[0, 1]
            if pd.notna(r) and abs(r) >= leakage_corr:
                flagged.add(c)

    # Deterministic theo nhóm 
    for c in X.columns:
        grp = df[[c, target_col]].dropna()
        if grp.empty:
            continue
        stat = grp.groupby(c)[target_col].nunique()
        frac_rows_in_det_groups = grp[grp[c].isin(stat[stat == 1].index)].shape[0] / max(1, grp.shape[0])
        if frac_rows_in_det_groups >= leakage_determinism:
            flagged.add(c)

    report["flagged_leakage"] = sorted(list(flagged))

    # Drop leakage
    if drop_flagged_leakage and flagged:
        X.drop(columns=list(flagged), inplace=True, errors="ignore")
        report["dropped_leakage"] = sorted(list(flagged))

    kept_cols = [c for c in X.columns]
    report["kept"] = kept_cols

    if verbose:
        print("\n[auto_prune_columns] dropped_id_like     :", report["dropped_id_like"])
        print("[auto_prune_columns] dropped_missing     :", report["dropped_missing"])
        print("[auto_prune_columns] dropped_quasi_const :", report["dropped_quasi_constant"])
        print("[auto_prune_columns] dropped_duplicates  :", report["dropped_duplicates"])
        print("[auto_prune_columns] dropped_collinear   :", report["dropped_collinear"])
        print("[auto_prune_columns] dropped_high_card   :", report["dropped_high_card"])
        print("[auto_prune_columns] dropped_leakage     :", report["dropped_leakage"])
        print("[auto_prune_columns] flagged_high_card   :", report["flagged_high_card"])
        print("[auto_prune_columns] flagged_leakage     :", report["flagged_leakage"])
        print(f"[auto_prune_columns] kept_cols (#{len(kept_cols)}):",
              kept_cols[:15], "..." if len(kept_cols) > 15 else "")

    return pd.concat([X, df[[target_col]]], axis=1), report


# 3) Tự chọn cột số/categorical

def auto_select_columns(df, target_col, max_cat_unique=20, id_threshold_ratio=0.9):
    assert target_col in df.columns, f"target_col '{target_col}' không có trong DataFrame!"
    n = len(df)
    Xdf = df.drop(columns=[target_col]).copy()

    maybe_id_cols = [c for c in Xdf.columns if Xdf[c].nunique(dropna=True) >= id_threshold_ratio * n]
    Xdf = Xdf.drop(columns=maybe_id_cols) if maybe_id_cols else Xdf

    num_cols = Xdf.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = Xdf.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    moved_num_to_cat = []
    for c in num_cols[:]:
        nunq = Xdf[c].nunique(dropna=True)
        if pd.api.types.is_integer_dtype(Xdf[c].dropna()) and nunq <= max_cat_unique:
            moved_num_to_cat.append(c)
            num_cols.remove(c)
            cat_cols.append(c)

    return num_cols, cat_cols, maybe_id_cols, moved_num_to_cat


# 4) Trực quan hóa

def plot_learning_curve(losses):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Binary Cross-Entropy)")
    plt.title("Learning Curve (LogReg from scratch)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names=("0", "1"), title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names
    )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    fig.tight_layout()
    plt.show()


def plot_roc_pr_curves(y_test, proba_scratch, proba_sklearn):
    fpr_s, tpr_s, _ = roc_curve(y_test, proba_scratch)
    fpr_k, tpr_k, _ = roc_curve(y_test, proba_sklearn)
    auc_s = roc_auc_score(y_test, proba_scratch)
    auc_k = roc_auc_score(y_test, proba_sklearn)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr_s, tpr_s, label=f"Scratch (AUC={auc_s:.3f})")
    plt.plot(fpr_k, tpr_k, label=f"Sklearn (AUC={auc_k:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    prec_s, rec_s, _ = precision_recall_curve(y_test, proba_scratch)
    prec_k, rec_k, _ = precision_recall_curve(y_test, proba_sklearn)
    ap_s = average_precision_score(y_test, proba_scratch)
    ap_k = average_precision_score(y_test, proba_sklearn)

    plt.figure(figsize=(6, 4))
    plt.plot(rec_s, prec_s, label=f"Scratch (AP={ap_s:.3f})")
    plt.plot(rec_k, prec_k, label=f"Sklearn (AP={ap_k:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_top_coefficients(feature_names, coefs, title="Top hệ số (|coef|)"):
    abs_coefs = np.abs(coefs)
    topk = min(15, len(abs_coefs))
    idx = np.argsort(abs_coefs)[-topk:][::-1]
    names_top = feature_names[idx]
    vals_top = coefs[idx]

    plt.figure(figsize=(8, max(4, 0.35 * topk)))
    y = np.arange(topk)
    plt.barh(y, vals_top)
    plt.yticks(y, names_top)
    plt.gca().invert_yaxis()
    plt.xlabel("Coefficient")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# 4.1) Làm đẹp tên hiển thị

def prettify_feature_names(feature_names, numeric_hint=None):

    if numeric_hint is None:
        numeric_hint = set()
    else:
        numeric_hint = set(numeric_hint)

    nice_map = {
        "sibsp": "SibSp",
        "parch": "Parch",
        "pclass": "Pclass",
        "adult_male": "Adult male",
        "who": "Person type",
        "embarked": "Embarked",
        "sex": "Sex",
        "fare": "Fare",
        "age": "Age",
        "class": "Class",
        "alone": "Alone",
    }

    pretty = []
    for raw in feature_names:
        # Bỏ tiền tố "num__"/"cat__" nếu có
        base = raw.split("__", 1)[-1]

        label = base
        if "_" in base:
            feat, val = base.rsplit("_", 1)
            feat_nice = nice_map.get(feat, feat.replace("_", " ").title())
            label = f"{feat_nice} = {val}"
        else:
            feat = base
            feat_nice = nice_map.get(feat, feat.replace("_", " ").title())
            label = feat_nice
            if feat in numeric_hint:
                label = f"{label} (std)"

        # tinh chỉnh giá trị thường gặp
        label = (label
                 .replace("= male", "= Male").replace("= female", "= Female")
                 .replace("= man", "= Man").replace("= woman", "= Woman").replace("= child", "= Child")
                 .replace("= first", "= First").replace("= second", "= Second").replace("= third", "= Third"))
        pretty.append(label)

    return np.array(pretty)


# 5) Chuẩn hóa nhãn về nhị phân 0/1 (bền với nhiều dataset)

def coerce_binary_labels(series, use_median=True, max_non_numeric_frac=0.05):
    """
    - Nếu nhãn ép được về số: 
        * Nếu chỉ gồm {0,1} -> giữ nguyên
        * Ngược lại -> nhị phân hóa theo median (mặc định: > median => 1)
    - Nếu không phải số: map theo tập pos/neg; nếu không map được -> lỗi
    """
    s = series.copy()

    s_num = pd.to_numeric(s, errors="coerce")
    non_num_frac = s_num.isna().mean()

    if non_num_frac <= max_non_numeric_frac:
        # Numeric route
        s_num = s_num.fillna(s_num.median())
        vals = pd.Series(s_num.dropna().unique()).astype(float)
        if set(np.unique(vals)).issubset({0.0, 1.0}):
            return s_num.astype(int).values
        if use_median:
            thr = float(s_num.median())
            return (s_num > thr).astype(int).values
        else:
            return (s_num > 0).astype(int).values

    # Text route
    s_str = s.astype(str).str.strip().str.lower()
    pos = {"1", "true", "yes", "y", "t", "positive", "pos", "survived", "spam"}
    neg = {"0", "false", "no", "n", "f", "negative", "neg", "dead", "ham"}

    def map_text(x):
        if x in pos: return 1
        if x in neg: return 0
        raise ValueError(f"Không map được nhãn '{x}' sang 0/1. Hãy chuẩn hóa nhãn hoặc cung cấp mapping riêng.")

    return s_str.apply(map_text).astype(int).values


# 6) Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str,
                        default="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
                        help="URL CSV dataset")
    parser.add_argument("--target", type=str, default="survived", help="Tên cột nhãn (target)")
    parser.add_argument("--sep", type=str, default=",", help="Separator CSV (vd: ';' cho bank marketing)")
    parser.add_argument("--drop-high-card", action="store_true", help="Drop cột categorical high-cardinality")
    args = parser.parse_args()

    url = args.url
    target_col = args.target
    sep = args.sep
    drop_high_card = args.drop_high_card

    # Đọc CSV từ URL
    df = pd.read_csv(url, sep=sep)

    # Lọc cột bất lợi tự động (có drop leakage)
    df, _ = auto_prune_columns(
        df, target_col=target_col,
        drop_high_card=drop_high_card,
        drop_flagged_leakage=True,
        verbose=True
    )

    # Tự động chọn cột số / phân loại
    features_num, features_cat, dropped_ids, moved_num_to_cat = auto_select_columns(df, target_col)
    print("\n[auto_select_columns]")
    print("  Numeric      :", features_num)
    print("  Categorical  :", features_cat)
    if dropped_ids:       print("  Dropped ID-like:", dropped_ids)
    if moved_num_to_cat:  print("  Numeric→Categorical:", moved_num_to_cat)

    # Tạo X, y (nhãn nhị phân tự động)
    X = df[features_num + features_cat].copy()
    y = coerce_binary_labels(df[target_col])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Preprocess
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                          ("scaler", StandardScaler())])
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary")
    except TypeError:
        try:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse=False, drop="if_binary")
        except TypeError:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                              ("onehot", onehot)])

    transformers = []
    if features_num: transformers.append(("num", numeric_transformer, features_num))
    if features_cat: transformers.append(("cat", categorical_transformer, features_cat))
    preprocess = ColumnTransformer(transformers=transformers) if transformers else "passthrough"

    X_train_prep = preprocess.fit_transform(X_train)
    X_test_prep = preprocess.transform(X_test)

    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        try:
            names = []
            if features_num:
                names += [f"num__{c}" for c in features_num]
            if features_cat:
                ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
                names += list(ohe.get_feature_names_out(features_cat))
            feature_names = np.array(names)
        except Exception:
            feature_names = np.array([f"f{i}" for i in range(X_train_prep.shape[1])])

    feature_names_pretty = prettify_feature_names(feature_names, numeric_hint=features_num)

    # Train: scratch
    scratch = LogisticRegressionScratch(lr=0.1, n_iters=3000, l2=0.0, verbose=False, random_state=42)
    scratch.fit(X_train_prep, y_train)
    proba_scratch = scratch.predict_proba(X_test_prep)
    y_pred_scratch = (proba_scratch >= 0.5).astype(int)

    # Train: sklearn
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(X_train_prep, y_train)
    proba_sklearn = lr.predict_proba(X_test_prep)[:, 1]
    y_pred_sklearn = (proba_sklearn >= 0.5).astype(int)

    # Đánh giá
    def eval_and_print(name, y_true, y_pred, proba=None):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        txt = f"[{name}] Acc={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}"
        if proba is not None:
            auc = roc_auc_score(y_true, proba)
            ap = average_precision_score(y_true, proba)
            txt += f"  AUC={auc:.4f}  AP={ap:.4f}"
        print(txt)
        print(classification_report(y_true, y_pred, digits=4))

    print("\n=== Hiệu năng trên tập test ===")
    eval_and_print("Scratch", y_test, y_pred_scratch, proba_scratch)
    eval_and_print("Sklearn", y_test, y_pred_sklearn, proba_sklearn)

    # Trực quan hóa
    plot_learning_curve(scratch.loss_history_)
    cm_scratch = confusion_matrix(y_test, y_pred_scratch)
    cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
    classes = ("Negative(0)", "Positive(1)")
    plot_confusion_matrix(cm_scratch, class_names=classes, title="Confusion Matrix - Scratch")
    plot_confusion_matrix(cm_sklearn, class_names=classes, title="Confusion Matrix - Sklearn")
    plot_roc_pr_curves(y_test, proba_scratch, proba_sklearn)

    # Top hệ số
    coef_scratch = scratch.coef_.ravel()
    coef_sklearn = lr.coef_.ravel()
    plot_top_coefficients(feature_names_pretty, coef_scratch, title="Top hệ số (|coef|) - Scratch")
    plot_top_coefficients(feature_names_pretty, coef_sklearn, title="Top hệ số (|coef|) - Sklearn")


if __name__ == "__main__":
    main()
