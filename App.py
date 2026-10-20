"""
Customer Churn Prediction
=========================
Run with:   streamlit run app.py      (or double-click run_app.bat / run_app.command)
Needs:      scaler.pkl and model.pkl next to this file, OR train a model in the "Train" tab.

Model input order (Age last):
    [Gender (Female=1, Male=0), Tenure, MonthlyCharges, Age]

Light / dark: use the app menu (top right) -> Settings -> Theme (System, Light or Dark).
"""
from __future__ import annotations

import inspect
import io
import re
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

# iOS system colours
BLUE, GREEN, ORANGE, RED = "#007AFF", "#34C759", "#FF9500", "#FF3B30"
GRID = "rgba(128,128,128,.22)"
FONT_STACK = ('-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", '
              '"Helvetica Neue", Inter, system-ui, sans-serif')

# --------------------------------------------------------------------------
# Theme file (light + dark). Streamlit reads colours only from .streamlit/config.toml.
# A config that forces base="light" removes the System/Light/Dark switch from the menu,
# so an old auto-generated config of that kind is upgraded automatically.
# --------------------------------------------------------------------------
THEME_MARK = "# churn-app-theme v2"
THEME_TOML = f"""{THEME_MARK}  (light + dark, switch in the app menu: Settings > Theme)
[theme]
primaryColor = "#007AFF"

[theme.light]
backgroundColor = "#F2F2F7"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#1C1C1E"

[theme.dark]
primaryColor = "#0A84FF"
backgroundColor = "#000000"
secondaryBackgroundColor = "#1C1C1E"
textColor = "#F2F2F7"
"""


def ensure_theme() -> bool:
    """Create or upgrade .streamlit/config.toml. Returns True if a refresh is needed."""
    try:
        paths = [Path.cwd() / ".streamlit" / "config.toml", BASE_DIR / ".streamlit" / "config.toml"]
        found = [p for p in dict.fromkeys(paths) if p.exists()]
        for p in found:
            text = p.read_text()
            if THEME_MARK in text:
                return False
            old_light_only = ('base = "light"' in text or "base='light'" in text) \
                and "#007AFF" in text and "theme.light" not in text
            if old_light_only:
                p.write_text(THEME_TOML)
                return True
        if found or (Path.home() / ".streamlit" / "config.toml").exists():
            return False                       # the user's own config: leave it alone
        paths[0].parent.mkdir(exist_ok=True)
        paths[0].write_text(THEME_TOML)
        return True
    except Exception:                          # read-only folder: just skip
        return False


if ensure_theme() and not st.session_state.get("theme_notice"):
    st.session_state["theme_notice"] = True
    st.toast("Theme updated. Refresh the page once to switch between light and dark.")


# --------------------------------------------------------------------------
# Streamlit version helper: "use_container_width" was replaced by width="stretch"
# --------------------------------------------------------------------------
def _st_version() -> tuple:
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


def stretch(fn) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if _st_version() >= (1, 50) and "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    if "width" in params:
        return {"width": "stretch"}
    return {}


def show_chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, key=key, **stretch(st.plotly_chart))


def usd(value: float) -> str:
    """Dollar amount that is safe inside st.markdown HTML (no LaTeX clash)."""
    return f"&#36;{value:,.0f}"


# --------------------------------------------------------------------------
# Styling: neutral, iOS-like, works in light and dark
# --------------------------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]) {{
    font-family: {FONT_STACK} !important;
    -webkit-font-smoothing: antialiased;
}}
.block-container {{padding-top: 4rem; padding-bottom: 2.5rem; max-width: 1180px;}}
.stApp h3 {{font-weight: 600; letter-spacing: -0.01em;}}

/* chart text follows the theme text colour (readable in dark mode) */
.js-plotly-plot .plotly text {{fill: currentColor !important;}}

.title {{font-size: 2.1rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.15;}}
.subtitle {{font-size: 1.02rem; opacity: .6; margin: 4px 0 18px 0;}}

/* cards take their tint from the text colour, so they work on light and dark backgrounds */
.card, .result, .kpi, .empty {{background: rgba(120,120,128,.12);
    background: color-mix(in srgb, currentColor 7%, transparent);}}
.card {{border-radius: 16px; padding: 4px 18px; margin-bottom: 12px;}}
.row {{display: flex; gap: 14px; align-items: flex-start; padding: 12px 0;
       border-bottom: 1px solid rgba(120,120,128,.25);
       border-bottom: 1px solid color-mix(in srgb, currentColor 14%, transparent);}}
.row:last-child {{border-bottom: none;}}
.row-label {{flex: 0 0 150px; font-size: .88rem; opacity: .6; padding-top: 1px;}}
.row-text {{flex: 1; min-width: 0; overflow-wrap: anywhere; font-size: .98rem; line-height: 1.45;}}
.num {{flex: 0 0 26px; height: 26px; border-radius: 50%; background: {BLUE}; color: #fff;
       font-size: .8rem; font-weight: 600; display: flex; align-items: center; justify-content: center;}}

.result {{display: flex; align-items: center; gap: 14px; padding: 16px 20px; border-radius: 16px;
          margin-bottom: 8px;}}
.dot {{width: 14px; height: 14px; border-radius: 50%; flex: 0 0 14px;}}
.result-title {{font-size: 1.25rem; font-weight: 600; letter-spacing: -0.01em;}}
.result-sub {{font-size: .92rem; opacity: .6;}}

.kpi {{border-radius: 14px; padding: 14px 16px; height: 100%; margin-bottom: 10px;}}
.kpi-label {{font-size: .78rem; opacity: .6;}}
.kpi-value {{font-size: 1.5rem; font-weight: 600; letter-spacing: -0.02em; margin-top: 2px;}}
.kpi-sub {{font-size: .78rem; opacity: .5;}}

.empty {{border-radius: 16px; padding: 38px 24px; text-align: center;}}
.empty b {{font-size: 1.1rem;}}
.empty div {{opacity: .6; margin-top: 4px;}}

.pill {{display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
        font-weight: 600; color: #fff; margin-left: 8px; vertical-align: middle;}}
.section-note {{font-size: .9rem; opacity: .6; margin: -4px 0 10px 0;}}

.side .row-label {{flex: 0 0 72px; font-size: .8rem;}}
.side .row {{padding: 7px 0;}}
.side .row-text {{font-size: .85rem;}}
.stButton > button, .stDownloadButton > button {{border-radius: 12px; font-weight: 600;}}
[data-testid="stBaseButton-primary"], button[kind="primary"] {{
    background-color: {BLUE} !important; border-color: {BLUE} !important; color: #fff !important;}}
div[role="slider"] {{background-color: {BLUE} !important;}}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Pure helpers (no Streamlit state): easy to test
# --------------------------------------------------------------------------
def encode_gender(g) -> int:
    return 1 if str(g).strip().lower() in ("female", "f") else 0


def make_X(gender_enc, tenure, monthly, age, use_age: bool) -> np.ndarray:
    cols = [gender_enc, tenure, monthly] + ([age] if use_age else [])
    arrays = np.broadcast_arrays(*[np.atleast_1d(np.asarray(c, dtype=float)) for c in cols])
    return np.column_stack(arrays)


def score_with(scaler, model, X: np.ndarray) -> np.ndarray:
    """Churn risk in % for every row of X."""
    Xs = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)
        classes = list(getattr(model, "classes_", [0, 1]))
        idx = len(classes) - 1
        for candidate in (1, True, "1", "Yes", "yes"):
            if candidate in classes:
                idx = classes.index(candidate)
                break
        return proba[:, idx] * 100.0
    pred = np.asarray(model.predict(Xs))
    return np.where(np.isin(pred, [1, True, "1", "Yes", "yes"]), 100.0, 0.0)


# Discount rule: risk of 75% or more gets a big discount, anything below it a small one.
HIGH_RISK_CUTOFF = 75
DISCOUNT_HIGH = (30, 40)    # % off when risk >= HIGH_RISK_CUTOFF
DISCOUNT_LOW = (5, 10)      # % off when risk is below HIGH_RISK_CUTOFF (but above the churn threshold)


def discount_range(risk: float) -> tuple[int, int]:
    return DISCOUNT_HIGH if risk >= HIGH_RISK_CUTOFF else DISCOUNT_LOW


def risk_level(risk: float) -> tuple[str, str]:
    if risk >= HIGH_RISK_CUTOFF:
        return "High", RED
    if risk >= 30:
        return "Medium", ORANGE
    return "Low", GREEN


PRIORITY_ACTION = {
    "Critical": f"Call within 24 hours + {DISCOUNT_HIGH[0]}–{DISCOUNT_HIGH[1]}% discount",
    "High": f"Call within 3 days + {DISCOUNT_HIGH[0]}–{DISCOUNT_HIGH[1]}% discount",
    "Moderate": f"Call within 7 days + {DISCOUNT_LOW[0]}–{DISCOUNT_LOW[1]}% discount",
    "No action": "No action needed",
}


def priority_label(risk: float, threshold: float) -> str:
    if risk < threshold:
        return "No action"
    if risk >= 85:
        return "Critical"
    if risk >= HIGH_RISK_CUTOFF:
        return "High"
    return "Moderate"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        key = re.sub(r"[^a-z]", "", str(col).lower())
        if key == "gender":
            mapping[col] = "Gender"
        elif key == "tenure":
            mapping[col] = "Tenure"
        elif key in ("monthlycharges", "monthlycharge"):
            mapping[col] = "MonthlyCharges"
        elif key == "age":
            mapping[col] = "Age"
        elif key == "churn":
            mapping[col] = "Churn"
    return df.rename(columns=mapping)


def parse_target(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.map({"yes": 1, "y": 1, "1": 1, "1.0": 1, "true": 1, "churn": 1, "churned": 1,
                  "no": 0, "n": 0, "0": 0, "0.0": 0, "false": 0, "stay": 0, "stayed": 0,
                  "retained": 0})


def validate_rows(raw: pd.DataFrame, use_age: bool, need_target: bool = False):
    """Return (valid_df, skipped_df). valid_df has numeric columns, '_g' (gender code), '_y' (target)."""
    df = normalize_columns(raw).reset_index(drop=True)
    numeric = ["Tenure", "MonthlyCharges"] + (["Age"] if use_age else [])
    required = ["Gender"] + numeric + (["Churn"] if need_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}. Required: {', '.join(required)}.")

    problems = pd.Series("", index=df.index, dtype=object)

    def flag(mask: pd.Series, text: str) -> None:
        problems.loc[mask] = problems.loc[mask] + text + "; "

    gender_code = df["Gender"].astype(str).str.strip().str.lower().map(
        {"female": 1, "f": 1, "male": 0, "m": 0})
    flag(gender_code.isna(), "Gender must be Male or Female")

    nums = {}
    for c in numeric:
        nums[c] = pd.to_numeric(df[c], errors="coerce")
        flag(nums[c].isna(), f"{c} is not a number")
    flag(nums["Tenure"] < 0, "Tenure cannot be negative")
    flag(nums["MonthlyCharges"] < 0, "MonthlyCharges cannot be negative")
    if use_age:
        flag((nums["Age"] < 0) | (nums["Age"] > 120), "Age must be between 0 and 120")

    target = None
    if need_target:
        target = parse_target(df["Churn"])
        flag(target.isna(), "Churn must be Yes/No or 1/0")

    ok = problems == ""
    valid = df.loc[ok].copy()
    for c in numeric:
        valid[c] = nums[c].loc[ok]
    valid["_g"] = gender_code.loc[ok].astype(int)
    if need_target:
        valid["_y"] = target.loc[ok].astype(int)
    skipped = raw.reset_index(drop=True).loc[~ok].copy()
    skipped.insert(0, "Reason", problems.loc[~ok].str.rstrip("; "))
    return valid.reset_index(drop=True), skipped.reset_index(drop=True)


def run_batch(raw: pd.DataFrame, threshold: float, scaler, model, use_age: bool):
    valid, skipped = validate_rows(raw, use_age)
    if valid.empty:
        raise ValueError("No valid rows found. Gender must be Male/Female and the other columns numeric.")
    age_vals = valid["Age"].to_numpy() if use_age else np.zeros(len(valid))
    risk = score_with(scaler, model, make_X(valid["_g"].to_numpy(), valid["Tenure"].to_numpy(),
                                            valid["MonthlyCharges"].to_numpy(), age_vals, use_age))
    out = valid.drop(columns="_g").copy()
    out["Risk %"] = np.round(risk, 1)
    out["Prediction"] = np.where(risk >= threshold, "Yes", "No")
    out["Priority"] = [priority_label(r, threshold) for r in risk]
    out["Suggested action"] = out["Priority"].map(PRIORITY_ACTION)
    out["Expected loss / year"] = np.round(risk / 100.0 * out["MonthlyCharges"] * 12, 0)
    # show the ID first and the results next, so they are visible without scrolling sideways
    id_cols = [c for c in out.columns if re.sub(r"[^a-z]", "", str(c).lower()) in ("customerid", "id", "customer")]
    result_cols = ["Risk %", "Priority", "Suggested action", "Expected loss / year"]
    rest = [c for c in out.columns if c not in id_cols + result_cols]
    return out[id_cols + result_cols + rest].reset_index(drop=True), skipped


def segment_table(out: pd.DataFrame, by: str) -> pd.DataFrame | None:
    df = out.copy()
    if by == "Tenure":
        df["Segment"] = pd.cut(df["Tenure"], [-1, 6, 12, 24, 48, 10**6],
                               labels=["0–6 mo", "7–12 mo", "13–24 mo", "25–48 mo", "49+ mo"])
    elif by == "Monthly charge":
        df["Segment"] = pd.cut(df["MonthlyCharges"], [-1, 40, 70, 100, 10**6],
                               labels=["Under 40", "40–70", "70–100", "Over 100"])
    elif by == "Age":
        if "Age" not in df.columns:
            return None
        age = pd.to_numeric(df["Age"], errors="coerce")
        if age.isna().all():
            return None
        df["Segment"] = pd.cut(age, [-1, 29, 44, 59, 200], labels=["Under 30", "30–44", "45–59", "60+"])
    else:
        df["Segment"] = df["Gender"].astype(str).str.strip().str.title()
    df["_rev"] = np.where(df["Prediction"] == "Yes", df["MonthlyCharges"] * 12, 0.0)
    df["_yes"] = (df["Prediction"] == "Yes").astype(float) * 100
    table = df.groupby("Segment", observed=True).agg(
        Customers=("Risk %", "size"), avg_risk=("Risk %", "mean"),
        likely=("_yes", "mean"), revenue=("_rev", "sum")).reset_index()
    table.columns = ["Segment", "Customers", "Average risk %", "Likely to churn %", "Yearly revenue at risk"]
    return table.round(1)


def prepare_training(raw: pd.DataFrame):
    """Validate an uploaded training file. Returns (valid_df, skipped_df, use_age)."""
    df = normalize_columns(raw)
    use_age = "Age" in df.columns
    valid, skipped = validate_rows(raw, use_age, need_target=True)
    return valid, skipped, use_age


def make_estimators() -> dict:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42,
                                                class_weight="balanced_subsample", n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    }


def feature_columns(use_age: bool) -> list:
    return ["_g", "Tenure", "MonthlyCharges"] + (["Age"] if use_age else [])


def train_candidates(valid: pd.DataFrame, use_age: bool, seed: int = 42) -> dict:
    """Train 3 model types on 80% of the rows and score them on the held-out 20%."""
    X = valid[feature_columns(use_age)].to_numpy(dtype=float)
    y = valid["_y"].to_numpy(dtype=int)
    n, pos = len(y), int(y.sum())
    neg = n - pos
    if n < 50:
        raise ValueError(f"Need at least 50 valid rows to train (found {n}).")
    if min(pos, neg) < 10:
        raise ValueError(f"Need at least 10 customers in each group (churned: {pos}, stayed: {neg}).")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    results = {}
    for name, est in make_estimators().items():
        est.fit(Xs_tr, y_tr)
        proba = est.predict_proba(Xs_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "model": est,
            "metrics": {
                "ROC-AUC": round(float(roc_auc_score(y_te, proba)), 3),
                "Accuracy": round(float(accuracy_score(y_te, pred)), 3),
                "Precision": round(float(precision_score(y_te, pred, zero_division=0)), 3),
                "Recall": round(float(recall_score(y_te, pred, zero_division=0)), 3),
                "F1": round(float(f1_score(y_te, pred, zero_division=0)), 3),
            },
            "cm": confusion_matrix(y_te, pred, labels=[0, 1]).tolist(),
        }
    return {"results": results, "n_train": len(y_tr), "n_test": len(y_te), "n_rows": n,
            "churn_rate": pos / n * 100}


def fit_final(name: str, valid: pd.DataFrame, use_age: bool):
    """Refit the chosen model type on ALL valid rows for use in the app."""
    X = valid[feature_columns(use_age)].to_numpy(dtype=float)
    y = valid["_y"].to_numpy(dtype=int)
    scaler = StandardScaler().fit(X)
    model = make_estimators()[name].fit(scaler.transform(X), y)
    return scaler, model


def model_importance(model) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        return np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    return None


# --------------------------------------------------------------------------
# Active model: your own trained model (this session) or the saved files
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_default_artifacts():
    return joblib.load(BASE_DIR / "scaler.pkl"), joblib.load(BASE_DIR / "model.pkl")


def get_active_model():
    custom = st.session_state.get("custom_model")
    if custom:
        return custom["scaler"], custom["model"], None
    try:
        scaler, model = load_default_artifacts()
        return scaler, model, None
    except FileNotFoundError:
        return None, None, ("Couldn't find 'scaler.pkl' and/or 'model.pkl' next to this script. "
                            "Train a model in the Train tab, or add both files.")
    except Exception as exc:
        return None, None, f"The model files could not be loaded: {exc}"


SCALER, MODEL, MODEL_PROBLEM = get_active_model()
N_FEATURES = 4
if SCALER is not None:
    for _obj in (SCALER, MODEL):
        _n = getattr(_obj, "n_features_in_", None)
        if _n:
            N_FEATURES = int(_n)
            break
    if N_FEATURES not in (3, 4):
        MODEL_PROBLEM = (f"The saved model expects {N_FEATURES} input features, but this app supports 3 "
                         "(Gender, Tenure, Monthly Charge) or 4 (adds Age). Retrain it in the Train tab.")
        SCALER = MODEL = None
HAVE_MODEL = SCALER is not None
USES_AGE = N_FEATURES == 4
FEATURE_NAMES = ["Gender", "Tenure", "Monthly Charge", "Age"][:N_FEATURES]


def build_X(g, t, m, a) -> np.ndarray:
    return make_X(g, t, m, a, USES_AGE)


def score(X: np.ndarray) -> np.ndarray:
    return score_with(SCALER, MODEL, X)


# --------------------------------------------------------------------------
# Business decision (plain, line-by-line)
# --------------------------------------------------------------------------
def build_decision(risk: float, threshold: float, inputs: dict, budget_pct: float) -> dict:
    monthly, tenure, age = inputs["monthly"], inputs["tenure"], inputs["age"]
    annual = monthly * 12
    if risk < threshold:
        return {
            "churn": False,
            "summary": [
                ("Decision", "<b>No action needed.</b> Do not spend retention budget on this customer."),
                ("Why", f"Churn risk is {risk:.1f}%, which is below your {threshold:.0f}% threshold."),
                ("Retention spend", usd(0)),
                ("Next review", "Run the prediction again if the customer's plan, bill or tenure changes."),
            ],
            "steps": [],
        }

    priority = priority_label(risk, threshold)
    color = {"Critical": RED, "High": ORANGE}.get(priority, BLUE)
    when = {"Critical": "within 24 hours", "High": "within 3 days"}.get(priority, "within 7 days")
    d_lo, d_hi = discount_range(risk)
    disc = f"{d_lo}–{d_hi}%"
    cost_lo, cost_hi = monthly * d_lo / 100 * 3, monthly * d_hi / 100 * 6    # 3 to 6 months of discount
    budget_amt = annual * budget_pct / 100
    offer_text = (f"Offer a {disc} discount for 3 to 6 months. "
                  f"Estimated cost: {usd(cost_lo)} to {usd(cost_hi)}.")
    if cost_hi > budget_amt:
        offer_text += (f" The top of that range is above your retention budget of {usd(budget_amt)}, "
                       "so start at the lower end.")

    steps = [
        ("Call the customer", f"The retention team contacts the customer {when}."),
        ("Make an offer", offer_text),
    ]
    if tenure < 12:
        steps.append(("Check the new-customer experience",
                      f"They joined only {tenure} month(s) ago. Ask how it is going and fix any set-up problems."))
    elif tenure >= 36:
        steps.append(("Recognise loyalty",
                      f"They have stayed {tenure} months. Thank them and offer a loyalty reward or priority support."))
    if monthly >= 90:
        steps.append(("Review the bill",
                      f"Their monthly charge is {usd(monthly)}. Check their usage and offer a cheaper plan if it fits."))
    if USES_AGE and age >= 60:
        steps.append(("Make support easy", "Offer a dedicated phone support line and simple billing."))
    elif USES_AGE and age < 30:
        steps.append(("Use digital channels", "Send the offer by app or SMS, with flexible short-term options."))
    steps.append(("Offer a longer contract", "Give a small discount if they move to a 1-year or 2-year contract."))
    steps.append(("Follow up", "Record the result in your CRM and run this prediction again after 30 days."))

    return {
        "churn": True,
        "summary": [
            ("Decision", f"<b>Retain this customer.</b><span class='pill' style='background:{color}'>{priority} priority</span>"),
            ("Why", f"Churn risk is {risk:.1f}%, which is at or above your {threshold:.0f}% threshold."),
            ("What is at stake", f"{usd(annual)} in revenue per year (12 months × {usd(monthly)})."),
            ("Retention budget", f"Spend up to {usd(annual * budget_pct / 100)} ({budget_pct:.0f}% of that revenue)."),
        ],
        "steps": steps,
    }


def rows_html(rows: list) -> str:
    body = "".join(f'<div class="row"><div class="row-label">{label}</div>'
                   f'<div class="row-text">{text}</div></div>' for label, text in rows)
    return f'<div class="card">{body}</div>'


def steps_html(steps: list) -> str:
    body = "".join(f'<div class="row"><div class="num">{i}</div><div class="row-text">'
                   f'<b>{title}</b><br>{text}</div></div>' for i, (title, text) in enumerate(steps, 1))
    return f'<div class="card">{body}</div>'


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def _layout(fig: go.Figure, height: int, **kwargs) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT_STACK), **kwargs)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def render_gauge(risk: float, threshold: float, key: str) -> None:
    _, color = risk_level(risk)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Churn risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(120,120,128,.18)",
            "borderwidth": 0,
            "threshold": {"line": {"color": "#8E8E93", "width": 3}, "thickness": 0.85, "value": threshold},
        },
    ))
    show_chart(_layout(fig, 250, margin=dict(l=20, r=20, t=50, b=10)), key)


def render_importance(key: str) -> None:
    values = model_importance(MODEL)
    if values is None or len(values) != len(FEATURE_NAMES):
        return
    df = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": values}).sort_values("Importance")
    fig = go.Figure(go.Bar(x=df["Importance"], y=df["Feature"], orientation="h", marker_color=BLUE))
    show_chart(_layout(fig, 240, margin=dict(l=10, r=10, t=10, b=10)), key)


def sweep_chart(var: str, values, title: str, current: float, inputs: dict,
                threshold: float, key: str) -> None:
    g = encode_gender(inputs["gender"])
    t, m, a = inputs["tenure"], inputs["monthly"], inputs["age"]
    vals = np.asarray(values, dtype=float)
    if var == "tenure":
        X = build_X(g, vals, m, a)
    elif var == "monthly":
        X = build_X(g, t, vals, a)
    else:
        X = build_X(g, t, m, vals)
    y = score(X)
    now = float(score(build_X(g, t, m, a))[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vals, y=y, mode="lines", line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=[current], y=[now], mode="markers",
                             marker=dict(size=12, color=RED, line=dict(color="white", width=2))))
    fig.add_hline(y=threshold, line_dash="dot", line_color="#8E8E93")
    show_chart(_layout(fig, 290, title=title, margin=dict(l=10, r=10, t=45, b=10),
                       yaxis=dict(range=[0, 100], title="Churn risk %"), showlegend=False), key)


# ---- Batch charts -----------------------------------------------------------
def render_confusion(cm, key: str, height: int = 260) -> None:
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Predicted stay", "Predicted churn"], y=["Actually stayed", "Actually churned"],
        colorscale=[[0, "rgba(0,122,255,.08)"], [1, BLUE]], showscale=False,
        text=cm, texttemplate="%{text}", textfont=dict(size=18),
        hovertemplate="%{y} / %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show_chart(_layout(fig, height, margin=dict(l=10, r=10, t=10, b=10)), key)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
DEFAULTS = {"gender": "Male", "age": 40, "tenure": 12, "monthly": 65}
PRESETS = {"New customer, high bill": {"gender": "Male", "age": 29, "tenure": 2, "monthly": 105}}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("threshold", 50)
st.session_state.setdefault("budget_pct", 15)
st.session_state.setdefault("history", [])
st.session_state.setdefault("last", None)          # {"inputs": {...}, "risk": float}
st.session_state.setdefault("custom_model", None)  # model trained on the user's own data
st.session_state.setdefault("train_result", None)


def apply_preset(name: str) -> None:
    st.session_state.update(PRESETS[name])


def reset_inputs() -> None:
    st.session_state.update(DEFAULTS)
    st.session_state["last"] = None


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Quick example")
    for _name in PRESETS:
        st.button(_name, key=f"preset_{_name}", on_click=apply_preset, args=(_name,), **stretch(st.button))
    st.button("Reset", key="reset_inputs", on_click=reset_inputs, **stretch(st.button))

    st.markdown("#### Settings")
    st.slider("Churn threshold (%)", 10, 90, key="threshold",
              help="A customer with risk at or above this value is treated as likely to churn.")
    st.slider("Retention budget (% of yearly revenue)", 5, 40, key="budget_pct",
              help="Maximum you would spend to keep a customer who is likely to churn.")

    st.markdown("#### Model")
    custom = st.session_state.get("custom_model")
    if HAVE_MODEL:
        if custom:
            card = [("Source", f"Your data ({custom['source']})"), ("Type", custom["name"]),
                    ("Trained on", f"{custom['rows']:,} customers"), ("ROC-AUC", f"{custom['auc']:.3f}")]
        else:
            card = [("Source", "Saved model files"), ("Type", type(MODEL).__name__)]
        card.append(("Inputs", ", ".join(FEATURE_NAMES)))
        st.markdown(f'<div class="side">{rows_html(card)}</div>', unsafe_allow_html=True)
        if not USES_AGE:
            st.warning("This model has no Age input. Train on data that has an Age column to use it.")
    else:
        st.warning(MODEL_PROBLEM or "No model is loaded.")

    hist = st.session_state["history"]
    if hist:
        avg = np.mean([h["Risk %"] for h in hist])
        st.markdown("#### This session")
        st.caption(f"{len(hist)} saved prediction(s) · average risk {avg:.1f}%")
    st.caption("Light or dark: open the ⋮ menu at the top right, then Settings, then Theme.")

threshold = float(st.session_state["threshold"])
budget_pct = float(st.session_state["budget_pct"])
inputs = {
    "gender": st.session_state["gender"],
    "age": int(st.session_state["age"]),
    "tenure": int(st.session_state["tenure"]),
    "monthly": float(st.session_state["monthly"]),
}


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
def no_model_notice() -> None:
    st.info(MODEL_PROBLEM or "No model is loaded yet. Open the Train tab and upload your data.")


def render_predict() -> None:
    if not HAVE_MODEL:
        no_model_notice()
        return
    left, right = st.columns([1, 1.5], gap="large")

    with left:
        st.markdown("### Customer profile")
        st.selectbox("Gender", ["Male", "Female"], key="gender")
        st.slider("Age (years)", 18, 90, key="age", disabled=not USES_AGE,
                  help="Used only if the model was trained with Age.")
        st.slider("Tenure (months)", 0, 72, key="tenure",
                  help="How long the customer has been with the company")
        st.slider("Monthly charge ($)", 18, 150, key="monthly",
                  help="What the customer pays per month")
        clicked = st.button("Predict churn", type="primary", key="predict_btn", **stretch(st.button))

    if clicked:
        try:
            risk_now = float(score(build_X(encode_gender(inputs["gender"]), inputs["tenure"],
                                           inputs["monthly"], inputs["age"]))[0])
            st.session_state["last"] = {"inputs": dict(inputs), "risk": risk_now}
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    last = st.session_state["last"]

    with right:
        st.markdown("### Result")
        if last is None:
            st.markdown('<div class="empty"><b>No prediction yet</b>'
                        '<div>Set the customer details and press <b>Predict churn</b>.</div></div>',
                        unsafe_allow_html=True)
        else:
            if last["inputs"] != inputs:
                st.caption("The details changed after this prediction. Press Predict churn to update it.")
            risk = last["risk"]
            label = "Yes" if risk >= threshold else "No"
            level, level_color = risk_level(risk)
            title = "Likely to churn" if label == "Yes" else "Likely to stay"
            st.markdown(
                f'<div class="result"><div class="dot" style="background:{level_color}"></div><div>'
                f'<div class="result-title">{title}</div>'
                f'<div class="result-sub">Churn risk {risk:.1f}% · {level} risk · threshold {threshold:.0f}%'
                f'</div></div></div>', unsafe_allow_html=True)
            render_gauge(risk, threshold, "gauge_main")
            if st.button("Save to history", key="save_history", **stretch(st.button)):
                li = last["inputs"]
                st.session_state["history"].append({
                    "Time": datetime.now().strftime("%H:%M:%S"), "Gender": li["gender"],
                    "Age": li["age"] if USES_AGE else "n/a", "Tenure": li["tenure"],
                    "Monthly Charge": li["monthly"], "Prediction": label,
                    "Risk %": round(risk, 1), "Risk Level": level})
                st.toast("Saved to history")

    if last is not None:
        st.markdown("### Business decision")
        decision = build_decision(last["risk"], threshold, last["inputs"], budget_pct)
        if decision["churn"]:
            st.markdown('<div class="section-note">Summary</div>', unsafe_allow_html=True)
            st.markdown(rows_html(decision["summary"]), unsafe_allow_html=True)
            st.markdown('<div class="section-note">Action plan, step by step</div>', unsafe_allow_html=True)
            st.markdown(steps_html(decision["steps"]), unsafe_allow_html=True)
        else:
            st.markdown(rows_html(decision["summary"]), unsafe_allow_html=True)


    if last is not None:
        with st.expander("Which inputs matter most to the model overall?"):
            render_importance("importance_main")
            st.caption("Overall importance across the whole model, not the reason for one prediction.")


def render_whatif() -> None:
    if not HAVE_MODEL:
        no_model_notice()
        return
    st.markdown("### What-if analysis")
    st.caption("Uses the customer details currently set in the Predict tab. The red dot is the current value "
               "and the dotted line is your threshold.")
    c1, c2, c3 = st.columns(3)
    with c1:
        sweep_chart("tenure", np.arange(0, 73, 1), "Risk vs tenure (months)",
                    inputs["tenure"], inputs, threshold, "sweep_tenure")
    with c2:
        sweep_chart("monthly", np.arange(18, 151, 1), "Risk vs monthly charge ($)",
                    inputs["monthly"], inputs, threshold, "sweep_monthly")
    with c3:
        if USES_AGE:
            sweep_chart("age", np.arange(18, 91, 1), "Risk vs age (years)",
                        inputs["age"], inputs, threshold, "sweep_age")
        else:
            st.info("Age chart unavailable: the model was trained without Age.")

    st.markdown("### Discount simulator")
    discount = st.slider("Discount on the monthly charge (%)", 0, 50, 10, key="sim_discount")
    g_enc = encode_gender(inputs["gender"])
    base_risk = float(score(build_X(g_enc, inputs["tenure"], inputs["monthly"], inputs["age"]))[0])
    new_monthly = max(18.0, inputs["monthly"] * (1 - discount / 100.0))
    new_risk = float(score(build_X(g_enc, inputs["tenure"], new_monthly, inputs["age"]))[0])
    loss_before = inputs["monthly"] * 12 * base_risk / 100.0
    loss_after = new_monthly * 12 * new_risk / 100.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current risk", f"{base_risk:.1f}%")
    m2.metric("Risk after discount", f"{new_risk:.1f}%", f"{new_risk - base_risk:+.1f} pts", delta_color="inverse")
    m3.metric("New monthly charge", f"${new_monthly:,.2f}", f"-${inputs['monthly'] - new_monthly:,.2f}",
              delta_color="off")
    m4.metric("Expected yearly loss", f"${loss_after:,.0f}", f"{loss_after - loss_before:+,.0f}",
              delta_color="inverse")
    st.caption("Expected yearly loss = yearly charges × churn risk. The model learned from past patterns, "
               "so treat this as a guide, not proof that a discount will cause this change.")


def render_batch() -> None:
    if not HAVE_MODEL:
        no_model_notice()
        return
    st.markdown("### Batch prediction")
    needed = "Gender, Tenure, MonthlyCharges" + (", Age" if USES_AGE else "")
    st.caption(f"Upload a CSV with these columns: {needed}. Names are matched loosely. Extra columns "
               "(like CustomerID) are kept. Add a Churn column (Yes/No) to also check the model's accuracy.")

    template = pd.DataFrame({"CustomerID": ["A1", "A2", "A3"], "Gender": ["Male", "Female", "Female"],
                             "Age": [29, 48, 67], "Tenure": [2, 60, 8], "MonthlyCharges": [105, 55, 118]})
    if not USES_AGE:
        template = template.drop(columns="Age")
    st.download_button("Download sample template", template.to_csv(index=False).encode("utf-8"),
                       "churn_template.csv", "text/csv", key="dl_template", **stretch(st.download_button))

    upload = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
    if upload is None:
        return
    try:
        out, skipped = run_batch(pd.read_csv(upload), threshold, SCALER, MODEL, USES_AGE)
    except Exception as exc:
        st.error(f"Could not process this file: {exc}")
        return

    churners = out[out["Prediction"] == "Yes"]
    rev_risk = float((churners["MonthlyCharges"] * 12).sum())
    exp_loss = float(out["Expected loss / year"].sum())
    urgent = int(out["Priority"].isin(["Critical", "High"]).sum())
    if len(skipped):
        st.warning(f"{len(skipped)} row(s) were skipped because of missing or invalid values. "
                   "See the list at the bottom of this tab.")

    t_over, t_prio, t_seg, t_all, t_acc = st.tabs(
        ["Overview", "Call list", "Segments", "All results", "Accuracy check"])

    with t_over:
        r1 = st.columns(4)
        r1[0].markdown(kpi("Customers scored", f"{len(out):,}"), unsafe_allow_html=True)
        r1[1].markdown(kpi("Likely to churn", f"{len(churners):,}",
                           f"{len(churners) / len(out) * 100:.1f}% of customers"), unsafe_allow_html=True)
        r1[2].markdown(kpi("Average risk", f"{out['Risk %'].mean():.1f}%"), unsafe_allow_html=True)
        r1[3].markdown(kpi("Urgent (High + Critical)", f"{urgent:,}", "call first"), unsafe_allow_html=True)
        r2 = st.columns(3)
        r2[0].markdown(kpi("Yearly revenue at risk", usd(rev_risk), "customers likely to churn"),
                       unsafe_allow_html=True)
        r2[1].markdown(kpi("Expected yearly loss", usd(exp_loss), "all customers, risk × yearly charges"),
                       unsafe_allow_html=True)
        r2[2].markdown(kpi("Max retention budget", usd(rev_risk * budget_pct / 100),
                           f"{budget_pct:.0f}% of revenue at risk"), unsafe_allow_html=True)

        c1, c2 = st.columns([1.3, 1])
        with c1:
            fig = go.Figure(go.Histogram(x=out["Risk %"], nbinsx=20, marker_color=BLUE))
            fig.add_vline(x=threshold, line_dash="dot", line_color="#8E8E93")
            show_chart(_layout(fig, 300, title="Distribution of churn risk",
                               margin=dict(l=10, r=10, t=45, b=10),
                               xaxis_title="Churn risk %", yaxis_title="Customers"), "batch_hist")
        with c2:
            order = ["Critical", "High", "Moderate", "No action"]
            counts = out["Priority"].value_counts().reindex(order).fillna(0)
            fig = go.Figure(go.Pie(labels=order, values=counts.values, hole=0.62, sort=False,
                                   marker=dict(colors=[RED, ORANGE, BLUE, GREEN]), textinfo="percent",
                                   textposition="outside", automargin=True))
            show_chart(_layout(fig, 300, title="Customers by priority",
                               margin=dict(l=10, r=10, t=45, b=10)), "batch_donut")

    with t_prio:
        st.markdown("### Who to call first")
        if churners.empty:
            st.success("No customer is above your churn threshold. No calls are needed.")
        else:
            ranked = churners.sort_values("Expected loss / year", ascending=False).reset_index(drop=True)
            top_n = st.slider("Number of customers", 1, int(min(200, len(ranked))),
                              int(min(20, len(ranked))), key="prio_n")
            total_loss = float(ranked["Expected loss / year"].sum())
            share = float(ranked.head(top_n)["Expected loss / year"].sum()) / total_loss * 100 if total_loss else 0
            st.caption(f"Ranked by expected yearly loss (risk × yearly charges). The top {top_n} customer(s) "
                       f"cover {share:.0f}% of the expected loss from likely churners.")
            st.dataframe(ranked.head(top_n), hide_index=True,
                         column_config={"Risk %": st.column_config.ProgressColumn(
                             "Risk %", min_value=0, max_value=100, format="%.1f")},
                         **stretch(st.dataframe))
            st.download_button("Download the full call list (CSV)",
                               ranked.to_csv(index=False).encode("utf-8"), "churn_call_list.csv",
                               "text/csv", key="dl_calls", **stretch(st.download_button))

    with t_seg:
        st.markdown("### Where is the risk?")
        options = ["Tenure", "Monthly charge", "Gender"] + (["Age"] if "Age" in out.columns else [])
        by = st.radio("Group customers by", options, horizontal=True, key="seg_by")
        table = segment_table(out, by)
        if table is None or table.empty:
            st.info("This grouping is not available for this file.")
        else:
            fig = go.Figure(go.Bar(
                x=table["Segment"].astype(str), y=table["Average risk %"], marker_color=BLUE,
                text=[f"{n} customers" for n in table["Customers"]], textposition="outside",
                hovertemplate="%{x}<br>Average risk %{y:.1f}%<extra></extra>"))
            show_chart(_layout(fig, 320, margin=dict(l=10, r=10, t=20, b=10),
                               yaxis=dict(range=[0, 115], title="Average churn risk %")), "seg_chart")
            worst = table.sort_values("Average risk %", ascending=False).iloc[0]
            st.caption(f"Highest-risk group: {worst['Segment']} (average risk {worst['Average risk %']:.1f}%, "
                       f"{int(worst['Customers'])} customers).")
            st.dataframe(table, hide_index=True, **stretch(st.dataframe))

    with t_all:
        f1, f2, f3, f4 = st.columns([1.2, 1, 1, 1.4])
        show = f1.radio("Show", ["All", "Likely to churn", "Likely to stay"], key="bf_show")
        min_risk = f2.slider("Minimum risk %", 0, 100, 0, key="bf_min")
        gender_f = f3.selectbox("Gender", ["All", "Female", "Male"], key="bf_gender")
        prios = f4.multiselect("Priority", ["Critical", "High", "Moderate", "No action"],
                               default=["Critical", "High", "Moderate", "No action"], key="bf_prio")
        view = out
        if show == "Likely to churn":
            view = view[view["Prediction"] == "Yes"]
        elif show == "Likely to stay":
            view = view[view["Prediction"] == "No"]
        view = view[view["Risk %"] >= min_risk]
        if gender_f != "All":
            view = view[view["Gender"].astype(str).str.strip().str.lower().str[0] == gender_f[0].lower()]
        view = view[view["Priority"].isin(prios)]
        st.caption(f"Showing {len(view):,} of {len(out):,} customers.")
        st.dataframe(view.sort_values("Risk %", ascending=False), hide_index=True,
                     column_config={"Risk %": st.column_config.ProgressColumn(
                         "Risk %", min_value=0, max_value=100, format="%.1f")},
                     **stretch(st.dataframe))
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download all results (CSV)", out.to_csv(index=False).encode("utf-8"),
                               "churn_batch_results.csv", "text/csv", key="dl_batch",
                               **stretch(st.download_button))
        with d2:
            st.download_button("Download filtered view (CSV)", view.to_csv(index=False).encode("utf-8"),
                               "churn_batch_filtered.csv", "text/csv", key="dl_batch_view",
                               **stretch(st.download_button))

    with t_acc:
        st.markdown("### How accurate is the model on this file?")
        if "Churn" not in out.columns:
            st.info("Add a Churn column (Yes/No or 1/0) to your file to compare the model's predictions "
                    "with what really happened.")
        else:
            actual = parse_target(out["Churn"])
            ok = actual.notna()
            if ok.sum() < 10 or actual[ok].nunique() < 2:
                st.info("Not enough rows with a valid Churn value (needs both Yes and No) to measure accuracy.")
            else:
                y_true = actual[ok].astype(int).to_numpy()
                proba = out.loc[ok, "Risk %"].to_numpy() / 100.0
                y_pred = (out.loc[ok, "Prediction"] == "Yes").astype(int).to_numpy()
                a = st.columns(5)
                a[0].metric("ROC-AUC", f"{roc_auc_score(y_true, proba):.3f}")
                a[1].metric("Accuracy", f"{accuracy_score(y_true, y_pred):.1%}")
                a[2].metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.1%}")
                a[3].metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.1%}")
                a[4].metric("F1", f"{f1_score(y_true, y_pred, zero_division=0):.2f}")
                render_confusion(confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(), "batch_cm")
                st.caption("Recall is the share of real churners the model caught. Precision is the share of "
                           "its churn warnings that were right. If the model was trained on this same file, "
                           "these numbers look better than they will on new customers. To make the model "
                           "learn from this file, use the Train tab.")

    if len(skipped):
        with st.expander(f"Skipped rows ({len(skipped)})"):
            st.dataframe(skipped, hide_index=True, **stretch(st.dataframe))


def render_train() -> None:
    st.markdown("### Train on your own data")
    st.caption("Upload a file with past customers and what happened to them. The app trains three models, "
               "tests them on 20% of your rows it did not train on, and lets you use the best one.")
    st.markdown(rows_html([
        ("Required columns", "Gender, Tenure, MonthlyCharges, Churn (Yes/No or 1/0)"),
        ("Optional", "Age. If it is in the file, the model uses it."),
        ("Size", "At least 50 rows, with at least 10 churned and 10 stayed customers."),
    ]), unsafe_allow_html=True)

    template = pd.DataFrame({"Gender": ["Male", "Female", "Female", "Male"], "Age": [29, 48, 67, 35],
                             "Tenure": [2, 60, 8, 24], "MonthlyCharges": [105, 55, 118, 70],
                             "Churn": ["Yes", "No", "Yes", "No"]})
    st.download_button("Download sample training file", template.to_csv(index=False).encode("utf-8"),
                       "churn_training_template.csv", "text/csv", key="dl_train_template",
                       **stretch(st.download_button))

    upload = st.file_uploader("Upload training CSV", type=["csv"], key="train_upload")
    if upload is not None:
        try:
            raw = pd.read_csv(upload)
            valid, skipped, use_age = prepare_training(raw)
        except Exception as exc:
            st.error(f"Could not use this file: {exc}")
            valid = None
        if valid is not None:
            churn_rate = valid["_y"].mean() * 100 if len(valid) else 0
            k = st.columns(4)
            k[0].markdown(kpi("Rows in file", f"{len(raw):,}"), unsafe_allow_html=True)
            k[1].markdown(kpi("Rows used", f"{len(valid):,}", f"{len(skipped)} skipped"), unsafe_allow_html=True)
            k[2].markdown(kpi("Churn rate", f"{churn_rate:.1f}%"), unsafe_allow_html=True)
            k[3].markdown(kpi("Inputs", "4" if use_age else "3", "with Age" if use_age else "no Age column"),
                          unsafe_allow_html=True)
            if len(skipped):
                with st.expander(f"Skipped rows ({len(skipped)})"):
                    st.dataframe(skipped, hide_index=True, **stretch(st.dataframe))
            if st.button("Train models", type="primary", key="train_btn", **stretch(st.button)):
                try:
                    with st.spinner("Training three models..."):
                        res = train_candidates(valid, use_age)
                    st.session_state["train_result"] = {
                        **res, "valid": valid, "use_age": use_age, "filename": upload.name}
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

    tr = st.session_state.get("train_result")
    if tr:
        st.markdown("### Results")
        results = tr["results"]
        table = pd.DataFrame({name: r["metrics"] for name, r in results.items()}).T
        best = table["ROC-AUC"].idxmax()
        st.caption(f"Measured on {tr['n_test']:,} customers the models did not see during training "
                   f"(trained on {tr['n_train']:,}). Best ROC-AUC: {best}.")
        st.dataframe(table, **stretch(st.dataframe))
        baseline = max(tr["churn_rate"], 100 - tr["churn_rate"])
        st.caption(f"A model that always guessed the most common outcome would already be {baseline:.0f}% accurate, "
                   "so compare models by ROC-AUC, and check Recall and Precision, not Accuracy alone.")

        names = list(results)
        choice = st.selectbox("Model to use", names, index=names.index(best), key="train_choice")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confusion matrix**")
            render_confusion(results[choice]["cm"], "train_cm")
        with c2:
            st.markdown("**What the model relies on**")
            imp = model_importance(results[choice]["model"])
            feat = ["Gender", "Tenure", "Monthly Charge"] + (["Age"] if tr["use_age"] else [])
            if imp is not None and len(imp) == len(feat):
                d = pd.DataFrame({"Feature": feat, "Importance": imp}).sort_values("Importance")
                fig = go.Figure(go.Bar(x=d["Importance"], y=d["Feature"], orientation="h", marker_color=BLUE))
                show_chart(_layout(fig, 260, margin=dict(l=10, r=10, t=10, b=10)), "train_imp")

        if st.button("Use this model in the app", type="primary", key="activate_btn", **stretch(st.button)):
            scaler, model = fit_final(choice, tr["valid"], tr["use_age"])
            st.session_state["custom_model"] = {
                "scaler": scaler, "model": model, "name": choice, "source": tr["filename"],
                "rows": len(tr["valid"]), "auc": results[choice]["metrics"]["ROC-AUC"],
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
            st.session_state["last"] = None
            st.rerun()

    custom = st.session_state.get("custom_model")
    if custom:
        st.success(f"The app is using your {custom['name']} model, trained on {custom['rows']:,} customers "
                   f"from {custom['source']}. Predictions and batch scoring now use it.")
        with st.expander("Keep this model"):
            st.caption("Your model only lives in this browser session. Save it to keep it, or download "
                       "the two files and put them next to app.py.")
            buf_s, buf_m = io.BytesIO(), io.BytesIO()
            joblib.dump(custom["scaler"], buf_s)
            joblib.dump(custom["model"], buf_m)
            d1, d2 = st.columns(2)
            with d1:
                st.download_button("Download scaler.pkl", buf_s.getvalue(), "scaler.pkl", key="dl_scaler",
                                   **stretch(st.download_button))
            with d2:
                st.download_button("Download model.pkl", buf_m.getvalue(), "model.pkl", key="dl_model",
                                   **stretch(st.download_button))
            if st.button("Save as the default model files", key="save_default", **stretch(st.button)):
                try:
                    for fname, obj in (("scaler.pkl", custom["scaler"]), ("model.pkl", custom["model"])):
                        target = BASE_DIR / fname
                        if target.exists():
                            target.replace(BASE_DIR / (fname + ".bak"))
                        joblib.dump(obj, target)
                    load_default_artifacts.clear()
                    st.success("Saved. The old files were kept as scaler.pkl.bak and model.pkl.bak.")
                except Exception as exc:
                    st.error(f"Could not write the files here ({exc}). Use the download buttons instead.")
        if st.button("Switch back to the original model", key="revert_btn", **stretch(st.button)):
            st.session_state["custom_model"] = None
            st.session_state["last"] = None
            st.rerun()


def render_history() -> None:
    st.markdown("### Prediction history (this session)")
    history = st.session_state["history"]
    if not history:
        st.info("Nothing saved yet. Press Save to history after a prediction.")
        return
    hist_df = pd.DataFrame(history)
    h1, h2, h3 = st.columns(3)
    h1.metric("Saved predictions", len(hist_df))
    h2.metric("Likely to churn", int((hist_df["Prediction"] == "Yes").sum()))
    h3.metric("Average risk", f"{hist_df['Risk %'].mean():.1f}%")
    st.dataframe(hist_df, hide_index=True, **stretch(st.dataframe))
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download history as CSV", hist_df.to_csv(index=False).encode("utf-8"),
                           "churn_predictions.csv", "text/csv", key="dl_history", **stretch(st.download_button))
    with d2:
        if st.button("Clear history", key="clear_history", **stretch(st.button)):
            st.session_state["history"] = []
            st.rerun()


def render_about() -> None:
    st.markdown("### About this app")
    st.markdown(
        """
- **Problem:** Predict which customers are at risk of leaving, so the business can target retention offers.
- **Inputs:** Gender, Age, Tenure and Monthly Charge.
- **Data:** IBM Telco Customer Churn dataset — cleaned, encoded, and scaled before training.
- **Class imbalance:** Churners are a minority class; SMOTE was applied to the training set only, to avoid leaking synthetic samples into evaluation.
- **Model:** Random Forest, chosen over Logistic Regression after comparing ROC-AUC (~0.84).
- **Why ROC-AUC:** With imbalanced classes, plain accuracy is misleading — ROC-AUC better reflects how well the model ranks churners above non-churners.

**How the business decision works**

- Risk at or above the threshold (default 50%) means *likely to churn* and gives a step-by-step retention plan.
- Priority grows with risk: Moderate (below 75%), High (75–85%) and Critical (85% and above).
- **Discount rule:** risk of 75% or more gets a 30–40% discount offer. Risk below 75% (but at or above the threshold) gets a 5–10% offer.
- Steps adapt to tenure, monthly charge and age.
- Risk below the threshold means no business action is needed.

**Learning from new data**

- The Train tab trains three model types on your file and tests them on rows they did not see.
- The model you pick is refitted on all your rows and used everywhere in the app.
        """
    )


st.markdown('<div class="title">Customer Churn Prediction</div>'
            '<div class="subtitle">Enter a customer profile, press Predict, and get the risk '
            'together with a clear business decision.</div>', unsafe_allow_html=True)

tab_predict, tab_whatif, tab_batch, tab_train, tab_history, tab_about = st.tabs(
    ["Predict", "What-If", "Batch", "Train", "History", "About"])
with tab_predict:
    render_predict()
with tab_whatif:
    render_whatif()
with tab_batch:
    render_batch()
with tab_train:
    render_train()
with tab_history:
    render_history()
with tab_about:
    render_about()