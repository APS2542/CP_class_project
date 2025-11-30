"""
Train tier-specific XGB models (quantile-based tiers) and save to exports/.
Tier mapping uses 5-year mean CO2 quantiles:
  Low <= 33rd percentile, Medium <= 67th percentile, High > 67th percentile

Usage:
    python train_tier_models.py
Outputs:
    exports/tier_model_High.json (if data exists)
    exports/tier_model_Medium.json
    exports/tier_model_Low.json
    exports/tier_models.json  (manifest tier -> path)
    exports/feature_importance.json (global + tier importances)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from model_deploy import FEATURE_COLS, TARGET_COL, preprocess

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT.parent / "cleaned_EDA_ready_timeseries.csv"
EXPORT_DIR = ROOT.parent / "exports"
EXPORT_DIR.mkdir(exist_ok=True)
GLOBAL_MODEL_PATH = ROOT.parent / "exports" / "xgb_model.json"
FI_PATH = EXPORT_DIR / "feature_importance.json"

LOW_Q = 0.33
HIGH_Q = 0.67
WINDOW = 5


def build_tier_map_quantile(df, end_year, window=WINDOW, low_q=LOW_Q, high_q=HIGH_Q):
    hist = df[df["Year"].between(end_year - window + 1, end_year)].dropna(subset=["CO2_total_mt"])
    if hist.empty:
        return {}
    mean_recent = hist.groupby("Country Code")["CO2_total_mt"].mean()
    q_low = mean_recent.quantile(low_q)
    q_high = mean_recent.quantile(high_q)

    def assign(x):
        if x <= q_low:
            return "Low"
        elif x <= q_high:
            return "Medium"
        else:
            return "High"

    return mean_recent.apply(assign).to_dict()


def train_tier_models(df: pd.DataFrame):
    models = {}
    params = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )
    for tier_name, df_t in df.groupby("Emission_Tier"):
        if len(df_t) < 30:
            continue
        model_t = XGBRegressor(**params)
        model_t.fit(df_t[FEATURE_COLS], df_t[TARGET_COL])
        models[tier_name] = model_t
    return models


def main():
    df_raw = pd.read_csv(TRAIN_PATH)
    df = preprocess(df_raw)
    tier_map = build_tier_map_quantile(df_raw, end_year=df_raw["Year"].max(), window=WINDOW)
    df["Emission_Tier"] = df["Country Code"].map(tier_map).fillna("Unlabeled")

    models = train_tier_models(df)
    manifest = {}
    for tier, m in models.items():
        out_path = EXPORT_DIR / f"tier_model_{tier}.json"
        m.save_model(out_path)
        manifest[tier] = str(out_path)
        print(f"Saved {tier} model -> {out_path}")

    manifest_path = EXPORT_DIR / "tier_models.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest saved to {manifest_path}")

    # save feature importances (global + tiers)
    fi = {"FEATURE_COLS": FEATURE_COLS, "tiers": {}}
    # global
    if GLOBAL_MODEL_PATH.exists():
        g_model = XGBRegressor()
        g_model.load_model(GLOBAL_MODEL_PATH)
        fi["global"] = g_model.feature_importances_.tolist()
    # tiers
    for tier, m in models.items():
        fi["tiers"][tier] = m.feature_importances_.tolist()
    FI_PATH.write_text(json.dumps(fi, indent=2))
    print(f"Feature importances saved to {FI_PATH}")


if __name__ == "__main__":
    main()
