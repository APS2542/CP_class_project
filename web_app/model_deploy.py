
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


FEATURE_COLS = ["log1p_GDP_current_usd","log1p_Population_total","log1p_Energy_use_kg_oil_pc","Renewable_energy_pct","LifeExp_years",]
TARGET_COL = "log1p_CO2_total_mt"
TIER_CUTOFF_YEAR = 2014


def build_tier_map(df: pd.DataFrame, end_year: int, window: int = 5) -> Dict[str, str]:
    """Map Country Code -> tier using mean CO2 over the last `window` years up to end_year."""
    hist = df[df["Year"].between(end_year - window + 1, end_year)].dropna(subset=["CO2_total_mt"])
    if hist.empty:
        return {}
    mean_recent = hist.groupby("Country Code")["CO2_total_mt"].mean()
    q = mean_recent.quantile([0.33, 0.67]).values

    def assign(x: float) -> str:
        if x <= q[0]:
            return "Low"
        elif x <= q[1]:
            return "Medium"
        else:
            return "High"

    return mean_recent.apply(assign).to_dict()


def apply_tier(df: pd.DataFrame, end_year: int) -> pd.DataFrame:
    tier_map = build_tier_map(df, end_year=end_year, window=5)
    df_out = df.copy()
    df_out["Emission_Tier"] = df_out["Country Code"].map(tier_map).fillna("Unlabeled")
    return df_out


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as the notebook."""
    df = df.copy()
    df["log1p_GDP_current_usd"] = np.log1p(df["GDP_current_usd"])
    df["log1p_Population_total"] = np.log1p(df["Population_total"])
    df["log1p_Energy_use_kg_oil_pc"] = np.log1p(df["Energy_use_kg_oil_pc"])
    df["log1p_CO2_total_mt"] = np.log1p(df["CO2_total_mt"])
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df = apply_tier(df, end_year=TIER_CUTOFF_YEAR)
    return df


def train_xgb(train_df: pd.DataFrame,params: Dict = None,) -> Tuple[XGBRegressor, float]:
    """Train XGB and return model and residual variance for bias correction."""
    params = params or {"n_estimators": 400,"learning_rate": 0.05,"max_depth": 5,"subsample": 0.8,"colsample_bytree": 0.8,
                        "reg_alpha": 0.1,"reg_lambda": 1.0,"random_state": 42}
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    resid_var = float(np.var(y_train - pred_train, ddof=1))
    return model, resid_var


def bias_corrected_predict(model: XGBRegressor,X_train: pd.DataFrame,X_new: pd.DataFrame,y_train: pd.Series) -> pd.DataFrame:
    """Predict with log-space bias correction and simple normal-based intervals, enforcing non-negative log."""
    train_pred_log = model.predict(X_train)
    resid_var = float(np.var(y_train - train_pred_log, ddof=1))
    resid_std = np.sqrt(resid_var)
    bias_term = 0.5 * resid_var

    pred_log = model.predict(X_new)
    pred_log = np.maximum(pred_log, 0.0)

    log_lower = np.maximum(pred_log - 1.645 * resid_std, 0.0)
    log_upper = np.maximum(pred_log + 1.645 * resid_std, 0.0)

    out = pd.DataFrame({
        "pred_log1p_CO2_total_mt": pred_log,
        "pred_CO2_total_mt_mean": np.expm1(pred_log + bias_term).clip(min=0),
        "pred_CO2_total_mt_p05": np.expm1(log_lower).clip(min=0),
        "pred_CO2_total_mt_p95": np.expm1(log_upper).clip(min=0),
    })
    return out


def save_model(model: XGBRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def save_meta(meta: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))


def main():
    script_dir = Path(__file__).resolve().parent
    default_train = (script_dir.parent / "cleaned_EDA_ready_timeseries.csv")
    default_model = (script_dir.parent / "exports" / "xgb_model.json")
    default_meta = (script_dir.parent / "exports" / "model_meta.json")

    parser = argparse.ArgumentParser(description="Train and export XGB model for deployment.")
    parser.add_argument("--train-path", type=Path, default=default_train)
    parser.add_argument("--model-out", type=Path, default=default_model)
    parser.add_argument("--meta-out", type=Path, default=default_meta)
    args = parser.parse_args()

    train_path = args.train_path
    if not train_path.is_absolute():
        # resolve relative to script directory if not absolute
        train_path = (script_dir / train_path).resolve()

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    df_raw = pd.read_csv(train_path)
    df = preprocess(df_raw)
    model, resid_var = train_xgb(df)
    save_model(model, args.model_out)

    meta = {"feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "tier_cutoff_year": TIER_CUTOFF_YEAR,
        "resid_var": resid_var,
        "train_rows": len(df),
        "train_year_min": int(df["Year"].min()),
        "train_year_max": int(df["Year"].max())}
    save_meta(meta, args.meta_out)
    print(f"Model saved to {args.model_out}")
    print(f"Meta saved to {args.meta_out}")


if __name__ == "__main__":
    main()
