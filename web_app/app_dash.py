from pathlib import Path
from typing import Any, Dict, List
import base64
import json

import dash
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from model_deploy import (
    FEATURE_COLS,
    TARGET_COL,
    TIER_CUTOFF_YEAR,
    preprocess,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "exports" / "xgb_model.json"
TRAIN_PATH = ROOT / "cleaned_EDA_ready_timeseries.csv"
TIER_MODELS_PATH = ROOT / "exports" / "tier_models.json" 
FI_PATH = ROOT / "exports" / "feature_importance.json"


def load_xgb(path: Path) -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(path)
    return model


def load_tier_models(path: Path):
    """
    Load tier-specific models directly from exports/tier_model_*.json.
    Ignore any machine-specific paths inside tier_models.json.
    """
    models = {}
    for tier in ["High", "Medium", "Low"]:
        candidate = ROOT / "exports" / f"tier_model_{tier}.json"
        if candidate.exists():
            m = XGBRegressor()
            m.load_model(candidate)
            models[tier] = m
    print("Loaded tier models:", list(models.keys()))
    return models



model_global = load_xgb(MODEL_PATH)
tier_models = load_tier_models(TIER_MODELS_PATH)  # may be empty

feature_importances = {}
if FI_PATH.exists():
    feature_importances = json.loads(FI_PATH.read_text())

df_train_raw = pd.read_csv(TRAIN_PATH)
df_train = preprocess(df_train_raw)
X_train = df_train[FEATURE_COLS]
y_train = df_train[TARGET_COL]

LOW_THR = 50.0
HIGH_THR = 500.0


def build_tier_map_quantile(df, end_year, window=5, low_q=0.33, high_q=0.67):
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


tier_map_old = build_tier_map_quantile(df_train_raw, end_year=TIER_CUTOFF_YEAR, window=5)
tier_map_current = build_tier_map_quantile(df_train_raw, end_year=df_train_raw["Year"].max(), window=5)
df_train["Tier_Current"] = df_train["Country Code"].map(tier_map_current)
df_train["Tier_Current"] = df_train["Country Code"].map(tier_map_current)

country_choices = (
    df_train_raw[["Country Code", "Country Name"]]
    .drop_duplicates()
    .sort_values("Country Name")["Country Name"]
    .tolist()
)
INPUT_STYLE = {
    "backgroundColor": "#FDFBFB",
    "width": "100%",
    "height": "24px",
    "border": "1px solid #e5e7eb",
    "borderRadius": "4px",
    "padding": "6px 10px",
    "lineHeight": "18px",}
INPUT_STYLE_DROPDOWN = {
    "width": "100%",
    "backgroundColor":"transparent",
    "border": "1px solid #e5e7eb",
    "boxShadow": "none",}
sample_btn_style = {
    "padding": "8px 18px",
    "background": "#f3f4f6",         
    "color": "#374151",
    "border": "1px solid #d1d5db",  
    "borderRadius": "10px",
    "cursor": "pointer",
    "fontSize": "15px",
    "fontWeight": "500",
    "transition": "all 0.2s ease"}

def bias_corrected_predict_capped(model, X_train, X_new, y_train, resid_std_cap: float = 0.7):
    """
    Predict with log-space bias correction and capped residual std to avoid unrealistically wide intervals.
    """
    train_pred_log = model.predict(X_train)
    resid_var = float(np.var(y_train - train_pred_log, ddof=1))
    resid_std = min(np.sqrt(resid_var), resid_std_cap)
    bias_term = 0.5 * (resid_std**2)

    pred_log = model.predict(X_new)
    pred_log = np.maximum(pred_log, 0.0)

    log_lower = np.maximum(pred_log - 1.645 * resid_std, 0.0)
    log_upper = np.maximum(pred_log + 1.645 * resid_std, 0.0)

    out = pd.DataFrame(
        {
            "pred_log1p_CO2_total_mt": pred_log,
            "pred_CO2_total_mt_mean": np.expm1(pred_log + bias_term).clip(min=0),
            "pred_CO2_total_mt_p05": np.expm1(log_lower).clip(min=0),
            "pred_CO2_total_mt_p95": np.expm1(log_upper).clip(min=0),
        }
    )
    return out

# load feature importances if available
def _build_hero_data_uri() -> str:
    img_path = ROOT / "co2.jpg"
    if img_path.exists():
        data = img_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='1400' height='120'>"
        "<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='0'>"
        "<stop offset='0%' stop-color='%23111827'/>"
        "<stop offset='100%' stop-color='%23256be0'/>"
        "</linearGradient></defs>"
        "<rect width='1400' height='120' rx='12' fill='url(#g)'/>"
        "<text x='32' y='75' fill='white' font-size='34' font-family='Inter, Arial, sans-serif' font-weight='700'>CO2 Forecast Dashboard</text>"
        "</svg>"
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


hero_svg = _build_hero_data_uri()


def build_feature_row(form: Dict[str, Any]) -> pd.DataFrame:
    cc = form.get("country_code", "")
    cn = form.get("country_name", "")
    year = int(form.get("year", 2024))

    def safe_pos(val: Any) -> float:
        try:
            v = float(val)
            return np.nan if v <= 0 else v
        except Exception:
            return np.nan

    gdp = safe_pos(form.get("gdp"))
    pop = safe_pos(form.get("population"))
    energy = safe_pos(form.get("energy_use"))
    renew = float(form.get("renewable") or 0)
    life = float(form.get("life_exp") or 0)

    data = {
        "Country Code": cc,
        "Country Name": cn,
        "Year": year,
        "GDP_current_usd": gdp,
        "Population_total": pop,
        "Energy_use_kg_oil_pc": energy,
        "Renewable_energy_pct": renew,
        "LifeExp_years": life,
    }
    df = pd.DataFrame([data])

    df["log1p_GDP_current_usd"] = np.log1p(df["GDP_current_usd"])
    df["log1p_Population_total"] = np.log1p(df["Population_total"])
    df["log1p_Energy_use_kg_oil_pc"] = np.log1p(df["Energy_use_kg_oil_pc"])
    df["Emission_Tier"] = df["Country Code"].map(tier_map_old).fillna("Unlabeled")
    return df


def make_imp_fig(title, importances, colors=None):
    colors = colors or ["#38393A"] * len(importances)
    fig = go.Figure(
        go.Bar(
            x=FEATURE_COLS,
            y=importances,
            marker=dict(color=colors),
            text=[f"{v:.3f}" for v in importances],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=title,
        height=260,
        margin=dict(l=20, r=20, t=40, b=30),
        xaxis_title="",
        yaxis_title="Importance",
    )
    return fig


app: Dash = dash.Dash(__name__,external_stylesheets=["https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"],)
app.title = "CO₂ Forecast Dashboard"
app.layout = html.Div(
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "12px 18px 32px 18px",
        "fontFamily": "Inter, Arial, sans-serif",
    },
    children=[
        html.Div(html.Img(src=hero_svg, style={"width": "100%", "borderRadius": "12px"}), style={"marginBottom": "12px"}),
        html.Div(
            style={"marginBottom": "10px"},
            children=[
                html.Button([
                html.I(className="fa-solid fa-circle-info",style={"marginRight": "6px"},),
                "How to use",],
                    id="help_btn",
                    n_clicks=0,
                    style={
                        "padding": "6px 14px",
                        "border": "1px solid #9ca3af",
                        "borderRadius": "8px",
                        "color": "#F8F6F6",
                        "cursor": "pointer",
                        "fontWeight": 600,
                        "background": "#090909",
                    },
                ),
                html.Div(
                    id="help_box",
                    style={
                        "display": "none",
                        "marginTop": "6px",
                        "padding": "10px",
                        "border": "1px solid #e5e7eb",
                        "borderRadius": "8px",
                        "background": "#ffffff",
                        "fontSize": "13px",
                    },
                    children=html.Ul(
                        style={"marginTop": "6px", "paddingLeft": "18px"},
                        children=[
                            html.Li("Forecasts CO₂ emissions (Mt) using an XGBoost regression model (Global or Tier-specific)"),
                            html.Li("Step 1: Select a country and fill data (or click a Sample button to auto-fill)."),
                            html.Li("Step 2: Choose Global or Tier-specific mode."),
                            html.Li("Step 3: Click predict to see the mean forecast and p05–p95 range (Mt), plus charts."),
                        ],
                    ),
                ),
            ],
        ),
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "gap": "12px",
                "marginTop": "4px",
                "flexWrap": "wrap",
            },
            children=[
                html.Div(
                    style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                    children=[
                        html.Button("Sample (High: USA)", id="sample_usa", n_clicks=0, style=sample_btn_style),
                        html.Button("Sample (Medium: Singapore)", id="sample_singapore", n_clicks=0, style=sample_btn_style),
                        html.Button("Sample (Low: Albania)", id="sample_albania", n_clicks=0, style=sample_btn_style),
                    ],
                ),
                html.Div(
                    style={
                        "display": "flex",
                        "gap": "12px",
                        "alignItems": "center",
                        "background": "#080808",
                        "border": "1px solid #e5e7eb",
                        "borderRadius": "8px",
                        "padding": "8px 12px",
                    },
                    children=[
                        html.Div("Model mode:", style={"fontSize": "16px","fontWeight": "600", "color": "#fcfdfe"}),
                        dcc.RadioItems(
                            id="model_mode",
                            options=[
                                {"label": "Global", "value": "global"},
                                {"label": "Tier-specific", "value": "tier"},
                            ],
                            value="global",
                            labelStyle={"marginRight": "12px"},
                            style={"display": "flex", "gap": "12px","color": "#fcfdfe"},
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "64px", "marginTop": "12px"},
            children=[
                html.Div(
                    style={"display": "flex", "flexDirection": "column", "gap": "12px"},
                    children=[
                        html.Div("Country", style={"fontSize": "16px","fontWeight": "600","color": "#4b5563", "marginBottom": "4px","minHeight": "24px"}),
                        dcc.Dropdown(
                            id="country_name",
                            options=[{"label": c, "value": c} for c in country_choices],
                            placeholder="Select country",
                            style=INPUT_STYLE_DROPDOWN,
                        ),
                        html.Div("Year", style={"fontSize": "16px","fontWeight": "600", "color": "#4b5563", "marginBottom": "4px","minHeight": "24px"}),
                        dcc.Input(id="year", type="number", placeholder="Year (e.g., 2024)", value=2024, debounce=True, style=INPUT_STYLE),
                        html.Div("Population (total)", style={"fontSize": "16px","fontWeight": "600", "color": "#4b5563", "marginBottom": "4px","minHeight": "24px"}),
                        dcc.Input(id="population", type="number", placeholder="e.g., 3.3e8", debounce=True, style=INPUT_STYLE),
                        html.Div("Renewable energy (%)", style={"fontSize": "16px","fontWeight": "600","color": "#4b5563", "marginBottom": "4px","minHeight": "24px"}),
                        dcc.Input(id="renewable", type="number", placeholder="e.g., 15", debounce=True, style=INPUT_STYLE),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "flexDirection": "column", "gap": "12px"},
                    children=[
                        html.Div("Country code", style={"fontSize": "16px","fontWeight": "600","color": "#4b5563", "marginBottom": "4px","minHeight": "24px"}),
                        dcc.Input(id="country_code", type="text", placeholder="e.g., USA", debounce=True, style=INPUT_STYLE),
                        html.Div("GDP (current US$)", style={"fontSize": "16px","fontWeight": "600","color": "#4b5563", "marginBottom": "4px","minHeight": "24px"}),
                        dcc.Input(id="gdp", type="number", placeholder="e.g., 2.5e13", debounce=True, style=INPUT_STYLE),
                        html.Div("Energy use (kg oil eq. per capita)", style={"fontSize": "16px","fontWeight": "600", "color": "#4b5563", "marginBottom": "4px","minHeight": "24px"}),
                        dcc.Input(id="energy_use", type="number", placeholder="e.g., 6000", debounce=True, style=INPUT_STYLE),
                        html.Div("Life expectancy (years)", style={"fontSize": "16px","fontWeight": "600", "color": "#4b5563", "marginBottom": "4px","minHeight": "24px"}),
                        dcc.Input(id="life_exp", type="number", placeholder="e.g., 78", debounce=True, style=INPUT_STYLE),
                    ],
                ),
            ],
        ),
        html.Button(
            "Predict",
            id="predict_btn",
            n_clicks=0,
            style={
                "marginTop": "16px",
                "padding": "10px 16px",
                "background": "#25282b",
                "color": "white",
                "border": "none",
                "borderRadius": "6px",
                "cursor": "pointer",
                "fontWeight": "600",
                "width": "120px",
            },
        ),
                html.Div(
            id="result",
            style={
                "marginTop": "18px",
                "fontSize": "16px",
                "padding": "16px",
                "border": "1px solid #e5e7eb",
                "borderRadius": "12px",
                "background": "linear-gradient(180deg, #f8fafc, #eef2ff)",
                "lineHeight": "1.6",
            },
        ),
        html.Div(
            style={
                "marginTop": "12px",
                "fontSize": "14px",
                "color": "#4b5563",
                "padding": "10px 0",
            },
            children=[
                html.Strong("Notes:"),
                html.Ul(
                    style={
                        "marginTop": "6px",
                        "paddingLeft": "20px",
                        "lineHeight": "1.6",
                    },
                    children=[
                        html.Li(
                            "p05–p95 is a non-parametric 90% interval showing typical low–high values in the data."
                        ),
                        html.Li(
                            "Developed by Aphisit st126130 & Thiri st126018"
                        ),
                        html.Li(
                            [
                                "Background image sourced from ",
                                html.A(
                                    "IQAir Newsroom",
                                    href="https://www.iqair.com/th/newsroom/carbon-dioxide",
                                    target="_blank",
                                    style={"color": "#2563eb"},
                                ),
                                ".",
                            ]
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("country_name", "value"),
    Output("country_code", "value"),
    Output("year", "value"),
    Output("population", "value"),
    Output("gdp", "value"),
    Output("renewable", "value"),
    Output("energy_use", "value"),
    Output("life_exp", "value"),
    Input("sample_usa", "n_clicks"),
    Input("sample_singapore", "n_clicks"),
    Input("sample_albania", "n_clicks"),
    prevent_initial_call=True,
)
def fill_samples(n_usa, n_india, n_ger):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "sample_usa":
        return "United States", "USA", 2025, 3.35e8, 2.7e13, 18, 6000, 78
    if btn == "sample_singapore":
        return "Singapore", "SGP", 2025, 5.9e6, 4.0e11, 4, 5400, 84
    if btn == "sample_albania":
        return "Albania", "ALB", 2025, 2.8e6, 1.8e10, 30, 1200, 78
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("result", "children"),
    inputs=[Input("predict_btn", "n_clicks")],
    state=[
        State("country_name", "value"),
        State("country_code", "value"),
        State("year", "value"),
        State("gdp", "value"),
        State("population", "value"),
        State("energy_use", "value"),
        State("renewable", "value"),
        State("life_exp", "value"),
        State("model_mode", "value"),
    ],
)
def on_predict(n_clicks, cn, cc, year, gdp, pop, energy, renew, life, mode):
    if not n_clicks:
        return ""

    form = {
        "country_name": cn or "",
        "country_code": cc or "",
        "year": year or 2024,
        "gdp": gdp,
        "population": pop,
        "energy_use": energy,
        "renewable": renew,
        "life_exp": life,
    }
    df_feat = build_feature_row(form)
    if df_feat[FEATURE_COLS].isna().any().any():
        return html.Div("Please fill information", style={"color": "red"})

    tier_current = tier_map_current.get(cc, "Unlabeled")

    model_used = model_global
    X_ref = X_train
    y_ref = y_train
    if mode == "tier" and tier_current in tier_models:
        model_used = tier_models[tier_current]
        tier_slice = df_train[df_train["Tier_Current"] == tier_current]
        if not tier_slice.empty:
            X_ref = tier_slice[FEATURE_COLS]
            y_ref = tier_slice[TARGET_COL]

    preds = bias_corrected_predict_capped(
        model=model_used,
        X_train=X_ref,
        X_new=df_feat[FEATURE_COLS],
        y_train=y_ref,
    )
    row = preds.iloc[0]
    mean = float(row["pred_CO2_total_mt_mean"])
    p05 = float(row["pred_CO2_total_mt_p05"])
    p95 = float(row["pred_CO2_total_mt_p95"])

    hist_df = df_train[df_train["Country Code"] == cc] if cc else pd.DataFrame()
    fig_hist = go.Figure()
    if not hist_df.empty:
        fig_hist.add_trace(
            go.Scatter(
                x=hist_df["Year"],
                y=hist_df["CO2_total_mt"],
                mode="lines+markers",
                name="Actual",
                line=dict(color="#111827"),
                marker=dict(size=6),
            )
        )
        last_year = hist_df["Year"].iloc[-1]
        last_val = hist_df["CO2_total_mt"].iloc[-1]
        fig_hist.add_trace(
            go.Scatter(
                x=[last_year, year],
                y=[last_val, mean],
                mode="lines",
                name="Bridge to forecast",
                line=dict(color="#2563eb", dash="dash", width=2, shape="linear"),
                opacity=0.7,
                showlegend=False,
            )
        )
    fig_hist.add_trace(
        go.Scatter(
            x=[year],
            y=[mean],
            mode="markers",
            name="Forecast mean",
            marker=dict(color="#2563eb", size=10),
        )
    )
    fig_hist.update_layout(
        title=f"CO₂ actual vs forecast | Tier: {tier_current}",
        yaxis_title="CO₂ (Mt)",
        xaxis_title="Year",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
    )

    imp_figs: List[go.Figure] = []
    if mode == "tier" and tier_current in tier_models:
        if feature_importances and "tiers" in feature_importances and tier_current in feature_importances["tiers"]:
            imp_figs.append(make_imp_fig(f"Key drivers ({tier_current})", feature_importances["tiers"][tier_current]))
        else:
            imp_figs.append(make_imp_fig(f"Key drivers ({tier_current})", tier_models[tier_current].feature_importances_))
    else:
        if feature_importances and "global" in feature_importances:
            imp_figs.append(make_imp_fig("Key drivers (global)", feature_importances["global"]))
        else:
            imp_figs.append(make_imp_fig("Key drivers (global)", model_global.feature_importances_))


    def fmt(x: float) -> str:
        return f"{x:,.2f}"

    summary_panel = html.Div(
        style={
            "marginTop": "12px",
            "marginBottom": "8px",
            "padding": "14px",
            "border": "1px solid #e5e7eb",
            "borderRadius": "12px",
            "background": "#ffffff",
        },
        children=[
            html.Div("CO₂ forecast (Mt)", style={"fontWeight": 700, "marginBottom": "6px", "fontSize": "16px"}),
            html.Div(f"Mode: {'Tier' if mode=='tier' else 'Global'} | Tier: {tier_current} | Year: {year}", style={"color": "#6b7280", "fontSize": "12px", "marginBottom": "4px"}),
            html.Div(f"Range (p05–p95): {fmt(p05)} – {fmt(p95)} Mt", style={"color": "#374151", "marginBottom": "10px", "fontSize": "13px"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))", "gap": "10px"},
                children=[
                    html.Div(
                        style={"border": "1px solid #e5e7eb", "borderRadius": "8px", "padding": "10px"},
                        children=[
                            html.Div("Mean (Mt)", style={"color": "#6b7280", "fontSize": "12px"}),
                            html.Div(f"{fmt(mean)}", style={"fontSize": "18px", "fontWeight": 700}),
                        ],
                    ),
                    html.Div(
                        style={"border": "1px solid #e5e7eb", "borderRadius": "8px", "padding": "10px"},
                        children=[
                            html.Div("p05 (Mt)", style={"color": "#6b7280", "fontSize": "12px"}),
                            html.Div(f"{fmt(p05)}", style={"fontSize": "18px", "fontWeight": 700}),
                        ],
                    ),
                    html.Div(
                        style={"border": "1px solid #e5e7eb", "borderRadius": "8px", "padding": "10px"},
                        children=[
                            html.Div("p95 (Mt)", style={"color": "#6b7280", "fontSize": "12px"}),
                            html.Div(f"{fmt(p95)}", style={"fontSize": "18px", "fontWeight": 700}),
                        ],
                    ),
                ],
            ),
        ],
    )

    return html.Div(
        children=[
            summary_panel,
            dcc.Graph(figure=fig_hist, config={"displayModeBar": False}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))", "gap": "10px"},
                children=[dcc.Graph(figure=fig, config={"displayModeBar": False}) for fig in imp_figs],
            ),
        ]
    )


@app.callback(
    Output("help_box", "style"),
    Input("help_btn", "n_clicks"),
    prevent_initial_call=False,
)
def toggle_help(n):
    base_style = {
        "marginTop": "6px",
        "padding": "10px",
        "border": "1px solid #e5e7eb",
        "borderRadius": "8px",
        "background": "#ffffff",
        "fontSize": "13px",
    }
    visible = bool(n and n % 2 == 1)
    base_style["display"] = "block" if visible else "none"
    return base_style


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
