# 🌍 CO₂ Emissions Forecast Dashboard

Interactive web application to forecast country-level CO₂ emissions (in Mt) using global and tier-specific machine-learning models.
Built with **Dash**, **XGBoost**, and real world indicators from the **World Bank**.

## 🔎 Overview

This project provides a clean and interactive dashboard for exploring and forecasting CO₂ emissions based on user-provided socioeconomic indicators.

It supports:
- **Global model**
- **Tier-specific models** (High / Medium / Low)
- **Forecast ranges** (p05–p95 interval)
- **Feature importance**
- **Benchmark comparison** with world & tier averages

## 🚀 Quick Start

```bash
git clone https://github.com/APS2542/CP_class_project.git
cd CP_class_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app_dash.py
```

Then open **http://127.0.0.1:8050**

## 📁 Repository Structure

CP_class_project/
- app_dash.py
- model_deploy.py
- cleaned_EDA_ready_timeseries.csv
- exports/ (model files & feature importance)
- requirements.txt
- co2.jpg
- README.md

## 🖥️ Dashboard Features

### Inputs
- Country, code, year
- GDP, population
- Energy use
- Renewable energy
- Life expectancy

### Outputs
- Mean CO₂ forecast
- p05–p95 interval
- Feature importance
- Tier classification

### Visualizations
- Historical CO₂ vs forecast
- Benchmarks (world & tier)
- Forecast range plot
- Feature importance

## 🔬 Modeling & Methodology

- Log-transform on GDP/pop/energy use
- Quantile-based tier classification
- XGBoost regression
- Log-space bias correction
- Residual variance cap for interval stability

## ⚠️ Limitations

- Does not account for future policy/technology shocks
- Missing data may reduce accuracy
- Tier classification may shift over time

## 🛠️ Future Improvements

- Multi-year forecasting
- Cloud deployment
- Export results
- Add more socioeconomic indicators
- Model performance comparison

## 👥 Authors

Aphisit (st126130)  
Thiri (st126018)

## 📚 Citation

CO₂ Emissions Forecast Dashboard — CP Class Project 2025 (Aphisit & Thiri)

