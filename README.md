## 🌍 CO₂ Emissions Forecasting Project

This project provides an interactive dashboard for analyzing and forecasting **CO₂ emissions by country** using machine learning models trained on the **World Bank — World Development Indicators (WDI)** dataset. 🔗 https://databank.worldbank.org/source/world-development-indicators

---

## 🏠 Home Overview

The homepage of the app presents:

### 🔹 **1. Interactive Input Panel**
Users can input:
- Country
- Year
- GDP (current US$)
- Population
- Energy use (kg oil eq. per capita)
- Renewable energy (%)
- Life expectancy  
and choose between:
- **Global model**
- **Tier-specific model**

There are also sample buttons for quickly loading pre-filled example countries such as **USA**, **Singapore**, and **Albania**.

### 🔹 **2. Forecast Results**
The system outputs:
- Predicted CO₂ emissions (mean)
- p05–p95 range (non-parametric 90% interval)
- Tier classification (High / Medium / Low)

### 🔹 **3. Visualizations**
The dashboard automatically generates:
- CO₂ Actual vs Forecast Line Chart  
- Feature Importance Chart (key drivers)

---

## 🖼️ Home Preview

Below is the latest version of the dashboard homepage:

![home_preview](./dataset/web_demo.png)

