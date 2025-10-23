Perfect — here’s the **English summary** of what data you need and what outputs you’re expected to produce based on the official **2025 Hackathon Prompt Final** document:

---

## 📥 **1. Data You Will Use (Inputs)**

You’re working with **3 renewable assets** (2 wind farms and 1 solar farm) located in **three U.S. markets**: **ERCOT**, **MISO**, and **CAISO**.
The dataset provided includes both **historical** and **forward-looking** data.

### ✅ **A. Historical Data** — *for calibration and risk modeling*

| Data Type                   | Description                                                            | Time Span | Frequency |
| --------------------------- | ---------------------------------------------------------------------- | --------- | --------- |
| **Hourly Generation Data**  | Hourly energy production (MWh) for each asset                          | 3 years   | Hourly    |
| **Market Prices (RT & DA)** | Hourly real-time (RT) and day-ahead (DA) prices                        | 3 years   | Hourly    |
| **Location Basis**          | Each asset’s **busbar price** and **hub price** (settlement reference) | —         | —         |

These data help model **correlations**, **volatility**, and **basis risk** between busbar and hub prices.

---

### ✅ **B. Forward Data** — *for price forecasting (2026–2030)*

| Data Type                        | Description                                                                 | Frequency |
| -------------------------------- | --------------------------------------------------------------------------- | --------- |
| **Monthly Forward Price Curves** | Forecasted **hub prices**, separated into **peak** and **off-peak** periods | Monthly   |

> 💡 *These forward curves represent the price for a flat 25 MW block of power at the reference hub and are used to project future prices.*

---

## 📤 **2. What You Need to Produce (Outputs)**

You must build a **quantitative, reproducible, and decision-oriented valuation model** that outputs results for each of the three assets over a **5-year term (2026–2030)**.

---

### ✅ **A. Expected Generation Forecast**

* Forecast **monthly energy generation** for each asset.
* Split generation into **peak** and **off-peak** periods.
* Based on historical generation patterns and capacity factors.

---

### ✅ **B. Risk-Adjusted Valuation Prices**

For each asset, compute **four fixed prices** (in $/MWh):

| Price Type                  | Market Level         | Description               |
| --------------------------- | -------------------- | ------------------------- |
| **Real-Time (RT) – Busbar** | Asset node           | Real-time local price     |
| **Real-Time (RT) – Hub**    | Market reference hub | Real-time benchmark price |
| **Day-Ahead (DA) – Busbar** | Asset node           | Day-ahead local price     |
| **Day-Ahead (DA) – Hub**    | Market reference hub | Day-ahead benchmark price |

Each fixed price should include a **component breakdown**:

* **Hub Price**
* **Basis Adjustment** (busbar–hub difference)
* **Risk Adjustment** (volatility, negative prices, etc.)

---

### ✅ **C. Risk Adjustment & Probability Level**

* Assume the company’s **risk appetite = P75**, meaning there’s a **75% probability** that the fixed-price hedge will outperform merchant exposure.
* Be prepared to analyze other risk levels (e.g., P50, P90).
* Explicitly address:

  * **Volume risk** (generation uncertainty)
  * **Price risk** (market volatility)
  * **Negative price events**
  * **Market-specific hedgeability** (which markets favor hedging vs merchant)

---

### ✅ **D. Documentation & Deliverables**

| Deliverable            | Requirements                                                                                                                                  |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model / Code**       | Submit via **GitHub repository** with: `README.md`, all scripts or notebooks, and environment file (`requirements.txt` or `environment.yml`). |
| **Documentation**      | Clearly state all assumptions, methods for price forecasting and risk modeling.                                                               |
| **Presentation (PPT)** | Summarize methodology, key assumptions, valuation results, and actionable insights (e.g., major risks, mitigation strategies).                |

---

## 🧠 **3. Modeling Approach (Suggested Framework)**

| Module                     | Goal                                    | Example Methods                                         |
| -------------------------- | --------------------------------------- | ------------------------------------------------------- |
| **Generation Forecasting** | Predict monthly energy output           | Time series models (SARIMA, Random Forest)              |
| **Price Forecasting**      | Extend forward curve + model volatility | Monte Carlo, GARCH, regression on fundamentals          |
| **Risk Adjustment**        | Compute risk-adjusted fixed price (P75) | Quantile analysis from simulated price distribution     |
| **Hedging Analysis**       | Compare busbar vs hub exposure          | Basis risk modeling, correlation analysis               |
| **Output Visualization**   | Show 5-year fixed-price valuations      | Charts for each market/asset using Matplotlib or Plotly |

---

Would you like me to create a clean **“Data & Deliverables Summary Table”** (in markdown or LaTeX) that you can directly include in your README or final slide deck? It’ll clearly show all required inputs and expected outputs side-by-side.
