# Renewable Energy Asset Valuation Model

## 📌 Project Description

A quantitative framework for valuing merchant renewable energy assets and determining risk-adjusted fixed prices for Power Purchase Agreements (PPAs). This model helps renewable energy developers evaluate the trade-off between merchant market exposure and fixed-price offtake structures for wind and solar assets across major US electricity markets.

## 🎯 Problem Statement

When renewable energy assets reach the end of their PPAs, they face merchant market exposure with volatile wholesale prices, congestion, and curtailment risks. This model provides a transparent, data-driven approach to:
- Value future energy output under uncertainty
- Determine fair fixed prices for 5-year recontracting
- Assess market-specific hedging strategies
- Quantify and price various risk factors

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install pandas numpy scipy openpyxl
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Hu-xiao-max/Hackahton25
cd renewable-energy-valuation

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from renewable_valuation import RenewableAssetValuation

# Initialize model with your data
model = RenewableAssetValuation('HackathonDataset.xlsx')

# Run complete analysis
results, market_scores = model.run_full_analysis()

# Results include risk-adjusted prices for each market
print(results['ERCOT']['fixed_prices'])
```

## 📊 Input Data Requirements

### Required Excel Format
The model expects an Excel file with three sheets (ERCOT, MISO, CAISO) containing:

**Historical Data Columns:**
- `Date`: Timestamp
- `HE`: Hour Ending (1-24)
- `P/OP`: Peak/Off-Peak indicator
- `Gen`: Generation (MW)
- `RT_Busbar`: Real-time busbar price ($/MWh)
- `RT_Hub`: Real-time hub price ($/MWh)
- `DA_Busbar`: Day-ahead busbar price ($/MWh)
- `DA_Hub`: Day-ahead hub price ($/MWh)

**Forward Curve Columns:**
- `Forward_Date`: Future month/year
- `Peak`: Peak period forward price ($/MWh)
- `Off_Peak`: Off-peak forward price ($/MWh)

## 🔧 Core Functionality

### 1. Historical Data Analysis
- **Generation Profiling**: Analyzes capacity factors and seasonal patterns
- **Price Analysis**: Examines historical price distributions and volatility
- **Basis Risk**: Quantifies hub-to-busbar price differentials
- **Negative Prices**: Assesses frequency and revenue impact

### 2. Risk Quantification
The model quantifies five key risk factors:
- **Volume Risk**: Generation variability (0-5% adjustment)
- **Negative Price Risk**: Revenue loss from negative prices
- **Basis Risk**: Location-specific price differential risk (0-3%)
- **Technology Risk**: Solar (2%) vs Wind (3%)
- **Market Liquidity Risk**: ERCOT (1%), CAISO (2%), MISO (3%)

### 3. Price Forecasting
- Monte Carlo simulation (1,000 scenarios)
- Stochastic volatility modeling
- Forward curve integration
- Seasonal pattern incorporation

### 4. Valuation Outputs

For each asset, the model provides:

```python
{
    'fixed_prices': {
        'RT_Hub': 58.32,      # $/MWh
        'RT_Busbar': 56.78,   # $/MWh
        'DA_Hub': 55.40,      # $/MWh
        'DA_Busbar': 53.91    # $/MWh
    },
    'risk_metrics': {
        'merchant_p50_revenue': 125000000,  # $
        'merchant_p75_revenue': 118000000,  # $
        'var_95': 108000000,               # $
        'cvar_95': 102000000               # $
    },
    'generation_forecast': {
        'annual_mwh': 175000,
        'peak_mwh': 78750,
        'offpeak_mwh': 96250
    }
}
```

## 📈 Analysis Features

### P-Level Sensitivity Analysis
Analyze different risk appetites:
```python
# Default P75 (75% probability of beating merchant)
model.risk_level = 0.75

# Conservative P90
model.risk_level = 0.90

# Aggressive P50
model.risk_level = 0.50
```

### Market Comparison
The model scores each market (0-100) and provides recommendations:
- **>70**: Strong hedge candidate
- **50-70**: Partial hedge (50-75% of volume)
- **<50**: Consider staying merchant

### Curtailment Scenarios
Models revenue impact when generation stops during negative prices:
```python
curtailment_impact = model.calculate_curtailment_impact(data)
# Returns revenue loss percentage
```

## 🏗️ Model Architecture

```
RenewableAssetValuation/
├── Data Loading & Processing
│   ├── load_data()
│   └── data validation
├── Risk Analysis
│   ├── analyze_basis_risk()
│   ├── analyze_negative_prices()
│   └── calculate_risk_adjustments()
├── Forecasting
│   ├── forecast_prices()
│   └── generate_monthly_generation_forecast()
├── Valuation
│   ├── calculate_risk_adjusted_price()
│   └── calculate_var_cvar()
└── Reporting
    ├── run_full_analysis()
    └── evaluate_market_attractiveness()
```

## 📝 Key Assumptions

- **Discount Rate**: 5% for NPV calculations
- **Simulation Scenarios**: 1,000 Monte Carlo paths
- **Price Distribution**: Log-normal with historical volatility
- **Basis Risk**: Normal distribution with historical parameters
- **Peak Hours**: ~45% of total hours (market-specific)
- **No consideration of**: Capacity payments, RECs, taxes, O&M costs

## 🎨 Customization

### Modify Risk Parameters
```python
# Adjust risk premiums
model.technology_risk = {'Solar': 0.015, 'Wind': 0.025}
model.market_liquidity = {'ERCOT': 0.005, 'MISO': 0.025}
```

### Add Custom Markets
```python
model.markets.append('PJM')
model.assets['PJM'] = 'Wind'
model.peak_hours['PJM'] = {'weekdays': list(range(7, 23)), ...}
```

## 📊 Sample Output

```
============================================================
RENEWABLE ENERGY ASSET VALUATION ANALYSIS
============================================================

==================== ERCOT - Wind Asset ====================

Asset Type: Wind
Expected Annual Generation: 175,320 MWh

Risk-Adjusted Fixed Prices ($/MWh):
  RT Hub:    $58.32
  RT Busbar: $56.78
  DA Hub:    $55.40
  DA Busbar: $53.91

Risk Adjustments Applied:
  volume_risk: 3.45%
  negative_price_risk: 1.23%
  basis_risk: 2.10%
  technology_risk: 3.00%
  liquidity_risk: 1.00%
  TOTAL: 10.78%

Market Recommendation: Strong candidate for fixed-price hedge
Confidence Score: 78.5/100
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Developed for the 2025 Energy Markets Hackathon
- Data sources: ERCOT, MISO, CAISO market operators
- Methodology inspired by industry best practices in renewable energy finance

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the development team.

---

**Note**: This model is for demonstration purposes. Always conduct thorough due diligence and consult with energy market professionals before making commercial decisions.