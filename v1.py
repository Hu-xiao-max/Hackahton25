"""
Renewable Energy Asset Valuation Model
For 2025 Hackathon: Merchant Renewable Energy Pricing
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RenewableAssetValuation:
    """
    A comprehensive model for valuing merchant renewable energy assets
    and determining risk-adjusted fixed prices for 5-year offtake contracts.
    """
    
    def __init__(self, file_path='HackathonDataset.xlsx'):
        """
        Initialize the valuation model with data paths and parameters.
        """
        self.file_path = file_path
        self.markets = ['ERCOT', 'MISO', 'CAISO']
        self.assets = {
            'ERCOT': 'Wind',
            'MISO': 'Wind', 
            'CAISO': 'Solar'
        }
        self.risk_level = 0.75  # P75 risk appetite
        self.forecast_years = [2026, 2027, 2028, 2029, 2030]
        
        # Peak hours definition for each market
        self.peak_hours = {
            'ERCOT': {'weekdays': list(range(7, 23)), 'saturday': [], 'sunday': []},
            'MISO': {'weekdays': list(range(8, 24)), 'saturday': [], 'sunday': []},
            'CAISO': {'weekdays': list(range(7, 23)), 'saturday': list(range(7, 23)), 'sunday': []}
        }
        
        # Load and process data
        self.data = {}
        self.forward_curves = {}
        self.load_data()
        
    def load_data(self):
        """Load historical data and forward curves from Excel file."""
        print("Loading data from Excel file...")
        
        for market in self.markets:
            # Read the Excel sheet starting from row 8 (0-indexed row 7)
            df = pd.read_excel(self.file_path, sheet_name=market, skiprows=7)
            
            # The first row after skipping should be the header
            # Let's handle the columns more carefully
            if len(df.columns) >= 13:
                # Rename columns based on expected structure
                df.columns = ['Date', 'HE', 'P_OP', 'Gen', 'RT_Busbar', 'RT_Hub', 
                             'DA_Busbar', 'DA_Hub', 'Empty1', 'Empty2', 
                             'Forward_Date', 'Peak', 'Off_Peak'] + list(df.columns[13:])
            else:
                # Handle case where there might be fewer columns
                col_names = ['Date', 'HE', 'P_OP', 'Gen', 'RT_Busbar', 'RT_Hub', 
                            'DA_Busbar', 'DA_Hub', 'Empty1', 'Empty2', 
                            'Forward_Date', 'Peak', 'Off_Peak']
                df.columns = col_names[:len(df.columns)]
            
            # Remove any rows where Date is not a valid date (like header rows that might have been included)
            df = df[pd.notna(df['Date'])]
            
            # Convert dates with error handling
            try:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Forward_Date'] = pd.to_datetime(df['Forward_Date'], errors='coerce')
            except Exception as e:
                print(f"Warning: Date conversion issue in {market}: {e}")
                # Try alternative date parsing
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
                df['Forward_Date'] = pd.to_datetime(df['Forward_Date'], format='mixed', errors='coerce')
            
            # Clean the data - remove rows with invalid dates in the Date column
            df = df.dropna(subset=['Date'])
            
            # Convert numeric columns
            numeric_cols = ['HE', 'Gen', 'RT_Busbar', 'RT_Hub', 'DA_Busbar', 'DA_Hub', 'Peak', 'Off_Peak']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Split historical and forward data
            historical = df[['Date', 'HE', 'P_OP', 'Gen', 'RT_Busbar', 'RT_Hub', 
                           'DA_Busbar', 'DA_Hub']].copy()
            historical = historical.dropna(subset=['Gen', 'RT_Hub'])  # Must have generation and price data
            
            # Extract forward price data
            forward = df[['Forward_Date', 'Peak', 'Off_Peak']].copy()
            forward = forward.dropna(subset=['Forward_Date', 'Peak', 'Off_Peak'])
            forward = forward.drop_duplicates(subset=['Forward_Date'])
            
            self.data[market] = historical
            self.forward_curves[market] = forward
            
            print(f"  {market}: Loaded {len(historical)} historical records and {len(forward)} forward price points")
            
        print(f"Data loaded successfully for {len(self.markets)} markets")
        
    def calculate_generation_profile(self, market):
        """
        Calculate expected generation profile by month and peak/off-peak periods.
        """
        df = self.data[market]
        asset_type = self.assets[market]
        
        # Ensure we have valid data
        if df.empty:
            print(f"Warning: No data available for {market}")
            return None
            
        # Calculate capacity factor patterns
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['Hour'] = df['HE']
        
        # Group by month and peak/off-peak
        monthly_cf = df.groupby(['Month', 'P_OP'])['Gen'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        # Calculate seasonal patterns for different asset types
        if asset_type == 'Solar':
            # Solar has strong seasonal and diurnal patterns
            hourly_cf = df.groupby(['Month', 'Hour'])['Gen'].mean().reset_index()
            seasonal_factor = df.groupby('Month')['Gen'].mean() / (df['Gen'].mean() + 1e-6)  # Avoid division by zero
        else:
            # Wind has different seasonal patterns
            hourly_cf = df.groupby(['Month', 'Hour'])['Gen'].mean().reset_index()
            seasonal_factor = df.groupby('Month')['Gen'].mean() / (df['Gen'].mean() + 1e-6)
        
        return {
            'monthly_cf': monthly_cf,
            'hourly_cf': hourly_cf,
            'seasonal_factor': seasonal_factor.to_dict(),
            'asset_type': asset_type
        }
    
    def analyze_basis_risk(self, market):
        """
        Analyze basis risk between hub and busbar prices.
        """
        df = self.data[market]
        
        if df.empty:
            return {
                'RT': {'mean': 0, 'std': 0, 'percentiles': {}, 'negative_hours': 0},
                'DA': {'mean': 0, 'std': 0, 'percentiles': {}, 'negative_hours': 0},
                'correlation': pd.DataFrame()
            }
        
        # Calculate basis differentials
        df['RT_Basis'] = df['RT_Busbar'] - df['RT_Hub']
        df['DA_Basis'] = df['DA_Busbar'] - df['DA_Hub']
        
        # Remove infinite and NaN values
        df['RT_Basis'] = df['RT_Basis'].replace([np.inf, -np.inf], np.nan)
        df['DA_Basis'] = df['DA_Basis'].replace([np.inf, -np.inf], np.nan)
        
        # Statistical analysis of basis
        basis_stats = {
            'RT': {
                'mean': df['RT_Basis'].mean() if not df['RT_Basis'].isna().all() else 0,
                'std': df['RT_Basis'].std() if not df['RT_Basis'].isna().all() else 0,
                'percentiles': df['RT_Basis'].dropna().quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict() if not df['RT_Basis'].isna().all() else {},
                'negative_hours': (df['RT_Basis'] < 0).sum() / max(len(df), 1)
            },
            'DA': {
                'mean': df['DA_Basis'].mean() if not df['DA_Basis'].isna().all() else 0,
                'std': df['DA_Basis'].std() if not df['DA_Basis'].isna().all() else 0,
                'percentiles': df['DA_Basis'].dropna().quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict() if not df['DA_Basis'].isna().all() else {},
                'negative_hours': (df['DA_Basis'] < 0).sum() / max(len(df), 1)
            }
        }
        
        # Correlation analysis
        correlation_df = df[['RT_Basis', 'DA_Basis', 'Gen']].dropna()
        if not correlation_df.empty:
            basis_stats['correlation'] = correlation_df.corr()
        else:
            basis_stats['correlation'] = pd.DataFrame()
        
        return basis_stats
    
    def analyze_negative_prices(self, market):
        """
        Analyze frequency and impact of negative price events.
        """
        df = self.data[market]
        
        negative_stats = {}
        for price_col in ['RT_Busbar', 'RT_Hub', 'DA_Busbar', 'DA_Hub']:
            neg_df = df[df[price_col] < 0]
            negative_stats[price_col] = {
                'frequency': len(neg_df) / len(df),
                'avg_negative_price': neg_df[price_col].mean() if len(neg_df) > 0 else 0,
                'total_hours': len(neg_df),
                'avg_generation_during_negative': neg_df['Gen'].mean() if len(neg_df) > 0 else 0,
                'revenue_impact': (neg_df[price_col] * neg_df['Gen']).sum() if len(neg_df) > 0 else 0
            }
        
        # Analyze curtailment scenarios
        curtailment_impact = self.calculate_curtailment_impact(df)
        negative_stats['curtailment_impact'] = curtailment_impact
        
        return negative_stats
    
    def calculate_curtailment_impact(self, df):
        """
        Calculate impact of curtailment during negative prices.
        """
        # Scenario: No generation when prices are negative
        df_curtailed = df.copy()
        
        revenue_impact = {}
        for price_col in ['RT_Hub', 'DA_Hub']:
            # Original revenue
            original_revenue = (df[price_col] * df['Gen']).sum()
            
            # Curtailed revenue (zero generation when price < 0)
            df_curtailed['Gen_Curtailed'] = df['Gen'].where(df[price_col] >= 0, 0)
            curtailed_revenue = (df[price_col] * df_curtailed['Gen_Curtailed']).sum()
            
            revenue_impact[price_col] = {
                'original_revenue': original_revenue,
                'curtailed_revenue': curtailed_revenue,
                'revenue_loss_pct': (original_revenue - curtailed_revenue) / original_revenue * 100
            }
        
        return revenue_impact
    
    def forecast_prices(self, market):
        """
        Forecast prices for 2026-2030 using forward curves and historical patterns.
        """
        historical = self.data[market]
        forward = self.forward_curves[market]
        
        # Calculate historical price volatility
        historical['RT_Returns'] = historical['RT_Hub'].pct_change()
        historical['DA_Returns'] = historical['DA_Hub'].pct_change()
        
        rt_vol = historical['RT_Returns'].std() * np.sqrt(8760)  # Annualized
        da_vol = historical['DA_Returns'].std() * np.sqrt(8760)
        
        # Calculate basis statistics
        basis_stats = self.analyze_basis_risk(market)
        
        # Generate price scenarios using Monte Carlo
        n_simulations = 1000
        price_scenarios = {}
        
        for year in self.forecast_years:
            year_scenarios = {
                'RT_Hub': [],
                'DA_Hub': [],
                'RT_Busbar': [],
                'DA_Busbar': []
            }
            
            # Get forward prices for the year
            year_forward = forward[forward['Forward_Date'].dt.year == year]
            
            for sim in range(n_simulations):
                # Simulate hub prices with volatility
                rt_shock = np.random.normal(0, rt_vol/np.sqrt(12), 12)  # Monthly shocks
                da_shock = np.random.normal(0, da_vol/np.sqrt(12), 12)
                
                # Apply shocks to forward curves
                monthly_rt_hub = []
                monthly_da_hub = []
                
                for month in range(1, 13):
                    month_forward = year_forward[year_forward['Forward_Date'].dt.month == month]
                    if not month_forward.empty:
                        base_peak = month_forward['Peak'].values[0]
                        base_offpeak = month_forward['Off_Peak'].values[0]
                        
                        # Weighted average for hub price (assuming 45% peak hours)
                        base_price = 0.45 * base_peak + 0.55 * base_offpeak
                        
                        rt_price = base_price * (1 + rt_shock[month-1])
                        da_price = base_price * (1 + da_shock[month-1] * 0.7)  # DA less volatile
                        
                        monthly_rt_hub.append(rt_price)
                        monthly_da_hub.append(da_price)
                
                year_scenarios['RT_Hub'].append(np.mean(monthly_rt_hub))
                year_scenarios['DA_Hub'].append(np.mean(monthly_da_hub))
                
                # Add basis to get busbar prices
                rt_basis = np.random.normal(basis_stats['RT']['mean'], 
                                          basis_stats['RT']['std'])
                da_basis = np.random.normal(basis_stats['DA']['mean'], 
                                          basis_stats['DA']['std'])
                
                year_scenarios['RT_Busbar'].append(np.mean(monthly_rt_hub) + rt_basis)
                year_scenarios['DA_Busbar'].append(np.mean(monthly_da_hub) + da_basis)
            
            price_scenarios[year] = year_scenarios
        
        return price_scenarios
    
    def calculate_var_cvar(self, revenues, confidence_level=0.75):
        """
        Calculate Value at Risk and Conditional Value at Risk.
        """
        sorted_revenues = np.sort(revenues)
        var_index = int((1 - confidence_level) * len(sorted_revenues))
        
        var = sorted_revenues[var_index]
        cvar = sorted_revenues[:var_index].mean()
        
        return var, cvar
    
    def calculate_risk_adjusted_price(self, market):
        """
        Calculate risk-adjusted fixed prices for the 5-year term.
        """
        # Get generation profile
        gen_profile = self.calculate_generation_profile(market)
        
        # Get price scenarios
        price_scenarios = self.forecast_prices(market)
        
        # Calculate expected revenues under merchant scenario
        merchant_revenues = []
        
        for sim in range(1000):
            total_revenue = 0
            for year in self.forecast_years:
                # Use average generation and prices for simplification
                avg_gen = self.data[market]['Gen'].mean()
                rt_hub_price = price_scenarios[year]['RT_Hub'][sim % len(price_scenarios[year]['RT_Hub'])]
                
                annual_revenue = avg_gen * rt_hub_price * 8760
                total_revenue += annual_revenue / (1.05 ** (year - 2026 + 1))  # Discount at 5%
            
            merchant_revenues.append(total_revenue)
        
        # Calculate P75 threshold
        p75_revenue = np.percentile(merchant_revenues, 25)  # 75% probability of beating this
        
        # Calculate fixed prices that would yield P75 revenue
        avg_annual_gen = self.data[market]['Gen'].mean() * 8760
        pv_factor = sum([1/(1.05**i) for i in range(1, 6)])  # 5-year PV factor
        
        base_fixed_price = p75_revenue / (avg_annual_gen * pv_factor)
        
        # Calculate prices for different products
        basis_stats = self.analyze_basis_risk(market)
        
        results = {
            'market': market,
            'asset_type': self.assets[market],
            'expected_annual_generation_mwh': avg_annual_gen,
            'fixed_prices': {
                'RT_Hub': base_fixed_price,
                'RT_Busbar': base_fixed_price + basis_stats['RT']['mean'],
                'DA_Hub': base_fixed_price * 0.95,  # DA typically lower risk
                'DA_Busbar': base_fixed_price * 0.95 + basis_stats['DA']['mean']
            },
            'risk_metrics': {
                'merchant_p50_revenue': np.median(merchant_revenues),
                'merchant_p75_revenue': p75_revenue,
                'var_95': self.calculate_var_cvar(merchant_revenues, 0.95)[0],
                'cvar_95': self.calculate_var_cvar(merchant_revenues, 0.95)[1]
            },
            'basis_risk': basis_stats,
            'negative_price_analysis': self.analyze_negative_prices(market)
        }
        
        # Add risk adjustments
        risk_adjustments = self.calculate_risk_adjustments(market, results)
        results['risk_adjustments'] = risk_adjustments
        
        # Apply risk adjustments to final prices
        for product in results['fixed_prices']:
            results['fixed_prices'][product] *= (1 - risk_adjustments['total_adjustment'])
        
        return results
    
    def calculate_risk_adjustments(self, market, base_results):
        """
        Calculate risk adjustments based on various factors.
        """
        adjustments = {}
        
        # Volume risk adjustment (higher for intermittent resources)
        gen_std = self.data[market]['Gen'].std()
        gen_mean = self.data[market]['Gen'].mean()
        cv = gen_std / gen_mean  # Coefficient of variation
        adjustments['volume_risk'] = min(cv * 0.1, 0.05)  # Cap at 5%
        
        # Negative price risk
        neg_freq = base_results['negative_price_analysis']['RT_Hub']['frequency']
        adjustments['negative_price_risk'] = neg_freq * 0.5  # 50% of negative hour frequency
        
        # Basis risk
        basis_vol = base_results['basis_risk']['RT']['std']
        adjustments['basis_risk'] = min(basis_vol / 100, 0.03)  # Cap at 3%
        
        # Technology-specific risk
        if self.assets[market] == 'Solar':
            adjustments['technology_risk'] = 0.02  # Solar more predictable
        else:
            adjustments['technology_risk'] = 0.03  # Wind less predictable
        
        # Market liquidity risk
        market_liquidity = {
            'ERCOT': 0.01,  # Most liquid
            'CAISO': 0.02,  # Moderate liquidity
            'MISO': 0.03   # Least liquid
        }
        adjustments['liquidity_risk'] = market_liquidity.get(market, 0.02)
        
        adjustments['total_adjustment'] = sum(adjustments.values())
        
        return adjustments
    
    def generate_monthly_generation_forecast(self, market):
        """
        Generate detailed monthly generation forecast for 2026-2030.
        """
        df = self.data[market]
        
        # Calculate monthly patterns
        monthly_stats = df.groupby([df['Date'].dt.month, 'P_OP'])['Gen'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        monthly_stats.columns = ['Month', 'Period', 'Mean_Gen', 'Std_Gen', 'Count']
        
        # Generate forecast
        forecast = []
        for year in self.forecast_years:
            for month in range(1, 13):
                peak_data = monthly_stats[(monthly_stats['Month'] == month) & 
                                         (monthly_stats['Period'] == 'P')]
                offpeak_data = monthly_stats[(monthly_stats['Month'] == month) & 
                                            (monthly_stats['Period'] == 'OP')]
                
                # Calculate hours in month
                days_in_month = pd.Period(f'{year}-{month:02d}').days_in_month
                
                # Estimate peak/off-peak hours based on market definitions
                if market == 'ERCOT':
                    peak_hours_per_day = 16  # HE 7-22
                    peak_days = days_in_month * 5/7  # Weekdays only
                elif market == 'MISO':
                    peak_hours_per_day = 16  # HE 8-23
                    peak_days = days_in_month * 5/7
                else:  # CAISO
                    peak_hours_per_day = 16  # HE 7-22
                    peak_days = days_in_month * 6/7  # Mon-Sat
                
                peak_hours = peak_hours_per_day * peak_days
                offpeak_hours = days_in_month * 24 - peak_hours
                
                forecast.append({
                    'Year': year,
                    'Month': month,
                    'Peak_Generation_MWh': peak_data['Mean_Gen'].values[0] * peak_hours if not peak_data.empty else 0,
                    'OffPeak_Generation_MWh': offpeak_data['Mean_Gen'].values[0] * offpeak_hours if not offpeak_data.empty else 0,
                    'Total_Generation_MWh': (peak_data['Mean_Gen'].values[0] * peak_hours if not peak_data.empty else 0) + 
                                           (offpeak_data['Mean_Gen'].values[0] * offpeak_hours if not offpeak_data.empty else 0)
                })
        
        return pd.DataFrame(forecast)
    
    def evaluate_market_attractiveness(self):
        """
        Evaluate which markets are more amenable to hedging.
        """
        market_scores = {}
        
        for market in self.markets:
            results = self.calculate_risk_adjusted_price(market)
            
            score = {
                'market': market,
                'asset_type': self.assets[market],
                'merchant_volatility': np.std([results['risk_metrics']['merchant_p50_revenue'],
                                              results['risk_metrics']['merchant_p75_revenue']]),
                'negative_price_frequency': results['negative_price_analysis']['RT_Hub']['frequency'],
                'basis_volatility': results['basis_risk']['RT']['std'],
                'recommended_action': '',
                'confidence_score': 0
            }
            
            # Calculate confidence score (0-100)
            volatility_score = max(0, 100 - score['merchant_volatility'] / 1000)
            neg_price_score = max(0, 100 - score['negative_price_frequency'] * 500)
            basis_score = max(0, 100 - score['basis_volatility'])
            
            score['confidence_score'] = (volatility_score + neg_price_score + basis_score) / 3
            
            # Recommendation
            if score['confidence_score'] > 70:
                score['recommended_action'] = 'Strong candidate for fixed-price hedge'
            elif score['confidence_score'] > 50:
                score['recommended_action'] = 'Consider partial hedge (50-75% of volume)'
            else:
                score['recommended_action'] = 'Consider staying merchant or minimal hedge'
            
            market_scores[market] = score
        
        return market_scores
    
    def run_full_analysis(self):
        """
        Run complete analysis for all markets and generate results.
        """
        print("\n" + "="*60)
        print("RENEWABLE ENERGY ASSET VALUATION ANALYSIS")
        print("="*60)
        
        all_results = {}
        
        for market in self.markets:
            print(f"\n{'='*20} {market} - {self.assets[market]} Asset {'='*20}")
            
            # Calculate risk-adjusted prices
            results = self.calculate_risk_adjusted_price(market)
            all_results[market] = results
            
            # Print summary
            print(f"\nAsset Type: {results['asset_type']}")
            print(f"Expected Annual Generation: {results['expected_annual_generation_mwh']:,.0f} MWh")
            
            print("\nRisk-Adjusted Fixed Prices ($/MWh):")
            print(f"  RT Hub:    ${results['fixed_prices']['RT_Hub']:.2f}")
            print(f"  RT Busbar: ${results['fixed_prices']['RT_Busbar']:.2f}")
            print(f"  DA Hub:    ${results['fixed_prices']['DA_Hub']:.2f}")
            print(f"  DA Busbar: ${results['fixed_prices']['DA_Busbar']:.2f}")
            
            print("\nRisk Adjustments Applied:")
            for risk_type, adjustment in results['risk_adjustments'].items():
                if risk_type != 'total_adjustment':
                    print(f"  {risk_type}: {adjustment*100:.2f}%")
            print(f"  TOTAL: {results['risk_adjustments']['total_adjustment']*100:.2f}%")
            
            print("\nNegative Price Analysis:")
            print(f"  RT Hub Negative Frequency: {results['negative_price_analysis']['RT_Hub']['frequency']*100:.2f}%")
            print(f"  Curtailment Revenue Impact: {results['negative_price_analysis']['curtailment_impact']['RT_Hub']['revenue_loss_pct']:.2f}%")
            
            # Generate monthly forecast
            monthly_forecast = self.generate_monthly_generation_forecast(market)
            print(f"\n2026 Generation Forecast (First 3 months):")
            print(monthly_forecast[monthly_forecast['Year'] == 2026].head(3).to_string(index=False))
        
        # Market comparison
        print("\n" + "="*60)
        print("MARKET ATTRACTIVENESS COMPARISON")
        print("="*60)
        
        market_scores = self.evaluate_market_attractiveness()
        
        for market, score in market_scores.items():
            print(f"\n{market}:")
            print(f"  Confidence Score: {score['confidence_score']:.1f}/100")
            print(f"  Recommendation: {score['recommended_action']}")
        
        return all_results, market_scores

# Main execution
if __name__ == "__main__":
    # Initialize the model
    model = RenewableAssetValuation('HackathonDataset.xlsx')
    
    # Run full analysis
    results, market_scores = model.run_full_analysis()
    
    # Additional analysis for different P-levels
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS: DIFFERENT P-LEVELS")
    print("="*60)
    
    p_levels = [0.50, 0.75, 0.90]
    
    for p_level in p_levels:
        print(f"\nP{int(p_level*100)} Analysis:")
        model.risk_level = p_level
        
        for market in model.markets:
            results = model.calculate_risk_adjusted_price(market)
            print(f"  {market} RT Hub: ${results['fixed_prices']['RT_Hub']:.2f}/MWh")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)