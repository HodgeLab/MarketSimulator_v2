
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "results/profile_based_results.csv"
df_profile = pd.read_csv(file_path)

# Convert to datetime
df_profile['fuel_price_timestamp'] = pd.to_datetime(df_profile['fuel_price_timestamp'])
df_profile.set_index('fuel_price_timestamp', inplace=True)

# Define fuel types
fuel_types = ['coal', 'gas', 'wind', 'solar', 'hydro']

# Extract relevant columns
fuel_output_cols = [col for col in df_profile.columns if any(ft in col for ft in fuel_types) and 'output' in col and 'total' not in col]

# Aggregate fuel outputs
fuel_output = df_profile[fuel_output_cols]
total_by_fuel_type = {
    fuel: fuel_output[[col for col in fuel_output_cols if fuel in col]].sum(axis=1)
    for fuel in fuel_types
}
df_fuel_mix = pd.DataFrame(total_by_fuel_type, index=df_profile.index)

# Emission factors (kg CO2/MWh)
emission_factors = {
    'coal': 1000,
    'gas': 500,
    'wind': 0,
    'solar': 0,
    'hydro': 0,
}

# Calculate emissions
df_emissions = df_fuel_mix.multiply({k: emission_factors[k] for k in df_fuel_mix.columns})
df_emissions['total_emissions'] = df_emissions.sum(axis=1)

# Totals
fuel_mix_total = df_fuel_mix.sum()
total_emissions = df_emissions['total_emissions'].sum()

# Write report
report = f"""
# Electricity Market Simulation Summary Report

## Dataset Overview
- Time Range: {df_profile.index.min().strftime('%Y-%m-%d')} to {df_profile.index.max().strftime('%Y-%m-%d')}
- Total Time Periods: {len(df_profile)}

## System-Wide Fuel Mix
- Wind: {fuel_mix_total['wind']:.2f} MWh
- Coal: {fuel_mix_total['coal']:.2f} MWh
- Gas: {fuel_mix_total['gas']:.2f} MWh
- Solar: {fuel_mix_total['solar']:.2f} MWh
- Hydro: {fuel_mix_total['hydro']:.2f} MWh

## Emissions Summary
- Estimated Total CO₂ Emissions: {total_emissions:.2e} kg
"""

with open("electricity_market_analysis_report.md", "w") as f:
    f.write(report)
