
import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare the data
df_profile = pd.read_csv("results/profile_based_results.csv")
df_profile['fuel_price_timestamp'] = pd.to_datetime(df_profile['fuel_price_timestamp'])
df_profile.set_index('fuel_price_timestamp', inplace=True)

# Plot Clearing Price and Social Welfare
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
df_profile['clearing_price'].plot(ax=axs[0], title='Clearing Price Over Time')
axs[0].set_ylabel("Price")
df_profile['social_welfare'].plot(ax=axs[1], title='Social Welfare Over Time')
axs[1].set_ylabel("Welfare")
plt.tight_layout()
plt.show()

# Plot Fuel Prices
df_profile[['fuel_price_natural_gas', 'fuel_price_coal', 'fuel_price_oil', 'fuel_price_uranium']].plot(
    figsize=(14, 6), title='Fuel Prices Over Time')
plt.ylabel("Price")
plt.xlabel("Time")
plt.tight_layout()
plt.show()

# Fuel mix by type
fuel_types = ['coal', 'gas', 'wind', 'solar', 'hydro']
fuel_output_cols = [col for col in df_profile.columns if any(ft in col for ft in fuel_types) and 'output' in col and 'total' not in col]
fuel_output = df_profile[fuel_output_cols]
total_by_fuel_type = {
    fuel: fuel_output[[col for col in fuel_output_cols if fuel in col]].sum(axis=1)
    for fuel in fuel_types
}
df_fuel_mix = pd.DataFrame(total_by_fuel_type, index=df_profile.index)

# Plot fuel mix
df_fuel_mix.plot(figsize=(14, 6), title="Fuel-Type Generation Mix Over Time")
plt.ylabel("MWh")
plt.tight_layout()
plt.show()

# Emissions
emission_factors = {
    'coal': 1000,
    'gas': 500,
    'wind': 0,
    'solar': 0,
    'hydro': 0,
}
df_emissions = df_fuel_mix.multiply({k: emission_factors[k] for k in df_fuel_mix.columns})
df_emissions['total_emissions'] = df_emissions.sum(axis=1)

# Plot emissions
df_emissions['total_emissions'].plot(figsize=(14, 6), title="Estimated CO2 Emissions Over Time", color='black')
plt.ylabel("kg CO2")
plt.xlabel("Time")
plt.tight_layout()
plt.show()
