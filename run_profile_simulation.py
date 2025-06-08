# Profile-Based Multi-Year Electricity Market Simulation
# Supports importing time series profiles for all key parameters

import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Import base classes
from main import (
    Simulation, Market, DayAheadMarket, Consumer, StorageUnit,
    CarbonTax, RenewableMandate, MarketResult, load_config,
    SimulationState, Agent, Bid, AgentState
)

from utility import (
    UtilityAgent, UtilityPortfolio, GeneratorUnit,
    MarginalCostBidding, StrategicBidding, OptimalBidding,
    MarketPowerAnalyzer, update_renewable_availability
)

from run_utility_simulation import ExtendedDayAheadMarket, analyze_utility_performance

logger = logging.getLogger(__name__)

class ProfileManager:
    """Manages time series profiles for various simulation parameters"""
    
    def __init__(self, profiles_dir: str = "profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles = {}
        self.time_index = None
        
    def load_profile(self, profile_name: str, file_path: str, 
                    column_mapping: Dict[str, str] = None) -> pd.DataFrame:
        """Load a profile from CSV file"""
        
        file_path = self.profiles_dir / file_path
        logger.info(f"Loading profile '{profile_name}' from {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(file_path, nrows=1).columns else False)
        
        # Rename columns if mapping provided
        if column_mapping:
            df = df.rename(columns=column_mapping)
            
        # Store profile
        self.profiles[profile_name] = df
        
        # Set or validate time index
        if self.time_index is None:
            self.time_index = df.index
            logger.info(f"Set time index with {len(self.time_index)} periods")
        else:
            if len(df) != len(self.time_index):
                logger.warning(f"Profile '{profile_name}' has {len(df)} rows, expected {len(self.time_index)}")
                
        return df
    
    def get_value(self, profile_name: str, time_period: int, 
                  column: str = None, default: float = None) -> Union[float, Dict]:
        """Get value from profile for specific time period"""
        
        if profile_name not in self.profiles:
            return default
            
        profile = self.profiles[profile_name]
        
        if time_period >= len(profile):
            # Wrap around for shorter profiles
            time_period = time_period % len(profile)
            
        if column:
            return profile.iloc[time_period][column] if column in profile.columns else default
        else:
            return profile.iloc[time_period].to_dict()
    
    def get_profile_info(self) -> Dict:
        """Get information about loaded profiles"""
        info = {}
        for name, profile in self.profiles.items():
            info[name] = {
                'rows': len(profile),
                'columns': list(profile.columns),
                'start': profile.index[0],
                'end': profile.index[-1]
            }
        return info

class ProfileBasedUtilityAgent(UtilityAgent):
    """Extended utility agent that uses profiles for dynamic parameters"""
    
    def __init__(self, utility_id: str, portfolio: UtilityPortfolio,
                 bidding_strategy, profile_manager: ProfileManager = None,
                 enable_bilateral_contracts: bool = False):
        super().__init__(utility_id, portfolio, bidding_strategy, enable_bilateral_contracts)
        self.profile_manager = profile_manager
        
    def update_generator_parameters(self, time_period: int):
        """Update generator parameters based on profiles"""
        
        if not self.profile_manager:
            return
            
        # Update fuel prices
        fuel_prices = self.profile_manager.get_value('fuel_prices', time_period)
        if fuel_prices:
            for gen in self.portfolio.generators:
                if gen.fuel_type in fuel_prices:
                    # Update marginal cost based on fuel price and heat rate
                    fuel_price = fuel_prices[gen.fuel_type]
                    heat_rate = 1 / gen.efficiency if gen.efficiency > 0 else 10  # MMBtu/MWh
                    gen.marginal_cost = fuel_price * heat_rate
                    
        # Update renewable availability
        for gen in self.portfolio.generators:
            if gen.unit_type == 'wind':
                # Get wind capacity factor for this utility's location
                wind_cf = self.profile_manager.get_value(
                    'wind_cf', time_period, 
                    column=f'{self.agent_id}_{gen.unit_id}',
                    default=self.profile_manager.get_value('wind_cf', time_period, 'default', 0.3)
                )
                gen.availability = wind_cf
                
            elif gen.unit_type == 'solar':
                # Get solar capacity factor
                solar_cf = self.profile_manager.get_value(
                    'solar_cf', time_period,
                    column=f'{self.agent_id}_{gen.unit_id}',
                    default=self.profile_manager.get_value('solar_cf', time_period, 'default', 0.0)
                )
                gen.availability = solar_cf
                
            elif gen.unit_type == 'hydro':
                # Get hydro availability
                hydro_avail = self.profile_manager.get_value(
                    'hydro_availability', time_period,
                    column=gen.unit_id,
                    default=0.8
                )
                gen.availability = hydro_avail
                
        # Update forced outage rates if profile exists
        outage_rates = self.profile_manager.get_value('forced_outages', time_period)
        if outage_rates:
            for gen in self.portfolio.generators:
                if gen.unit_id in outage_rates:
                    gen.availability *= (1 - outage_rates[gen.unit_id])
    
    def create_bid(self, market_state: Dict, time_period: int) -> List[Bid]:
        """Create bids with profile-based updates"""
        
        # Update generator parameters first
        self.update_generator_parameters(time_period)
        
        # Add profile data to market state
        if self.profile_manager:
            market_state['fuel_prices'] = self.profile_manager.get_value('fuel_prices', time_period)
            market_state['time_period'] = time_period
            
        # Call parent method
        return super().create_bid(market_state, time_period)

class ProfileBasedConsumer(Consumer):
    """Consumer that uses load profiles"""
    
    def __init__(self, agent_id: str, profile_manager: ProfileManager,
                 load_profile_name: str = 'load_profile',
                 price_elasticity: float = -0.1):
        # Initialize with dummy profile, will be updated
        super().__init__(agent_id, [100], price_elasticity)
        self.profile_manager = profile_manager
        self.load_profile_name = load_profile_name
        
    def create_bid(self, market_state: Dict, time_period: int) -> List[Bid]:
        """Create demand bid using profile"""
        
        # Get demand from profile
        demand = self.profile_manager.get_value(
            self.load_profile_name, time_period,
            column=self.agent_id,
            default=100.0
        )
        
        # Apply price elasticity if last price available
        if 'last_price' in market_state and self.price_elasticity != 0:
            price_ratio = market_state['last_price'] / self.base_price
            demand_adjustment = 1 + self.price_elasticity * (price_ratio - 1)
            demand = demand * max(0.5, min(1.5, demand_adjustment))
            
        return [Bid(
            agent_id=self.agent_id,
            market_type='day_ahead',
            quantity=demand,
            price=1000.0,  # High willingness to pay
            time_period=time_period,
            bid_type='demand'
        )]

class ProfileBasedSimulation(Simulation):
    """Extended simulation that uses profiles for multi-year runs"""
    
    def __init__(self, config: Dict, profile_manager: ProfileManager):
        self.config = config
        self.profile_manager = profile_manager
        self.agents = []
        self.markets = {}
        self.policies = []
        self.state = SimulationState()
        self.utilities = []
        
        self._initialize_profile_simulation()
        
    def _initialize_profile_simulation(self):
        """Initialize simulation with profile-based agents"""
        
        # Create utility agents with profile support
        for util_config in self.config.get('utilities', []):
            utility = self._create_profile_utility(util_config)
            self.agents.append(utility)
            self.utilities.append(utility)
            logger.info(f"Created profile-based utility: {utility.agent_id}")
            
        # Create profile-based consumers
        for consumer_config in self.config.get('consumers', []):
            if 'load_profile_name' in consumer_config:
                consumer = ProfileBasedConsumer(
                    agent_id=consumer_config['agent_id'],
                    profile_manager=self.profile_manager,
                    load_profile_name=consumer_config['load_profile_name'],
                    price_elasticity=consumer_config.get('price_elasticity', -0.1)
                )
            else:
                # Fallback to regular consumer
                consumer = Consumer(**consumer_config['params'])
            self.agents.append(consumer)
            logger.info(f"Created consumer: {consumer.agent_id}")
            
        # Create storage units
        for storage_config in self.config.get('storage', []):
            agent = StorageUnit(**storage_config['params'])
            self.agents.append(agent)
            
        # Create markets
        for market_type in self.config.get('markets', ['day_ahead']):
            if market_type == 'day_ahead':
                self.markets[market_type] = ExtendedDayAheadMarket()
                
        # Create policies
        self._create_policies()
        
    def _create_profile_utility(self, util_config: Dict) -> ProfileBasedUtilityAgent:
        """Create a profile-based utility agent"""
        
        # Create generators
        generators = []
        for gen_config in util_config['generators']:
            gen = GeneratorUnit(**gen_config)
            generators.append(gen)
            
        # Create portfolio
        portfolio = UtilityPortfolio(
            utility_id=util_config['id'],
            generators=generators
        )
        
        # Create strategy
        strategy = self._create_strategy(util_config)
        
        # Create utility with profile manager
        utility = ProfileBasedUtilityAgent(
            utility_id=util_config['id'],
            portfolio=portfolio,
            bidding_strategy=strategy,
            profile_manager=self.profile_manager,
            enable_bilateral_contracts=util_config.get('enable_contracts', False)
        )
        
        return utility
        
    def _create_strategy(self, util_config: Dict):
        """Create bidding strategy"""
        strategy_type = util_config.get('strategy', 'marginal_cost')
        
        if strategy_type == 'marginal_cost':
            return MarginalCostBidding()
        elif strategy_type == 'strategic':
            return StrategicBidding(
                base_markup=util_config.get('markup', 0.1)
            )
        elif strategy_type == 'optimal':
            return OptimalBidding(
                forecast_horizon=util_config.get('forecast_horizon', 24)
            )
        else:
            return MarginalCostBidding()
            
    def _create_policies(self):
        """Create policies with potential profile-based parameters"""
        for policy_config in self.config.get('policies', []):
            if policy_config['type'] == 'carbon_tax':
                # Could make carbon tax profile-based in future
                self.policies.append(CarbonTax(policy_config['tax_rate']))
            elif policy_config['type'] == 'renewable_mandate':
                self.policies.append(RenewableMandate(policy_config['target']))
                
    def run(self, time_periods: int = None):
        """Run simulation for specified periods or length of profiles"""
        
        # Determine number of periods from profiles if not specified
        if time_periods is None:
            time_periods = len(self.profile_manager.time_index) if self.profile_manager.time_index is not None else 8760
            
        logger.info(f"Starting profile-based simulation for {time_periods} periods")
        
        # Market state initialization
        market_state = {
            'last_price': 50.0,
            'price_forecast': [50.0] * 24  # Will be updated dynamically
        }
        
        # Results collection
        results_data = []
        
        for t in range(time_periods):
            self.state.time_period = t
            
            # Update market state with profile data
            market_state['fuel_prices'] = self.profile_manager.get_value('fuel_prices', t)
            market_state['timestamp'] = self.profile_manager.profiles.get('load_profile', pd.DataFrame()).index[t] if 'load_profile' in self.profile_manager.profiles else t
            
            # Get price forecast from profile if available
            price_forecast = self.profile_manager.get_value('price_forecast', t)
            if price_forecast:
                market_state['price_forecast'] = [price_forecast.get(f'hour_{h}', 50) for h in range(24)]
                
            # Calculate supply ratio
            total_capacity = sum(u.portfolio.total_capacity for u in self.utilities)
            total_available = 0
            for utility in self.utilities:
                utility.update_generator_parameters(t)
                total_available += sum(g.capacity * g.availability for g in utility.portfolio.generators)
            market_state['supply_ratio'] = total_available / total_capacity if total_capacity > 0 else 1
            
            # Run markets
            for market_name, market in self.markets.items():
                result = market.run_auction(self.agents, t, market_state)
                self.state.update(market_name, result)
                
                # Apply policies
                for policy in self.policies:
                    policy.apply(result, self.agents)
                    
                # Update market state
                market_state['last_price'] = result.clearing_price
                
                # Collect results
                row_data = {
                    'time_period': t,
                    'timestamp': market_state.get('timestamp', t),
                    'market_type': market_name,
                    'clearing_price': result.clearing_price,
                    'total_cleared': result.total_cleared,
                    'social_welfare': result.social_welfare
                }
                
                # Add utility generation data
                for utility in self.utilities:
                    utility_total = 0
                    for gen in utility.portfolio.generators:
                        bid_id = f"{utility.agent_id}_{gen.unit_id}"
                        gen_output = result.cleared_quantities.get(bid_id, 0)
                        row_data[f'{utility.agent_id}_{gen.unit_type}_output'] = row_data.get(f'{utility.agent_id}_{gen.unit_type}_output', 0) + gen_output
                        utility_total += gen_output
                    row_data[f'{utility.agent_id}_total_output'] = utility_total
                    
                # Add fuel prices if available
                if market_state.get('fuel_prices'):
                    for fuel, price in market_state['fuel_prices'].items():
                        row_data[f'fuel_price_{fuel}'] = price
                        
                results_data.append(row_data)
                
            # Progress logging
            if t % 24 == 0:
                logger.info(f"Completed day {t//24 + 1} (period {t})")
            if t % (365 * 24) == 0 and t > 0:
                logger.info(f"Completed year {t//(365*24)}")
                
        # Convert to DataFrame
        results = pd.DataFrame(results_data)
        logger.info("Profile-based simulation completed")
        
        return results

def create_example_profiles(output_dir: str = "profiles"):
    """Create example profile CSV files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamps for 2 years of hourly data
    start_date = pd.Timestamp('2024-01-01')
    timestamps = pd.date_range(start=start_date, periods=2*365*24, freq='H')
    
    # 1. Load Profile - varies by hour, day of week, and season
    load_data = []
    for ts in timestamps:
        hour = ts.hour
        dow = ts.dayofweek
        month = ts.month
        
        # Base load pattern
        hour_factor = 0.7 + 0.3 * np.sin((hour - 6) * np.pi / 12) if 6 <= hour <= 18 else 0.7
        
        # Weekday vs weekend
        dow_factor = 1.0 if dow < 5 else 0.85
        
        # Seasonal variation
        season_factor = 1 + 0.2 * np.sin((month - 1) * np.pi / 6)  # Peak in summer
        
        # Random variation
        random_factor = 1 + np.random.normal(0, 0.05)
        
        city_load = 800 * hour_factor * dow_factor * season_factor * random_factor
        industrial_load = 400 * dow_factor * (0.9 + 0.1 * hour_factor) * random_factor
        
        load_data.append({
            'timestamp': ts,
            'city_load': city_load,
            'industrial_load': industrial_load,
            'total_load': city_load + industrial_load
        })
        
    load_df = pd.DataFrame(load_data)
    load_df.to_csv(f"{output_dir}/load_profile.csv", index=False)
    print(f"Created {output_dir}/load_profile.csv")
    
    # 2. Wind Capacity Factor Profile
    wind_data = []
    for i, ts in enumerate(timestamps):
        # Seasonal pattern - higher in winter
        seasonal = 0.35 + 0.15 * np.sin((ts.month - 1) * np.pi / 6 + np.pi)
        
        # Daily variation
        daily = 0.1 * np.sin(ts.hour * np.pi / 12)
        
        # Random walk for realistic variation
        if i == 0:
            random_walk = 0
        else:
            random_walk = 0.8 * wind_data[-1].get('random_walk', 0) + 0.2 * np.random.normal(0, 0.1)
            
        cf = max(0.05, min(0.95, seasonal + daily + random_walk))
        
        wind_data.append({
            'timestamp': ts,
            'default': cf,
            'offshore': cf * 1.2,  # Offshore typically higher
            'random_walk': random_walk
        })
        
    wind_df = pd.DataFrame(wind_data).drop('random_walk', axis=1)
    wind_df.to_csv(f"{output_dir}/wind_cf.csv", index=False)
    print(f"Created {output_dir}/wind_cf.csv")
    
    # 3. Solar Capacity Factor Profile
    solar_data = []
    for ts in timestamps:
        hour = ts.hour
        month = ts.month
        
        # No generation at night
        if hour < 6 or hour > 18:
            cf = 0
        else:
            # Peak at noon
            hour_factor = np.sin((hour - 6) * np.pi / 12)
            
            # Seasonal - higher in summer
            season_factor = 0.8 + 0.4 * np.sin((month - 3) * np.pi / 6)
            
            # Cloud cover random factor
            cloud_factor = np.random.beta(8, 2)  # Mostly clear with some clouds
            
            cf = hour_factor * season_factor * cloud_factor
            
        solar_data.append({
            'timestamp': ts,
            'default': cf
        })
        
    solar_df = pd.DataFrame(solar_data)
    solar_df.to_csv(f"{output_dir}/solar_cf.csv", index=False)
    print(f"Created {output_dir}/solar_cf.csv")
    
    # 4. Fuel Prices Profile
    fuel_data = []
    gas_price = 3.0  # Starting price $/MMBtu
    coal_price = 2.0  # $/MMBtu
    
    for i, ts in enumerate(timestamps):
        # Gas price with trend and volatility
        if i > 0:
            gas_change = 0.001 + 0.02 * np.random.normal(0, 1)
            gas_price *= (1 + gas_change)
            gas_price = max(1.5, min(10, gas_price))  # Keep in reasonable range
            
            # Coal less volatile
            coal_change = 0.0005 + 0.01 * np.random.normal(0, 1)
            coal_price *= (1 + coal_change)
            coal_price = max(1.0, min(5, coal_price))
            
        fuel_data.append({
            'timestamp': ts,
            'natural_gas': gas_price,
            'coal': coal_price,
            'oil': gas_price * 2.5,  # Oil typically more expensive
            'uranium': 0.5  # Nuclear fuel cost
        })
        
    fuel_df = pd.DataFrame(fuel_data)
    fuel_df.to_csv(f"{output_dir}/fuel_prices.csv", index=False)
    print(f"Created {output_dir}/fuel_prices.csv")
    
    # 5. Hydro Availability Profile (seasonal)
    hydro_data = []
    for ts in timestamps:
        month = ts.month
        
        # Spring peak from snowmelt
        if 3 <= month <= 6:
            availability = 0.9 + 0.1 * np.random.random()
        # Summer/fall lower
        elif 7 <= month <= 10:
            availability = 0.5 + 0.2 * np.random.random()
        # Winter moderate
        else:
            availability = 0.7 + 0.1 * np.random.random()
            
        hydro_data.append({
            'timestamp': ts,
            'CP_Hydro_1': availability
        })
        
    hydro_df = pd.DataFrame(hydro_data)
    hydro_df.to_csv(f"{output_dir}/hydro_availability.csv", index=False)
    print(f"Created {output_dir}/hydro_availability.csv")
    
    print(f"\nCreated example profiles for {len(timestamps)} hours ({len(timestamps)/(365*24):.1f} years)")
    
def main():
    """Main function to run profile-based simulation"""
    
    # Create example profiles if they don't exist
    if not os.path.exists("profiles/load_profile.csv"):
        print("Creating example profiles...")
        create_example_profiles()
    
    # Load configuration
    print("\nLoading configuration...")
    config = yaml.safe_load(open('profile_config.yaml', 'r'))
    
    # Initialize profile manager
    print("\nLoading profiles...")
    pm = ProfileManager("profiles")
    
    # Load all profiles specified in config
    for profile_config in config.get('profiles', []):
        pm.load_profile(
            profile_name=profile_config['name'],
            file_path=profile_config['file'],
            column_mapping=profile_config.get('column_mapping')
        )
    
    # Show loaded profiles
    print("\nLoaded profiles:")
    for name, info in pm.get_profile_info().items():
        print(f"  {name}: {info['rows']} rows, columns: {info['columns'][:3]}...")
    
    # Create and run simulation
    print("\nCreating simulation...")
    sim = ProfileBasedSimulation(config, pm)
    
    # Run for the length of the profiles (or specified periods)
    num_periods = config.get('simulation', {}).get('time_periods')
    print(f"\nRunning simulation for {num_periods or 'all'} periods...")
    results = sim.run(num_periods)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    output_file = 'results/profile_based_results.csv'
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"Results shape: {results.shape}")
    
    # Basic analysis
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    
    print(f"\nTime span: {results['timestamp'].min()} to {results['timestamp'].max()}")
    print(f"Total periods: {len(results)}")
    print(f"Average clearing price: ${results['clearing_price'].mean():.2f}/MWh")
    print(f"Price range: ${results['clearing_price'].min():.2f} - ${results['clearing_price'].max():.2f}/MWh")
    print(f"Total energy: {results['total_cleared'].sum()/1e6:.2f} TWh")
    
    # Utility summary
    print("\nUtility Generation Summary:")
    for utility in sim.utilities:
        col = f'{utility.agent_id}_total_output'
        if col in results.columns:
            total_gen = results[col].sum() / 1e6  # Convert to TWh
            avg_gen = results[col].mean()
            capacity_factor = avg_gen / utility.portfolio.total_capacity * 100
            print(f"  {utility.agent_id}:")
            print(f"    Total: {total_gen:.2f} TWh")
            print(f"    Average: {avg_gen:.0f} MW")
            print(f"    Capacity Factor: {capacity_factor:.1f}%")
    
    # Fuel price summary if available
    fuel_cols = [col for col in results.columns if col.startswith('fuel_price_')]
    if fuel_cols:
        print("\nAverage Fuel Prices:")
        for col in fuel_cols:
            fuel_type = col.replace('fuel_price_', '')
            avg_price = results[col].mean()
            print(f"  {fuel_type}: ${avg_price:.2f}/MMBtu")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_profile_visualizations(results, sim)
    
    print("\nSimulation complete!")

def create_profile_visualizations(results: pd.DataFrame, sim):
    """Create visualizations for profile-based results"""
    
    os.makedirs('results/plots', exist_ok=True)
    
    # Convert timestamp if it's not already datetime
    if 'timestamp' in results.columns and not pd.api.types.is_datetime64_any_dtype(results['timestamp']):
        results['timestamp'] = pd.to_datetime(results['timestamp'])
    
    # 1. Price duration curve by year
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract year if timestamp available
    if 'timestamp' in results.columns:
        results['year'] = results['timestamp'].dt.year
        years = results['year'].unique()
        
        ax = axes[0, 0]
        for year in years:
            year_data = results[results['year'] == year]['clearing_price'].values
            sorted_prices = np.sort(year_data)[::-1]
            percentiles = np.linspace(0, 100, len(sorted_prices))
            ax.plot(percentiles, sorted_prices, label=f'Year {year}', linewidth=2)
        ax.set_xlabel('Percentage of Time (%)')
        ax.set_ylabel('Price ($/MWh)')
        ax.set_title('Price Duration Curves by Year')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Monthly average prices
    ax = axes[0, 1]
    if 'timestamp' in results.columns:
        results['month'] = results['timestamp'].dt.to_period('M')
        monthly_avg = results.groupby('month')['clearing_price'].mean()
        monthly_avg.index = monthly_avg.index.to_timestamp()
        ax.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2)
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Price ($/MWh)')
        ax.set_title('Monthly Average Electricity Prices')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Generation mix over time
    ax = axes[1, 0]
    # Stack area plot of generation by type
    gen_types = ['coal', 'gas', 'wind', 'solar', 'hydro']
    colors = ['#696969', '#FFA500', '#1E90FF', '#FFD700', '#00CED1']
    
    # Aggregate generation by type across all utilities
    gen_by_type = pd.DataFrame(index=results.index)
    for gen_type in gen_types:
        gen_by_type[gen_type] = 0
        for utility in sim.utilities:
            col = f'{utility.agent_id}_{gen_type}_output'
            if col in results.columns:
                gen_by_type[gen_type] += results[col]
    
    # Resample to daily for cleaner visualization
    if 'timestamp' in results.columns:
        gen_by_type['timestamp'] = results['timestamp']
        daily_gen = gen_by_type.groupby(pd.Grouper(key='timestamp', freq='D')).mean()
        
        # Create stacked area plot
        ax.stackplot(daily_gen.index, 
                    [daily_gen[gt].values for gt in gen_types if gt in daily_gen.columns],
                    labels=[gt for gt in gen_types if gt in daily_gen.columns],
                    colors=[colors[i] for i, gt in enumerate(gen_types) if gt in daily_gen.columns],
                    alpha=0.8)
        ax.set_xlabel('Date')
        ax.set_ylabel('Generation (MW)')
        ax.set_title('Daily Generation Mix')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # 4. Fuel prices vs electricity prices
    ax = axes[1, 1]
    if 'fuel_price_natural_gas' in results.columns and 'timestamp' in results.columns:
        # Resample to daily
        # daily_data = results[['timestamp', 'clearing_price', 'fuel_price_natural_gas']].copy()
        # daily_data.set_index('timestamp', inplace=True)
        # daily_data = results.groupby(pd.Grouper(key='timestamp', freq='D')).mean()
        
        ax2 = ax.twinx()
        
        # Plot electricity price
        l1 = ax.plot(daily_data.index, daily_data['clearing_price'], 
                     'b-', label='Electricity Price', linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Electricity Price ($/MWh)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot gas price
        l2 = ax2.plot(daily_data.index, daily_data['fuel_price_natural_gas'], 
                      'r-', label='Natural Gas Price', linewidth=2)
        ax2.set_ylabel('Gas Price ($/MMBtu)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combined legend
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left')
        
        ax.set_title('Electricity vs Natural Gas Prices')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/profile_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional detailed plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Renewable penetration over time
    ax = axes[0, 0]
    if 'timestamp' in results.columns:
        # Calculate renewable percentage
        renewable_gen = gen_by_type[['wind', 'solar', 'hydro']].sum(axis=1)
        total_gen = gen_by_type[gen_types].sum(axis=1)
        renewable_pct = (renewable_gen / total_gen * 100).fillna(0)
        
        # Resample to daily
        renewable_pct_df = pd.DataFrame({
            'timestamp': results['timestamp'],
            'renewable_pct': renewable_pct
        })
        daily_renewable = renewable_pct_df.groupby(pd.Grouper(key='timestamp', freq='D')).mean()
        
        ax.plot(daily_renewable.index, daily_renewable['renewable_pct'], linewidth=2)
        ax.axhline(y=35, color='r', linestyle='--', label='35% RPS Target')
        ax.set_xlabel('Date')
        ax.set_ylabel('Renewable Percentage (%)')
        ax.set_title('Daily Renewable Penetration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    # 2. Load vs generation scatter
    ax = axes[0, 1]
    if 'city_load' in results.columns:
        total_load = results[['city_load', 'industrial_load']].sum(axis=1)
        ax.scatter(total_load, results['clearing_price'], 
                  c=results['time_period'] % (365*24), cmap='viridis', alpha=0.5, s=1)
        ax.set_xlabel('Total Load (MW)')
        ax.set_ylabel('Clearing Price ($/MWh)')
        ax.set_title('Price vs Load Relationship')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for time of year
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Hour of Year')
    
    # 3. Utility market shares
    ax = axes[1, 0]
    utility_totals = {}
    for utility in sim.utilities:
        col = f'{utility.agent_id}_total_output'
        if col in results.columns:
            utility_totals[utility.agent_id] = results[col].sum()
    
    if utility_totals:
        utilities = list(utility_totals.keys())
        values = list(utility_totals.values())
        colors_util = plt.cm.Set3(np.linspace(0, 1, len(utilities)))
        
        ax.pie(values, labels=utilities, autopct='%1.1f%%', colors=colors_util)
        ax.set_title('Total Generation Market Share')
    
    # 4. Price volatility over time
    ax = axes[1, 1]
    if 'timestamp' in results.columns:
        # Calculate rolling standard deviation
        results['price_volatility'] = results['clearing_price'].rolling(window=24*7).std()
        
        # Resample to weekly
        weekly_vol = results.groupby(pd.Grouper(key='timestamp', freq='W'))['price_volatility'].mean()
        
        ax.plot(weekly_vol.index, weekly_vol.values, linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price Volatility ($/MWh)')
        ax.set_title('Weekly Price Volatility (7-day rolling std)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/profile_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created visualizations in results/plots/")

# Make sure this is at the very end of the file
if __name__ == "__main__":
    main()