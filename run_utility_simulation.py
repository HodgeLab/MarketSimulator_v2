# Fixed version of the utility simulation integration

import numpy as np
import pandas as pd
import yaml
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from main simulator
from main import (
    Simulation, Market, DayAheadMarket, Consumer, StorageUnit,
    CarbonTax, RenewableMandate, MarketResult, load_config,
    SimulationState, Agent, Bid, AgentState
)

# Import utility agent components
from utility import (
    UtilityAgent, UtilityPortfolio, GeneratorUnit,
    MarginalCostBidding, StrategicBidding, OptimalBidding,
    MarketPowerAnalyzer, update_renewable_availability
)

class ExtendedDayAheadMarket(DayAheadMarket):
    """Extended market that properly handles utility agents"""
    
    def run_auction(self, agents: List[Agent], time_period: int,
                   market_state: Dict) -> MarketResult:
        """Run day-ahead market auction with utility support"""
        
        # Collect bids
        all_bids = []
        for agent in agents:
            bids = agent.create_bid(market_state, time_period)
            all_bids.extend(bids)
            
        # Separate supply and demand
        supply_bids = [bid for bid in all_bids if bid.bid_type == 'supply']
        demand_bids = [bid for bid in all_bids if bid.bid_type == 'demand']
        
        # Clear market
        result = self.clearing_engine.clear_market(supply_bids, demand_bids, time_period)
        self.results_history.append(result)
        
        # Update all agents
        for agent in agents:
            if isinstance(agent, UtilityAgent):
                # Utilities handle their own internal dispatch
                agent.update_state(result, 0)
            elif isinstance(agent, StorageUnit):
                # Storage has special handling
                agent.update_state(result, 0)
            else:
                # Regular agents (consumers, traditional generators)
                if agent.agent_id in result.cleared_quantities:
                    cleared_qty = result.cleared_quantities[agent.agent_id]
                    agent.update_state(result, cleared_qty)
                    
        return result

class ExtendedSimulation(Simulation):
    """Extended simulation class that handles utility agents"""
    
    def __init__(self, config: Dict):
        # Initialize parent attributes directly
        self.config = config
        self.agents = []
        self.markets = {}
        self.policies = []
        self.state = SimulationState()
        self.utilities = []  # Track utility agents separately
        
        self._initialize_extended_simulation()
        
    def _initialize_extended_simulation(self):
        """Initialize with support for utility agents"""
        
        # Create utility agents
        for util_config in self.config.get('utilities', []):
            utility = self._create_utility_from_config(util_config)
            self.agents.append(utility)
            self.utilities.append(utility)
            logger.info(f"Created utility: {utility.agent_id} with {len(utility.portfolio.generators)} generators")
            
        # Create traditional agents (consumers, storage)
        for consumer_config in self.config.get('consumers', []):
            agent = Consumer(**consumer_config['params'])
            self.agents.append(agent)
            logger.info(f"Created consumer: {agent.agent_id}")
            
        for storage_config in self.config.get('storage', []):
            agent = StorageUnit(**storage_config['params'])
            self.agents.append(agent)
            logger.info(f"Created storage: {agent.agent_id}")
            
        # Create markets - use extended market for utilities
        for market_type in self.config.get('markets', ['day_ahead']):
            if market_type == 'day_ahead':
                self.markets[market_type] = ExtendedDayAheadMarket()
                logger.info("Created ExtendedDayAheadMarket for utility support")
                
        # Create policies
        for policy_config in self.config.get('policies', []):
            if policy_config['type'] == 'carbon_tax':
                self.policies.append(CarbonTax(policy_config['tax_rate']))
            elif policy_config['type'] == 'renewable_mandate':
                self.policies.append(RenewableMandate(policy_config['target']))
                
    def _create_utility_from_config(self, util_config: Dict) -> UtilityAgent:
        """Create a utility agent from configuration"""
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
        
        # Select strategy
        strategy_type = util_config.get('strategy', 'marginal_cost')
        if strategy_type == 'marginal_cost':
            strategy = MarginalCostBidding()
        elif strategy_type == 'strategic':
            strategy = StrategicBidding(
                base_markup=util_config.get('markup', 0.1)
            )
        elif strategy_type == 'optimal':
            strategy = OptimalBidding(
                forecast_horizon=util_config.get('forecast_horizon', 24)
            )
            
        # Create utility
        utility = UtilityAgent(
            utility_id=util_config['id'],
            portfolio=portfolio,
            bidding_strategy=strategy,
            enable_bilateral_contracts=util_config.get('enable_contracts', False)
        )
        
        # Add bilateral contracts if configured
        for contract in self.config.get('bilateral_contracts', []):
            if contract['seller'] == util_config['id']:
                utility.add_bilateral_contract(
                    contract_id=contract['contract_id'],
                    quantity=contract['quantity'],
                    start_period=contract['start_period'],
                    end_period=contract['end_period'],
                    strike_price=contract['strike_price']
                )
                
        return utility
        
    def run(self, time_periods: int):
        """Run simulation with enhanced features for utilities"""
        logger.info(f"Starting extended simulation for {time_periods} periods")
        logger.info(f"Number of utilities: {len(self.utilities)}")
        for utility in self.utilities:
            logger.info(f"  - {utility.agent_id}: {utility.portfolio.total_capacity} MW")
        
        # Market power analyzer
        market_power = MarketPowerAnalyzer()
        
        market_state = {
            'last_price': 50.0,
            'price_forecast': [40 + 20*np.sin(2*np.pi*h/24) for h in range(24)]
        }
        
        for t in range(time_periods):
            self.state.time_period = t
            hour = t % 24
            
            # Update renewable availability for all utilities
            for utility in self.utilities:
                update_renewable_availability(utility, hour)
                
            # Calculate supply ratio for strategic bidding
            total_capacity = sum(u.portfolio.total_capacity for u in self.utilities)
            total_available = sum(
                sum(g.capacity * g.availability for g in u.portfolio.generators)
                for u in self.utilities
            )
            market_state['supply_ratio'] = total_available / total_capacity if total_capacity > 0 else 1
            
            # Run market
            for market_name, market in self.markets.items():
                result = market.run_auction(self.agents, t, market_state)
                self.state.update(market_name, result)
                
                # Log clearing information for first few periods
                if t < 3:
                    logger.info(f"Period {t}: Price=${result.clearing_price:.2f}, Total={result.total_cleared:.1f} MW")
                    logger.info(f"  Cleared quantities: {len(result.cleared_quantities)} units")
                
                # Apply policies
                for policy in self.policies:
                    policy.apply(result, self.agents)
                    
                # Update market state
                market_state['last_price'] = result.clearing_price
                
                # Calculate market power metrics
                if t % 24 == 0:  # Daily calculation
                    hhi = market_power.calculate_hhi(
                        self.utilities, 
                        self.state.market_results[market_name][-24:]
                    )
                    self.state.system_metrics['hhi'].append(hhi)
                    
                    # Calculate Lerner indices
                    for utility in self.utilities:
                        lerner = market_power.calculate_lerner_index(
                            utility, result.clearing_price
                        )
                        self.state.system_metrics[f'lerner_{utility.agent_id}'].append(lerner)
                        
            if t % 24 == 0:
                logger.info(f"Completed day {t//24 + 1}")
                
        logger.info("Extended simulation completed")
        return self.get_extended_results()
        
    def get_extended_results(self) -> pd.DataFrame:
        """Get results with utility-specific metrics - FIXED VERSION"""
        # Get base results
        results = self.get_results()
        
        # Create a dictionary to track generation by type for each utility
        utility_columns = {}
        
        # Initialize all columns
        for utility in self.utilities:
            uid = utility.agent_id
            for gen_type in ['coal', 'gas', 'wind', 'solar', 'hydro', 'battery']:
                col_name = f'{uid}_{gen_type}_output'
                utility_columns[col_name] = []
            utility_columns[f'{uid}_total_output'] = []
        
        # Process each time period
        for t in range(len(results)):
            # Initialize generation by type for each utility
            utility_gen_by_type = {}
            utility_total_gen = {}
            
            for utility in self.utilities:
                utility_gen_by_type[utility.agent_id] = {
                    'coal': 0.0, 'gas': 0.0, 'wind': 0.0, 
                    'solar': 0.0, 'hydro': 0.0, 'battery': 0.0
                }
                utility_total_gen[utility.agent_id] = 0.0
            
            # Get the market result for this time period
            if t < len(self.state.market_results['day_ahead']):
                market_result = self.state.market_results['day_ahead'][t]
                
                # Process cleared quantities
                for bid_id, quantity in market_result.cleared_quantities.items():
                    # Find which utility this belongs to
                    for utility in self.utilities:
                        # Check if this bid belongs to this utility
                        for gen in utility.portfolio.generators:
                            expected_bid_id = f"{utility.agent_id}_{gen.unit_id}"
                            if bid_id == expected_bid_id:
                                utility_gen_by_type[utility.agent_id][gen.unit_type] += quantity
                                utility_total_gen[utility.agent_id] += quantity
                                break
            
            # Record the data for this time period
            for utility in self.utilities:
                uid = utility.agent_id
                # Record generation by type
                for gen_type in ['coal', 'gas', 'wind', 'solar', 'hydro', 'battery']:
                    col_name = f'{uid}_{gen_type}_output'
                    utility_columns[col_name].append(utility_gen_by_type[uid][gen_type])
                
                # Record total generation
                col_name = f'{uid}_total_output'
                utility_columns[col_name].append(utility_total_gen[uid])
        
        # Add all utility columns to results
        for col_name, values in utility_columns.items():
            if len(values) == len(results):
                results[col_name] = values
            else:
                logger.warning(f"Column {col_name} has {len(values)} values but results has {len(results)} rows")
            
        return results

def analyze_utility_performance(simulation: ExtendedSimulation, results: pd.DataFrame):
    """Analyze and visualize utility performance"""
    
    # Create output directory if it doesn't exist
    os.makedirs('results/plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Utility Performance Analysis', fontsize=16)
    
    # 1. Market share over time
    ax1 = axes[0, 0]
    for utility in simulation.utilities:
        col_name = f'{utility.agent_id}_total_output'
        if col_name in results.columns:
            utility_output = results[col_name].values
            market_share = utility_output / results['total_cleared'].values * 100
            ax1.plot(results['time_period'], market_share, label=utility.agent_id, linewidth=2)
    ax1.set_xlabel('Time Period (hours)')
    ax1.set_ylabel('Market Share (%)')
    ax1.set_title('Market Share Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Generation mix by utility
    ax2 = axes[0, 1]
    utility_mix = {}
    for utility in simulation.utilities:
        mix = {}
        for gen_type in ['coal', 'gas', 'wind', 'solar', 'hydro']:
            col_name = f'{utility.agent_id}_{gen_type}_output'
            if col_name in results.columns:
                mix[gen_type] = results[col_name].sum()
        utility_mix[utility.agent_id] = mix
        
    # Create stacked bar chart
    utilities = list(utility_mix.keys())
    gen_types = ['coal', 'gas', 'wind', 'solar', 'hydro']
    colors = ['#696969', '#FFA500', '#1E90FF', '#FFD700', '#00CED1']
    
    bottom = np.zeros(len(utilities))
    for i, gen_type in enumerate(gen_types):
        values = [utility_mix[u].get(gen_type, 0) for u in utilities]
        ax2.bar(utilities, values, bottom=bottom, label=gen_type, color=colors[i])
        bottom += values
        
    ax2.set_ylabel('Total Generation (MWh)')
    ax2.set_title('Generation Mix by Utility')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Profitability comparison
    ax3 = axes[1, 0]
    profits = []
    names = []
    for utility in simulation.utilities:
        metrics = utility.get_portfolio_metrics()
        profits.append(metrics['total_profit'])
        names.append(utility.agent_id)
        
    bars = ax3.bar(names, profits, alpha=0.7)
    # Color bars by profit/loss
    for bar, profit in zip(bars, profits):
        bar.set_color('green' if profit > 0 else 'red')
    ax3.set_ylabel('Total Profit ($)')
    ax3.set_title('Utility Profitability')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Average prices over time
    ax4 = axes[1, 1]
    ax4.plot(results['time_period'], results['clearing_price'], linewidth=2, color='blue')
    ax4.set_xlabel('Time Period (hours)')
    ax4.set_ylabel('Clearing Price ($/MWh)')
    ax4.set_title('Market Clearing Prices')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/utility_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("UTILITY PERFORMANCE SUMMARY")
    print("="*60)
    
    for utility in simulation.utilities:
        metrics = utility.get_portfolio_metrics()
        print(f"\n{utility.agent_id}:")
        print(f"  Total Capacity: {metrics['total_capacity']:.0f} MW")
        print(f"  Capacity Factor: {metrics['capacity_factor']*100:.1f}%")
        print(f"  Total Energy: {metrics['total_energy']:.0f} MWh")
        print(f"  Total Profit: ${metrics['total_profit']:,.0f}")
        if metrics['total_energy'] > 0:
            print(f"  Profit/MWh: ${metrics['total_profit']/metrics['total_energy']:.2f}")
        
    # Market concentration
    if 'hhi' in simulation.state.system_metrics:
        avg_hhi = np.mean(simulation.state.system_metrics['hhi'])
        print(f"\nMarket Concentration (HHI): {avg_hhi:.0f}")
        if avg_hhi < 1500:
            print("  Market is unconcentrated")
        elif avg_hhi < 2500:
            print("  Market is moderately concentrated")
        else:
            print("  Market is highly concentrated")

def main():
    """Run simulation with utility agents"""
    
    # Load configuration
    print("Loading configuration...")
    
    # You need to merge the two config files or load them properly
    try:
        # Try to load from single file first
        config = yaml.safe_load(open('config/utility_config.yaml', 'r'))
    except:
        # Otherwise load and merge
        main_config = yaml.safe_load(open('main_config.yaml', 'r'))
        utilities_config = yaml.safe_load(open('utilities_config.yaml', 'r'))
        config = {**main_config, **utilities_config}
    
    # Create and run extended simulation
    print("\nCreating simulation...")
    sim = ExtendedSimulation(config)
    
    print(f"\nRunning simulation for {config['simulation']['time_periods']} periods...")
    results = sim.run(config['simulation']['time_periods'])
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results.to_csv('results/utility_simulation_results.csv', index=False)
    print("Results saved to results/utility_simulation_results.csv")
    
    # Quick summary of results
    print("\n" + "="*60)
    print("QUICK RESULTS SUMMARY")
    print("="*60)
    print(f"Average clearing price: ${results['clearing_price'].mean():.2f}/MWh")
    print(f"Total energy cleared: {results['total_cleared'].sum():.0f} MWh")
    
    # Check if utilities are generating
    for utility in sim.utilities:
        col_name = f'{utility.agent_id}_total_output'
        if col_name in results.columns:
            total_gen = results[col_name].sum()
            avg_gen = results[col_name].mean()
            print(f"\n{utility.agent_id}:")
            print(f"  Total generation: {total_gen:.0f} MWh")
            print(f"  Average generation: {avg_gen:.1f} MW")
            print(f"  Max generation: {results[col_name].max():.1f} MW")
    
    # Analyze utility performance
    analyze_utility_performance(sim, results)
    
    # Print individual utility metrics
    print("\nDetailed Unit Metrics:")
    for utility in sim.utilities:
        print(f"\n{utility.agent_id} Portfolio Metrics:")
        metrics = utility.get_portfolio_metrics()
        for unit_id, unit_metrics in metrics['unit_metrics'].items():
            # Check if the unit generated anything
            total_gen = unit_metrics.get('total_generation', 0)
            if total_gen > 0 or unit_metrics.get('capacity_factor', 0) > 0:  # Show units that generated
                print(f"  {unit_id}:")
                print(f"    Type: {unit_metrics['type']}")
                print(f"    Capacity: {unit_metrics['capacity']} MW")
                if 'total_generation' in unit_metrics:
                    print(f"    Total Generation: {total_gen:.0f} MWh")
                print(f"    Capacity Factor: {unit_metrics['capacity_factor']*100:.1f}%")
                print(f"    Revenue: ${unit_metrics['revenue']:,.0f}")
                print(f"    Cost: ${unit_metrics['cost']:,.0f}")
                print(f"    Profit: ${unit_metrics['profit']:,.0f}")
                print(f"    Number of Starts: {unit_metrics['starts']}")

if __name__ == "__main__":
    main()