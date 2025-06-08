# Electricity Market Simulator - Core Implementation

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import yaml
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DATA STRUCTURES ====================

@dataclass
class Bid:
    """Represents a bid in the electricity market"""
    agent_id: str
    market_type: str
    quantity: float  # MW
    price: float     # $/MWh
    time_period: int
    bid_type: str    # 'supply' or 'demand'
    metadata: Dict = field(default_factory=dict)

@dataclass
class MarketResult:
    """Results from market clearing"""
    clearing_price: float
    cleared_quantities: Dict[str, float]
    total_cleared: float
    social_welfare: float
    time_period: int
    metadata: Dict = field(default_factory=dict)

@dataclass
class AgentState:
    """Tracks agent state across simulation"""
    agent_id: str
    capacity: float
    marginal_cost: float
    ramp_rate: float
    min_output: float
    current_output: float = 0.0
    profit: float = 0.0
    energy_produced: float = 0.0

# ==================== AGENTS ====================

class Agent(ABC):
    """Base class for all market participants"""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = None
        self.history = defaultdict(list)
    
    @abstractmethod
    def create_bid(self, market_state: Dict, time_period: int) -> List[Bid]:
        """Generate bids for the market"""
        pass
    
    def update_state(self, market_result: MarketResult, cleared_quantity: float):
        """Update agent state after market clearing"""
        if self.agent_id in market_result.cleared_quantities:
            if self.agent_type == 'consumer':
                # For consumers, cost is what they pay
                cost = cleared_quantity * market_result.clearing_price
                self.state.profit -= cost  # Negative because they pay
                self.state.current_output = cleared_quantity
                self.history['cost'].append(cost)
                self.history['consumption'].append(cleared_quantity)
                self.history['price'].append(market_result.clearing_price)
            elif self.agent_type in ['generator', 'storage']:
                # For generators, calculate profit
                revenue = cleared_quantity * market_result.clearing_price
                cost = cleared_quantity * self.state.marginal_cost if self.state else 0
                profit = revenue - cost
                
                if self.state:
                    self.state.profit += profit
                    self.state.current_output = cleared_quantity
                    self.state.energy_produced += cleared_quantity
                
                self.history['profit'].append(profit)
                self.history['output'].append(cleared_quantity)
                self.history['price'].append(market_result.clearing_price)

class Generator(Agent):
    """Represents a power generator"""
    
    def __init__(self, agent_id: str, capacity: float, marginal_cost: float, 
                 ramp_rate: float = None, min_output: float = 0.0):
        super().__init__(agent_id, 'generator')
        self.state = AgentState(
            agent_id=agent_id,
            capacity=capacity,
            marginal_cost=marginal_cost,
            ramp_rate=ramp_rate or capacity,
            min_output=min_output
        )
    
    def create_bid(self, market_state: Dict, time_period: int) -> List[Bid]:
        """Simple marginal cost bidding"""
        available_capacity = min(
            self.state.capacity,
            self.state.current_output + self.state.ramp_rate
        )
        
        return [Bid(
            agent_id=self.agent_id,
            market_type='day_ahead',
            quantity=available_capacity,
            price=self.state.marginal_cost,
            time_period=time_period,
            bid_type='supply'
        )]

class RenewableGenerator(Generator):
    """Renewable generator with variable output"""
    
    def __init__(self, agent_id: str, capacity: float, marginal_cost: float = 0.0,
                 availability_profile: List[float] = None):
        super().__init__(agent_id, capacity, marginal_cost)
        self.availability_profile = availability_profile or [1.0] * 24
        
    def create_bid(self, market_state: Dict, time_period: int) -> List[Bid]:
        """Bid available renewable capacity at low/zero marginal cost"""
        hour = time_period % len(self.availability_profile)
        available = self.state.capacity * self.availability_profile[hour]
        
        return [Bid(
            agent_id=self.agent_id,
            market_type='day_ahead',
            quantity=available,
            price=self.state.marginal_cost,
            time_period=time_period,
            bid_type='supply'
        )]

class Consumer(Agent):
    """Represents electricity demand"""
    
    def __init__(self, agent_id: str, demand_profile: List[float], 
                 price_elasticity: float = -0.1):
        super().__init__(agent_id, 'consumer')
        self.demand_profile = demand_profile
        self.price_elasticity = price_elasticity
        self.base_price = 50.0  # Reference price for elasticity
        # Initialize state for consumers
        self.state = AgentState(
            agent_id=agent_id,
            capacity=max(demand_profile),  # Max demand as capacity
            marginal_cost=0.0,  # Consumers don't have marginal cost
            ramp_rate=float('inf'),
            min_output=0.0
        )
        
    def create_bid(self, market_state: Dict, time_period: int) -> List[Bid]:
        """Create demand bid with optional price elasticity"""
        hour = time_period % len(self.demand_profile)
        base_demand = self.demand_profile[hour]
        
        # Simple price-elastic demand
        if 'last_price' in market_state:
            price_ratio = market_state['last_price'] / self.base_price
            demand_adjustment = 1 + self.price_elasticity * (price_ratio - 1)
            demand = base_demand * max(0.5, min(1.5, demand_adjustment))
        else:
            demand = base_demand
            
        return [Bid(
            agent_id=self.agent_id,
            market_type='day_ahead',
            quantity=demand,
            price=1000.0,  # High willingness to pay
            time_period=time_period,
            bid_type='demand'
        )]

class StorageUnit(Agent):
    """Energy storage system"""
    
    def __init__(self, agent_id: str, capacity: float, power_rating: float,
                 efficiency: float = 0.9, initial_soc: float = 0.5):
        super().__init__(agent_id, 'storage')
        self.capacity = capacity
        self.power_rating = power_rating
        self.efficiency = efficiency
        self.soc = initial_soc * capacity  # State of charge
        self.min_soc = 0.1 * capacity
        self.max_soc = 0.9 * capacity
        # Initialize state for storage
        self.state = AgentState(
            agent_id=agent_id,
            capacity=power_rating,
            marginal_cost=0.0,  # Storage has zero marginal cost
            ramp_rate=power_rating,
            min_output=-power_rating  # Can be negative (charging)
        )
        
    def update_state(self, market_result: MarketResult, cleared_quantity: float):
        """Update storage state including state of charge"""
        # Handle charge/discharge bids
        charge_id = f"{self.agent_id}_charge"
        discharge_id = f"{self.agent_id}_discharge"
        
        if charge_id in market_result.cleared_quantities:
            # Charging
            charge_amount = market_result.cleared_quantities[charge_id]
            self.soc += charge_amount * self.efficiency
            self.soc = min(self.soc, self.max_soc)
            cost = charge_amount * market_result.clearing_price
            self.state.profit -= cost
            self.history['soc'].append(self.soc)
            self.history['action'].append('charge')
            
        elif discharge_id in market_result.cleared_quantities:
            # Discharging
            discharge_amount = market_result.cleared_quantities[discharge_id]
            self.soc -= discharge_amount / self.efficiency
            self.soc = max(self.soc, self.min_soc)
            revenue = discharge_amount * market_result.clearing_price
            self.state.profit += revenue
            self.history['soc'].append(self.soc)
            self.history['action'].append('discharge')
        else:
            # No action
            self.history['soc'].append(self.soc)
            self.history['action'].append('idle')
        
    def create_bid(self, market_state: Dict, time_period: int) -> List[Bid]:
        """Create charge/discharge bids based on price expectations"""
        bids = []
        
        # Simple strategy: charge when price low, discharge when high
        if 'price_forecast' in market_state:
            current_price = market_state['price_forecast'][time_period % 24]
            avg_price = np.mean(market_state['price_forecast'])
            
            if current_price < 0.9 * avg_price and self.soc < self.max_soc:
                # Charge bid
                charge_capacity = min(self.power_rating, (self.max_soc - self.soc))
                bids.append(Bid(
                    agent_id=f"{self.agent_id}_charge",
                    market_type='day_ahead',
                    quantity=charge_capacity,
                    price=current_price * 1.1,
                    time_period=time_period,
                    bid_type='demand'
                ))
            elif current_price > 1.1 * avg_price and self.soc > self.min_soc:
                # Discharge bid
                discharge_capacity = min(self.power_rating, (self.soc - self.min_soc))
                bids.append(Bid(
                    agent_id=f"{self.agent_id}_discharge",
                    market_type='day_ahead',
                    quantity=discharge_capacity,
                    price=current_price * 0.9,
                    time_period=time_period,
                    bid_type='supply'
                ))
                
        return bids

# ==================== MARKET CLEARING ====================

class MarketClearing:
    """Handles market clearing and pricing"""
    
    def __init__(self, pricing_method: str = 'uniform'):
        self.pricing_method = pricing_method
        
    def clear_market(self, supply_bids: List[Bid], demand_bids: List[Bid],
                    time_period: int) -> MarketResult:
        """Clear market using merit order dispatch"""
        
        # Sort bids
        supply_bids = sorted(supply_bids, key=lambda x: x.price)
        demand_bids = sorted(demand_bids, key=lambda x: x.price, reverse=True)
        
        # Build supply and demand curves
        supply_curve = self._build_curve(supply_bids)
        demand_curve = self._build_curve(demand_bids)
        
        # Find intersection
        clearing_price, total_cleared = self._find_intersection(supply_curve, demand_curve)
        
        # Determine cleared quantities
        cleared_quantities = {}
        remaining = total_cleared
        
        for bid in supply_bids:
            if bid.price <= clearing_price and remaining > 0:
                cleared = min(bid.quantity, remaining)
                cleared_quantities[bid.agent_id] = cleared
                remaining -= cleared
                
        # Also track demand clearing
        remaining_demand = total_cleared
        for bid in demand_bids:
            if bid.price >= clearing_price and remaining_demand > 0:
                cleared = min(bid.quantity, remaining_demand)
                if bid.agent_id not in cleared_quantities:
                    cleared_quantities[bid.agent_id] = cleared
                remaining_demand -= cleared
                
        # Calculate social welfare
        social_welfare = self._calculate_welfare(supply_bids, demand_bids, 
                                               cleared_quantities, clearing_price)
        
        return MarketResult(
            clearing_price=clearing_price,
            cleared_quantities=cleared_quantities,
            total_cleared=total_cleared,
            social_welfare=social_welfare,
            time_period=time_period
        )
    
    def _build_curve(self, bids: List[Bid]) -> List[Tuple[float, float]]:
        """Build price-quantity curve from bids"""
        curve = [(0, 0)]
        cumulative = 0
        
        for bid in bids:
            cumulative += bid.quantity
            curve.append((cumulative, bid.price))
            
        return curve
    
    def _find_intersection(self, supply: List[Tuple[float, float]], 
                          demand: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Find market clearing price and quantity"""
        # Get total demand from the last point in demand curve
        if not demand or len(demand) < 2:
            return 0.0, 0.0
            
        total_demand = demand[-1][0]  # Total quantity is the last quantity in demand curve
        
        # Find where supply meets demand
        for i in range(len(supply) - 1):
            q1, p1 = supply[i]
            q2, p2 = supply[i + 1]
            
            if q1 <= total_demand <= q2:
                # Interpolate price at intersection
                if q2 > q1:
                    clearing_price = p1 + (p2 - p1) * (total_demand - q1) / (q2 - q1)
                else:
                    clearing_price = p2
                return clearing_price, total_demand
                
        # If supply insufficient, use highest supply price
        return supply[-1][1], min(supply[-1][0], total_demand)
    
    def _calculate_welfare(self, supply_bids, demand_bids, cleared_quantities, price):
        """Calculate social welfare"""
        producer_surplus = sum(
            cleared_quantities.get(bid.agent_id, 0) * (price - bid.price)
            for bid in supply_bids
        )
        # Simplified consumer surplus
        consumer_surplus = sum(
            bid.quantity * (bid.price - price)
            for bid in demand_bids
            if bid.price >= price
        )
        return producer_surplus + consumer_surplus

# ==================== MARKET TYPES ====================

class Market(ABC):
    """Base class for different market types"""
    
    def __init__(self, market_type: str):
        self.market_type = market_type
        self.clearing_engine = MarketClearing()
        self.results_history = []
        
    @abstractmethod
    def run_auction(self, agents: List[Agent], time_period: int, 
                   market_state: Dict) -> MarketResult:
        """Run market auction"""
        pass

class DayAheadMarket(Market):
    """Day-ahead energy market"""
    
    def __init__(self):
        super().__init__('day_ahead')
        
    def run_auction(self, agents: List[Agent], time_period: int,
                   market_state: Dict) -> MarketResult:
        """Run day-ahead market auction"""
        
        # Collect bids
        all_bids = []
        for agent in agents:
            all_bids.extend(agent.create_bid(market_state, time_period))
            
        # Separate supply and demand
        supply_bids = [bid for bid in all_bids if bid.bid_type == 'supply']
        demand_bids = [bid for bid in all_bids if bid.bid_type == 'demand']
        
        # Clear market
        result = self.clearing_engine.clear_market(supply_bids, demand_bids, time_period)
        self.results_history.append(result)
        
        # Update agents
        for agent in agents:
            if isinstance(agent, StorageUnit):
                # Storage units have special update logic
                agent.update_state(result, 0)
            elif agent.agent_id in result.cleared_quantities:
                agent.update_state(result, result.cleared_quantities[agent.agent_id])
                
        return result

class RealTimeMarket(Market):
    """Real-time balancing market"""
    
    def __init__(self):
        super().__init__('real_time')
        
    def run_auction(self, agents: List[Agent], time_period: int,
                   market_state: Dict) -> MarketResult:
        """Run real-time market with deviations from day-ahead"""
        # Simplified implementation
        return super().run_auction(agents, time_period, market_state)

# ==================== POLICIES ====================

class Policy(ABC):
    """Base class for market policies"""
    
    @abstractmethod
    def apply(self, market_result: MarketResult, agents: List[Agent]) -> None:
        """Apply policy to market results"""
        pass

class CarbonTax(Policy):
    """Carbon tax policy"""
    
    def __init__(self, tax_rate: float):
        self.tax_rate = tax_rate  # $/tCO2
        self.emission_factors = {
            'coal': 0.95,      # tCO2/MWh
            'gas': 0.45,       # tCO2/MWh
            'renewable': 0.0,   # tCO2/MWh
        }
        
    def apply(self, market_result: MarketResult, agents: List[Agent]) -> None:
        """Apply carbon tax to generators"""
        for agent in agents:
            if isinstance(agent, Generator) and agent.agent_id in market_result.cleared_quantities:
                # Determine fuel type (simplified)
                if agent.state.marginal_cost > 50:
                    fuel_type = 'coal'
                elif agent.state.marginal_cost > 20:
                    fuel_type = 'gas'
                else:
                    fuel_type = 'renewable'
                    
                emissions = (market_result.cleared_quantities[agent.agent_id] * 
                           self.emission_factors[fuel_type])
                carbon_cost = emissions * self.tax_rate
                
                # Adjust agent profit
                agent.state.profit -= carbon_cost
                agent.history['carbon_cost'].append(carbon_cost)

class RenewableMandate(Policy):
    """Renewable portfolio standard"""
    
    def __init__(self, target_percentage: float):
        self.target_percentage = target_percentage
        
    def apply(self, market_result: MarketResult, agents: List[Agent]) -> None:
        """Track renewable generation percentage"""
        total_generation = sum(market_result.cleared_quantities.values())
        renewable_generation = sum(
            market_result.cleared_quantities.get(agent.agent_id, 0)
            for agent in agents
            if isinstance(agent, RenewableGenerator)
        )
        
        renewable_percentage = renewable_generation / total_generation if total_generation > 0 else 0
        market_result.metadata['renewable_percentage'] = renewable_percentage
        
        # Could implement penalties/incentives here

# ==================== SIMULATION ENGINE ====================

class SimulationState:
    """Tracks overall simulation state"""
    
    def __init__(self):
        self.time_period = 0
        self.market_results = defaultdict(list)
        self.system_metrics = defaultdict(list)
        
    def update(self, market_type: str, result: MarketResult):
        """Update simulation state with market results"""
        self.market_results[market_type].append(result)
        
        # Track system metrics
        self.system_metrics['lmp'].append(result.clearing_price)
        self.system_metrics['total_generation'].append(result.total_cleared)
        self.system_metrics['social_welfare'].append(result.social_welfare)

class Simulation:
    """Main simulation controller"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.agents = []
        self.markets = {}
        self.policies = []
        self.state = SimulationState()
        
        self._initialize_simulation()
        
    def _initialize_simulation(self):
        """Initialize agents, markets, and policies from config"""
        
        # Create agents
        for gen_config in self.config.get('generators', []):
            if gen_config['type'] == 'thermal':
                agent = Generator(**gen_config['params'])
            elif gen_config['type'] == 'renewable':
                agent = RenewableGenerator(**gen_config['params'])
            self.agents.append(agent)
            
        for consumer_config in self.config.get('consumers', []):
            agent = Consumer(**consumer_config['params'])
            self.agents.append(agent)
            
        for storage_config in self.config.get('storage', []):
            agent = StorageUnit(**storage_config['params'])
            self.agents.append(agent)
            
        # Create markets
        for market_type in self.config.get('markets', ['day_ahead']):
            if market_type == 'day_ahead':
                self.markets[market_type] = DayAheadMarket()
            elif market_type == 'real_time':
                self.markets[market_type] = RealTimeMarket()
                
        # Create policies
        for policy_config in self.config.get('policies', []):
            if policy_config['type'] == 'carbon_tax':
                self.policies.append(CarbonTax(policy_config['tax_rate']))
            elif policy_config['type'] == 'renewable_mandate':
                self.policies.append(RenewableMandate(policy_config['target']))
                
    def run(self, time_periods: int):
        """Run simulation for specified time periods"""
        logger.info(f"Starting simulation for {time_periods} periods")
        
        market_state = {
            'last_price': 50.0,
            'price_forecast': [40 + 20*np.sin(2*np.pi*h/24) for h in range(24)]
        }
        
        for t in range(time_periods):
            self.state.time_period = t
            
            # Run each market
            for market_name, market in self.markets.items():
                result = market.run_auction(self.agents, t, market_state)
                self.state.update(market_name, result)
                
                # Apply policies
                for policy in self.policies:
                    policy.apply(result, self.agents)
                    
                # Update market state
                market_state['last_price'] = result.clearing_price
                
            if t % 24 == 0:
                logger.info(f"Completed day {t//24 + 1}")
                
        logger.info("Simulation completed")
        return self.get_results()
        
    def get_results(self) -> pd.DataFrame:
        """Compile simulation results into DataFrame"""
        results = []
        
        for market_type, market_results in self.state.market_results.items():
            for result in market_results:
                results.append({
                    'time_period': result.time_period,
                    'market_type': market_type,
                    'clearing_price': result.clearing_price,
                    'total_cleared': result.total_cleared,
                    'social_welfare': result.social_welfare,
                    **result.metadata
                })
                
        return pd.DataFrame(results)

# ==================== CONFIGURATION LOADER ====================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_default_config() -> Dict:
    """Create default configuration"""
    return {
        'simulation': {
            'time_periods': 168,  # 1 week
            'time_resolution': 'hourly'
        },
        'generators': [
            {
                'type': 'thermal',
                'params': {
                    'agent_id': 'coal_1',
                    'capacity': 500,
                    'marginal_cost': 35,
                    'ramp_rate': 100,
                    'min_output': 150
                }
            },
            {
                'type': 'thermal',
                'params': {
                    'agent_id': 'gas_1',
                    'capacity': 300,
                    'marginal_cost': 45,
                    'ramp_rate': 200,
                    'min_output': 50
                }
            },
            {
                'type': 'renewable',
                'params': {
                    'agent_id': 'wind_1',
                    'capacity': 200,
                    'marginal_cost': 0,
                    'availability_profile': [0.3 + 0.2*np.sin(2*np.pi*h/24) 
                                           for h in range(24)]
                }
            },
            {
                'type': 'renewable',
                'params': {
                    'agent_id': 'solar_1',
                    'capacity': 150,
                    'marginal_cost': 0,
                    'availability_profile': [max(0, np.sin(np.pi*(h-6)/12)) 
                                           for h in range(24)]
                }
            }
        ],
        'consumers': [
            {
                'params': {
                    'agent_id': 'load_1',
                    'demand_profile': [600 + 200*np.sin(2*np.pi*(h-14)/24) 
                                     for h in range(24)],
                    'price_elasticity': -0.1
                }
            }
        ],
        'storage': [
            {
                'params': {
                    'agent_id': 'battery_1',
                    'capacity': 100,
                    'power_rating': 50,
                    'efficiency': 0.9,
                    'initial_soc': 0.5
                }
            }
        ],
        'markets': ['day_ahead'],
        'policies': [
            {
                'type': 'carbon_tax',
                'tax_rate': 50  # $/tCO2
            },
            {
                'type': 'renewable_mandate',
                'target': 0.3  # 30% renewable target
            }
        ]
    }

# ==================== MAIN RUNNER ====================

def main(config_path: Optional[str] = None):
    """Main simulation runner"""
    
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        config = create_default_config()
        
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
        
    # Create and run simulation
    sim = Simulation(config)
    results = sim.run(config['simulation']['time_periods'])
    
    # Save results
    results.to_csv('results/simulation_results.csv', index=False)
    logger.info(f"Results saved to results/simulation_results.csv")
    
    # Print summary statistics
    print("\n=== Simulation Summary ===")
    print(f"Average LMP: ${results['clearing_price'].mean():.2f}/MWh")
    print(f"Total Generation: {results['total_cleared'].sum():.0f} MWh")
    print(f"Total Social Welfare: ${results['social_welfare'].sum():,.0f}")
    
    # Agent summaries
    print("\n=== Agent Performance ===")
    for agent in sim.agents:
        if hasattr(agent, 'state') and hasattr(agent.state, 'profit'):
            print(f"{agent.agent_id}: Profit=${agent.state.profit:,.0f}, "
                  f"Energy={agent.state.energy_produced:.0f} MWh")

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)