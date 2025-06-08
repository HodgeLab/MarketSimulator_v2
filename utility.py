# Utility Agent Extension for Electricity Market Simulator
# Represents utilities with multiple generators and unified bidding strategies

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
from collections import defaultdict
import logging

# Import base classes from your existing simulator
from main import Agent, Generator, Bid, MarketResult, AgentState

logger = logging.getLogger(__name__)

# ==================== PORTFOLIO DATA STRUCTURES ====================

@dataclass
class GeneratorUnit:
    """Represents a single generator within a utility's portfolio"""
    unit_id: str
    unit_type: str  # 'coal', 'gas', 'nuclear', 'wind', 'solar', 'hydro'
    capacity: float  # MW
    marginal_cost: float  # $/MWh
    min_output: float  # MW
    ramp_rate: float  # MW/period
    startup_cost: float  # $
    min_downtime: int  # periods
    min_uptime: int  # periods
    efficiency: float  # 0-1
    fuel_type: str
    location: Optional[str] = None
    current_output: float = 0.0
    is_online: bool = False
    hours_online: int = 0
    hours_offline: int = 0
    availability: float = 1.0  # For renewable/maintenance

@dataclass
class UtilityPortfolio:
    """Portfolio of generators owned by a utility"""
    utility_id: str
    generators: List[GeneratorUnit]
    total_capacity: float = field(init=False)
    
    def __post_init__(self):
        self.total_capacity = sum(g.capacity for g in self.generators)
        self.generators_by_type = defaultdict(list)
        for gen in self.generators:
            self.generators_by_type[gen.unit_type].append(gen)

@dataclass
class BiddingDecision:
    """Represents a bidding decision for a generator"""
    unit_id: str
    quantity: float
    price: float
    priority: int  # For internal dispatch order
    must_run: bool = False

# ==================== BIDDING STRATEGIES ====================

class BiddingStrategy(ABC):
    """Abstract base class for utility bidding strategies"""
    
    @abstractmethod
    def create_bids(self, portfolio: UtilityPortfolio, 
                   market_state: Dict, time_period: int) -> List[BiddingDecision]:
        """Generate bidding decisions for all units in portfolio"""
        pass

class MarginalCostBidding(BiddingStrategy):
    """Simple strategy: bid each unit at its marginal cost"""
    
    def create_bids(self, portfolio: UtilityPortfolio, 
                   market_state: Dict, time_period: int) -> List[BiddingDecision]:
        decisions = []
        
        for gen in portfolio.generators:
            if gen.availability > 0:
                decisions.append(BiddingDecision(
                    unit_id=gen.unit_id,
                    quantity=gen.capacity * gen.availability,
                    price=gen.marginal_cost,
                    priority=1
                ))
                
        return decisions

class StrategicBidding(BiddingStrategy):
    """Strategic bidding with markup based on market conditions"""
    
    def __init__(self, base_markup: float = 0.1, 
                 scarcity_threshold: float = 0.8):
        self.base_markup = base_markup
        self.scarcity_threshold = scarcity_threshold
        
    def create_bids(self, portfolio: UtilityPortfolio, 
                   market_state: Dict, time_period: int) -> List[BiddingDecision]:
        decisions = []
        
        # Estimate supply scarcity
        if 'supply_ratio' in market_state:
            scarcity = 1 - market_state['supply_ratio']
            markup = self.base_markup * (1 + scarcity * 2)
        else:
            markup = self.base_markup
            
        for gen in portfolio.generators:
            if gen.availability > 0:
                # Apply different markups based on unit type
                if gen.unit_type in ['coal', 'gas']:
                    unit_markup = markup
                elif gen.unit_type == 'nuclear':
                    unit_markup = 0  # Nuclear bids at marginal cost
                else:  # Renewables
                    unit_markup = -0.01  # Slight negative to ensure dispatch
                    
                bid_price = gen.marginal_cost * (1 + unit_markup)
                
                decisions.append(BiddingDecision(
                    unit_id=gen.unit_id,
                    quantity=gen.capacity * gen.availability,
                    price=bid_price,
                    priority=self._get_priority(gen)
                ))
                
        return decisions
    
    def _get_priority(self, gen: GeneratorUnit) -> int:
        """Assign dispatch priority based on unit characteristics"""
        priorities = {
            'nuclear': 1,
            'wind': 2,
            'solar': 2,
            'hydro': 3,
            'gas': 4,
            'coal': 5
        }
        return priorities.get(gen.unit_type, 6)

class OptimalBidding(BiddingStrategy):
    """Optimization-based bidding considering unit commitment and portfolio effects"""
    
    def __init__(self, forecast_horizon: int = 24, 
                 risk_aversion: float = 0.5):
        self.forecast_horizon = forecast_horizon
        self.risk_aversion = risk_aversion
        
    def create_bids(self, portfolio: UtilityPortfolio, 
                   market_state: Dict, time_period: int) -> List[BiddingDecision]:
        decisions = []
        
        # Get price forecast
        if 'price_forecast' in market_state:
            expected_prices = market_state['price_forecast']
        else:
            expected_prices = [50] * self.forecast_horizon
            
        # Solve unit commitment problem (simplified)
        commitment_decisions = self._solve_unit_commitment(
            portfolio, expected_prices, time_period
        )
        
        # Create bids based on commitment decisions
        for gen in portfolio.generators:
            if gen.unit_id in commitment_decisions:
                commit_decision = commitment_decisions[gen.unit_id]
                
                if commit_decision['committed']:
                    # Strategic pricing based on expected market conditions
                    bid_price = self._calculate_strategic_price(
                        gen, expected_prices[time_period % 24], portfolio
                    )
                    
                    decisions.append(BiddingDecision(
                        unit_id=gen.unit_id,
                        quantity=commit_decision['quantity'],
                        price=bid_price,
                        priority=commit_decision['priority'],
                        must_run=commit_decision.get('must_run', False)
                    ))
                    
        return decisions
    
    def _solve_unit_commitment(self, portfolio: UtilityPortfolio, 
                              prices: List[float], current_period: int) -> Dict:
        """Simplified unit commitment (full version would use MILP)"""
        commitment = {}
        
        for gen in portfolio.generators:
            # Check constraints
            can_start = gen.hours_offline >= gen.min_downtime
            can_stop = gen.hours_online >= gen.min_uptime
            
            # Simple decision rule based on expected profit
            expected_profit = (prices[current_period % 24] - gen.marginal_cost) * gen.capacity
            
            if gen.is_online:
                # Check if should stay online
                if expected_profit < -gen.startup_cost and can_stop:
                    committed = False
                else:
                    committed = True
            else:
                # Check if should start
                if expected_profit > gen.startup_cost and can_start:
                    committed = True
                else:
                    committed = False
                    
            commitment[gen.unit_id] = {
                'committed': committed,
                'quantity': gen.capacity * gen.availability if committed else 0,
                'priority': self._get_commitment_priority(gen),
                'must_run': gen.unit_type == 'nuclear' and gen.is_online
            }
            
        return commitment
    
    def _calculate_strategic_price(self, gen: GeneratorUnit, 
                                 expected_price: float, 
                                 portfolio: UtilityPortfolio) -> float:
        """Calculate strategic bid price considering market power"""
        # Base price is marginal cost
        base_price = gen.marginal_cost
        
        # Add opportunity cost for flexible units
        if gen.unit_type in ['gas', 'hydro']:
            opportunity_cost = max(0, expected_price - base_price) * 0.1
            base_price += opportunity_cost
            
        # Consider portfolio position
        market_share = portfolio.total_capacity / 10000  # Assume 10GW market
        market_power_markup = base_price * market_share * 0.5
        
        return base_price + market_power_markup
    
    def _get_commitment_priority(self, gen: GeneratorUnit) -> int:
        """Priority for unit commitment"""
        if gen.unit_type == 'nuclear':
            return 1
        elif gen.unit_type in ['wind', 'solar']:
            return 2
        elif gen.min_output / gen.capacity > 0.5:  # Inflexible units
            return 3
        else:
            return 4

# ==================== UTILITY AGENT ====================

class UtilityAgent(Agent):
    """Represents a utility company with multiple generators and unified strategy"""
    
    def __init__(self, utility_id: str, portfolio: UtilityPortfolio,
                 bidding_strategy: BiddingStrategy = None,
                 enable_bilateral_contracts: bool = False):
        super().__init__(utility_id, 'utility')
        self.portfolio = portfolio
        self.bidding_strategy = bidding_strategy or MarginalCostBidding()
        self.enable_bilateral_contracts = enable_bilateral_contracts
        
        # Initialize state
        self.state = AgentState(
            agent_id=utility_id,
            capacity=portfolio.total_capacity,
            marginal_cost=self._calculate_weighted_marginal_cost(),
            ramp_rate=sum(g.ramp_rate for g in portfolio.generators),
            min_output=sum(g.min_output for g in portfolio.generators)
        )
        
        # Track unit-level performance
        self.unit_states = {
            gen.unit_id: {
                'output': 0.0,
                'revenue': 0.0,
                'cost': 0.0,
                'starts': 0,
                'hours_run': 0
            } for gen in portfolio.generators
        }
        
        # Bilateral contracts
        self.contracts = []
        
    def _calculate_weighted_marginal_cost(self) -> float:
        """Calculate capacity-weighted average marginal cost"""
        total_cost = sum(g.capacity * g.marginal_cost for g in self.portfolio.generators)
        return total_cost / self.portfolio.total_capacity
        
    def create_bid(self, market_state: Dict, time_period: int) -> List[Bid]:
        """Create bids for all generators using unified strategy"""
        # Get bidding decisions from strategy
        decisions = self.bidding_strategy.create_bids(
            self.portfolio, market_state, time_period
        )
        
        # Convert decisions to market bids
        bids = []
        for decision in decisions:
            if decision.quantity > 0:
                # Use consistent bid ID format: utility_id_unit_id
                bid_id = f"{self.agent_id}_{decision.unit_id}"
                
                bids.append(Bid(
                    agent_id=bid_id,
                    market_type='day_ahead',
                    quantity=decision.quantity,
                    price=decision.price,
                    time_period=time_period,
                    bid_type='supply',
                    metadata={
                        'unit_id': decision.unit_id,
                        'priority': decision.priority,
                        'must_run': decision.must_run,
                        'utility_id': self.agent_id
                    }
                ))
                
        # Add bilateral contract positions if enabled
        if self.enable_bilateral_contracts:
            contract_bids = self._create_contract_bids(time_period)
            bids.extend(contract_bids)
            
        # Log bid creation for debugging
        if time_period < 3:  # Log first few periods
            print(f"\n{self.agent_id} created {len(bids)} bids for period {time_period}:")
            for bid in bids[:5]:  # Show first 5 bids
                print(f"  {bid.agent_id}: {bid.quantity:.1f} MW @ ${bid.price:.2f}/MWh")
            
        return bids
    
    def _create_contract_bids(self, time_period: int) -> List[Bid]:
        """Create bids to cover bilateral contract positions"""
        contract_bids = []
        
        for contract in self.contracts:
            if contract['start_period'] <= time_period < contract['end_period']:
                # Negative price to ensure dispatch for contract coverage
                contract_bids.append(Bid(
                    agent_id=f"{self.agent_id}_contract_{contract['id']}",
                    market_type='day_ahead',
                    quantity=contract['quantity'],
                    price=-10.0,  # Negative to ensure dispatch
                    time_period=time_period,
                    bid_type='supply',
                    metadata={'contract_id': contract['id']}
                ))
                
        return contract_bids
    
    def update_state(self, market_result: MarketResult, cleared_quantity: float):
        """Update utility and unit states after market clearing"""
        # Update overall utility state
        total_cleared = 0
        total_revenue = 0
        total_cost = 0
        
        # Update individual unit states
        for gen in self.portfolio.generators:
            unit_bid_id = f"{self.agent_id}_{gen.unit_id}"
            
            if unit_bid_id in market_result.cleared_quantities:
                unit_cleared = market_result.cleared_quantities[unit_bid_id]
                unit_revenue = unit_cleared * market_result.clearing_price
                unit_cost = unit_cleared * gen.marginal_cost
                
                # Update generator state
                gen.current_output = unit_cleared
                if unit_cleared > 0 and not gen.is_online:
                    gen.is_online = True
                    gen.hours_online = 0
                    gen.hours_offline = 0
                    self.unit_states[gen.unit_id]['starts'] += 1
                    total_cost += gen.startup_cost
                elif unit_cleared == 0 and gen.is_online:
                    gen.is_online = False
                    gen.hours_online = 0
                    gen.hours_offline = 0
                    
                # Update time counters
                if gen.is_online:
                    gen.hours_online += 1
                else:
                    gen.hours_offline += 1
                    
                # Track unit performance
                self.unit_states[gen.unit_id]['output'] = unit_cleared
                self.unit_states[gen.unit_id]['revenue'] += unit_revenue
                self.unit_states[gen.unit_id]['cost'] += unit_cost
                self.unit_states[gen.unit_id]['hours_run'] += 1 if unit_cleared > 0 else 0
                
                total_cleared += unit_cleared
                total_revenue += unit_revenue
                total_cost += unit_cost
                
        # Update utility state
        self.state.current_output = total_cleared
        self.state.profit += (total_revenue - total_cost)
        self.state.energy_produced += total_cleared
        
        # Record history
        self.history['total_output'].append(total_cleared)
        self.history['total_revenue'].append(total_revenue)
        self.history['total_cost'].append(total_cost)
        self.history['profit'].append(total_revenue - total_cost)
        self.history['clearing_price'].append(market_result.clearing_price)
        
        # Record unit-level history
        for gen in self.portfolio.generators:
            unit_id = gen.unit_id
            self.history[f'unit_{unit_id}_output'].append(
                self.unit_states[unit_id]['output']
            )
            
    def add_bilateral_contract(self, contract_id: str, quantity: float,
                             start_period: int, end_period: int, 
                             strike_price: float):
        """Add a bilateral contract to the utility's portfolio"""
        self.contracts.append({
            'id': contract_id,
            'quantity': quantity,
            'start_period': start_period,
            'end_period': end_period,
            'strike_price': strike_price
        })
        
    def get_portfolio_metrics(self) -> Dict:
        """Calculate portfolio-level performance metrics"""
        metrics = {
            'total_capacity': self.portfolio.total_capacity,
            'available_capacity': sum(g.capacity * g.availability 
                                    for g in self.portfolio.generators),
            'online_capacity': sum(g.capacity for g in self.portfolio.generators 
                                 if g.is_online),
            'capacity_factor': self.state.energy_produced / (
                self.portfolio.total_capacity * len(self.history['total_output'])
            ) if self.history['total_output'] else 0,
            'average_marginal_cost': self._calculate_weighted_marginal_cost(),
            'total_profit': self.state.profit,
            'total_energy': self.state.energy_produced,
            'unit_metrics': {}
        }
        
        # Add unit-level metrics
        for gen in self.portfolio.generators:
            unit_id = gen.unit_id
            unit_data = self.unit_states[unit_id]
            
            metrics['unit_metrics'][unit_id] = {
                'type': gen.unit_type,
                'capacity': gen.capacity,
                'capacity_factor': unit_data['hours_run'] / len(self.history['total_output'])
                                 if self.history['total_output'] else 0,
                'revenue': unit_data['revenue'],
                'cost': unit_data['cost'],
                'profit': unit_data['revenue'] - unit_data['cost'],
                'starts': unit_data['starts'],
                'marginal_cost': gen.marginal_cost
            }
            
        return metrics

# ==================== MARKET POWER ANALYSIS ====================

class MarketPowerAnalyzer:
    """Analyze market power and strategic behavior of utilities"""
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_hhi(self, utilities: List[UtilityAgent], 
                     market_results: List[MarketResult]) -> float:
        """Calculate Herfindahl-Hirschman Index"""
        total_generation = defaultdict(float)
        
        for result in market_results:
            for agent_id, quantity in result.cleared_quantities.items():
                # Extract utility ID from bid ID
                utility_id = agent_id.split('_')[0]
                total_generation[utility_id] += quantity
                
        total_market = sum(total_generation.values())
        market_shares = [gen / total_market for gen in total_generation.values()]
        
        hhi = sum(share**2 for share in market_shares) * 10000
        return hhi
        
    def calculate_lerner_index(self, utility: UtilityAgent, 
                              market_price: float) -> float:
        """Calculate Lerner Index (markup over marginal cost)"""
        marginal_cost = utility._calculate_weighted_marginal_cost()
        return (market_price - marginal_cost) / market_price
        
    def analyze_pivotal_supplier(self, utilities: List[UtilityAgent],
                               demand: float) -> Dict[str, bool]:
        """Identify pivotal suppliers (needed to meet demand)"""
        total_capacity = sum(u.portfolio.total_capacity for u in utilities)
        pivotal = {}
        
        for utility in utilities:
            remaining_capacity = total_capacity - utility.portfolio.total_capacity
            pivotal[utility.agent_id] = remaining_capacity < demand
            
        return pivotal

# ==================== EXAMPLE USAGE ====================

def create_example_utility():
    """Create an example utility with diverse portfolio"""
    
    # Create generator units
    generators = [
        GeneratorUnit(
            unit_id="coal_1",
            unit_type="coal",
            capacity=500,
            marginal_cost=30,
            min_output=200,
            ramp_rate=50,
            startup_cost=10000,
            min_downtime=4,
            min_uptime=4,
            efficiency=0.35,
            fuel_type="coal"
        ),
        GeneratorUnit(
            unit_id="coal_2",
            unit_type="coal",
            capacity=400,
            marginal_cost=32,
            min_output=150,
            ramp_rate=40,
            startup_cost=8000,
            min_downtime=4,
            min_uptime=4,
            efficiency=0.34,
            fuel_type="coal"
        ),
        GeneratorUnit(
            unit_id="gas_ccgt_1",
            unit_type="gas",
            capacity=300,
            marginal_cost=45,
            min_output=100,
            ramp_rate=150,
            startup_cost=5000,
            min_downtime=2,
            min_uptime=2,
            efficiency=0.55,
            fuel_type="natural_gas"
        ),
        GeneratorUnit(
            unit_id="gas_peaker_1",
            unit_type="gas",
            capacity=150,
            marginal_cost=80,
            min_output=30,
            ramp_rate=150,
            startup_cost=2000,
            min_downtime=1,
            min_uptime=1,
            efficiency=0.35,
            fuel_type="natural_gas"
        ),
        GeneratorUnit(
            unit_id="wind_farm_1",
            unit_type="wind",
            capacity=200,
            marginal_cost=0,
            min_output=0,
            ramp_rate=200,
            startup_cost=0,
            min_downtime=0,
            min_uptime=0,
            efficiency=1.0,
            fuel_type="wind",
            availability=0.3  # 30% capacity factor
        ),
        GeneratorUnit(
            unit_id="solar_farm_1",
            unit_type="solar",
            capacity=100,
            marginal_cost=0,
            min_output=0,
            ramp_rate=100,
            startup_cost=0,
            min_downtime=0,
            min_uptime=0,
            efficiency=1.0,
            fuel_type="solar",
            availability=0.0  # Will vary by hour
        )
    ]
    
    # Create portfolio
    portfolio = UtilityPortfolio(
        utility_id="utility_1",
        generators=generators
    )
    
    # Create utility with strategic bidding
    strategy = StrategicBidding(base_markup=0.1, scarcity_threshold=0.8)
    utility = UtilityAgent(
        utility_id="utility_1",
        portfolio=portfolio,
        bidding_strategy=strategy,
        enable_bilateral_contracts=True
    )
    
    return utility

def update_renewable_availability(utility: UtilityAgent, hour: int):
    """Update renewable availability based on hour"""
    # Solar availability pattern (peak at noon)
    solar_pattern = max(0, np.sin(np.pi * max(0, min(hour - 6, 12)) / 12))
    
    # Wind variability
    wind_base = 0.3
    wind_variation = 0.2 * np.sin(2 * np.pi * hour / 24)
    wind_pattern = max(0.1, wind_base + wind_variation)  # Minimum 10% availability
    
    # Update generators
    for gen in utility.portfolio.generators:
        if gen.unit_type == 'solar':
            gen.availability = solar_pattern
            # Solar should be 0 at night
            if hour < 6 or hour > 18:
                gen.availability = 0.0
        elif gen.unit_type == 'wind':
            # Use base availability from config and add variation
            base_availability = 0.35  # Default if not specified
            if hasattr(gen, 'base_availability'):
                base_availability = gen.base_availability
            gen.availability = max(0.1, base_availability + 0.1 * np.sin(2 * np.pi * hour / 24))
        # Other types keep their configured availability

# ==================== INTEGRATION WITH MAIN SIMULATOR ====================

def integrate_utilities_into_simulation(config: Dict) -> List[Agent]:
    """Create utility agents for the main simulation"""
    agents = []
    
    # Create multiple competing utilities
    utility_configs = config.get('utilities', [])
    
    for util_config in utility_configs:
        # Create generators for this utility
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
        else:
            strategy = MarginalCostBidding()
            
        # Create utility agent
        utility = UtilityAgent(
            utility_id=util_config['id'],
            portfolio=portfolio,
            bidding_strategy=strategy,
            enable_bilateral_contracts=util_config.get('enable_contracts', False)
        )
        
        agents.append(utility)
        
    return agents

if __name__ == "__main__":
    # Example: Create and test a utility
    utility = create_example_utility()
    
    # Update renewable availability for hour 12 (noon)
    update_renewable_availability(utility, 12)
    
    # Create market state
    market_state = {
        'last_price': 50.0,
        'price_forecast': [40 + 20*np.sin(2*np.pi*h/24) for h in range(24)],
        'supply_ratio': 0.9  # 90% of capacity available
    }
    
    # Generate bids
    bids = utility.create_bid(market_state, time_period=12)
    
    print(f"Utility {utility.agent_id} Portfolio:")
    print(f"Total Capacity: {utility.portfolio.total_capacity} MW")
    print(f"\nGenerated {len(bids)} bids:")
    for bid in bids:
        print(f"  {bid.agent_id}: {bid.quantity:.1f} MW @ ${bid.price:.2f}/MWh")
        
    # Show portfolio metrics
    print("\nPortfolio Composition:")
    for gen_type, units in utility.portfolio.generators_by_type.items():
        total_cap = sum(u.capacity for u in units)
        print(f"  {gen_type}: {total_cap} MW ({len(units)} units)")