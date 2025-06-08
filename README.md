# Electricity Market Simulator - Complete Documentation

## Overview

This is a comprehensive electricity market simulator that models day-ahead electricity markets with multiple utilities, diverse generation portfolios, and sophisticated bidding strategies. The simulator supports both short-term analysis and multi-year simulations using time-series profiles.

### Key Capabilities

- **Multi-utility competition** with portfolio-based bidding
- **Diverse generation mix**: coal, gas, nuclear, wind, solar, hydro, storage
- **Multiple bidding strategies**: marginal cost, strategic, and optimization-based
- **Profile-based simulation** for multi-year analysis
- **Policy modeling**: carbon tax, renewable mandates, capacity payments
- **Comprehensive analytics** and visualization

## Features

### Market Modeling
- Day-ahead energy market with hourly resolution
- Merit-order dispatch with uniform pricing
- Market power analysis (HHI, Lerner index)
- Bilateral contracts support

### Agent Types
1. **Utility Companies**: Portfolio of generators with unified bidding
2. **Independent Generators**: Single-unit participants
3. **Consumers**: Price-elastic demand
4. **Storage**: Battery systems with arbitrage

### Bidding Strategies
- **Marginal Cost**: Competitive bidding at production cost
- **Strategic**: Markup-based bidding with market power
- **Optimal**: Unit commitment with forward-looking optimization

### Dynamic Profiles
- Hourly load profiles with seasonal patterns
- Renewable capacity factors (wind/solar)
- Fuel price time series
- Hydro availability curves


## Installation

### Requirements

```bash
Python 3.8+
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
pyyaml >= 5.4.0
```

### Setup

```bash
# Clone or download the project
cd electricity-market-simulator

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib seaborn pyyaml
```

## 🏃 Quick Start

### Option 1: Basic Simulation (Single Generators)

```bash
# Run basic simulation with individual generators
python main.py

# Visualize results
python visualization.py
```

### Option 2: Utility Simulation (Portfolio-Based)

```bash
# Run simulation with utility companies
python run_utility_simulation.py

# Results saved to: results/utility_simulation_results.csv
# Plots saved to: results/plots/
```

### Option 3: Profile-Based Multi-Year Simulation

```bash
# Run multi-year simulation with profiles
python run_profile_simulation.py

# This will:
# 1. Create example profiles if missing
# 2. Run simulation for profile duration
# 3. Generate comprehensive analysis
```

## 📄 File Descriptions

### Core Files

#### `main.py`
- Core simulation engine
- Base classes: `Agent`, `Generator`, `Consumer`, `StorageUnit`
- Market clearing mechanism
- Policy implementations

#### `utility_agent.py`
- `UtilityAgent`: Manages portfolio of generators
- `GeneratorUnit`: Detailed generator model
- Bidding strategies: `MarginalCostBidding`, `StrategicBidding`, `OptimalBidding`
- Market power analysis tools

#### `run_utility_simulation.py`
- `ExtendedSimulation`: Handles utility agents
- `ExtendedDayAheadMarket`: Proper utility state updates
- Fixed tracking and results generation
- Performance analysis functions

#### `run_profile_simulation.py`
- `ProfileManager`: Loads and manages CSV profiles
- `ProfileBasedUtilityAgent`: Dynamic parameter updates
- `ProfileBasedSimulation`: Multi-year simulation engine
- Automatic profile generation

### Configuration Files

#### `default_config.yaml`
Basic configuration for simple agents:
```yaml
simulation:
  time_periods: 168  # 1 week
generators:
  - type: thermal
    params:
      capacity: 500
      marginal_cost: 30
```

#### `utilities_config.yaml`
Detailed utility portfolios:
```yaml
utilities:
  - id: "MegaPower_Corp"
    strategy: "strategic"
    markup: 0.15
    generators:
      - unit_id: "MP_Coal_1"
        capacity: 600
        marginal_cost: 28
```

#### `profile_config.yaml`
Profile-based configuration:
```yaml
profiles:
  - name: "load_profile"
    file: "load_profile.csv"
  - name: "fuel_prices"
    file: "fuel_prices.csv"
```

### Profile Files

All in CSV format with timestamp column:

- **load_profile.csv**: Hourly demand for each consumer
- **wind_cf.csv**: Wind capacity factors (0-1)
- **solar_cf.csv**: Solar capacity factors by hour
- **fuel_prices.csv**: $/MMBtu for gas, coal, etc.
- **hydro_availability.csv**: Seasonal water availability

## ⚙️ Configuration Guide

### Basic Generator Configuration

```yaml
generators:
  - unit_id: "unit_1"
    unit_type: "gas"
    capacity: 100        # MW
    marginal_cost: 45    # $/MWh
    min_output: 30       # MW
    ramp_rate: 50        # MW/period
    startup_cost: 5000   # $
    efficiency: 0.55     # Heat rate = 1/efficiency
```

### Utility Configuration

```yaml
utilities:
  - id: "UtilityName"
    strategy: "strategic"     # or "marginal_cost", "optimal"
    markup: 0.10             # 10% markup for strategic
    generators: [...]         # List of units
```

### Consumer Configuration

```yaml
consumers:
  - agent_id: "city_load"
    demand_profile: [800, 850, ...]  # Or use profile reference
    price_elasticity: -0.1           # Demand response
```

## 🎮 Running Simulations

### Basic Run

```python
# In Python
from main import Simulation, create_default_config

config = create_default_config()
sim = Simulation(config)
results = sim.run(168)  # 1 week
```

### Utility Simulation

```bash
# Command line
python run_utility_simulation.py

# Or with custom config
python run_utility_simulation.py --config my_config.yaml
```

### Profile-Based Simulation

```bash
# Auto-generates example profiles if missing
python run_profile_simulation.py

# Runs for entire profile duration (e.g., 2 years)
```

## Analysis and Results

### Output Files

1. **CSV Results**: Hourly market data
   - Time period and timestamp
   - Clearing prices
   - Generation by unit and type
   - Fuel prices (if profile-based)

2. **Visualizations**:
   - Price duration curves
   - Generation dispatch stacks
   - Market share analysis
   - Price vs fuel correlations
   - Renewable penetration

## Next Steps (In the process)

1. **Add Transmission Constraints**
   ```python
   class NetworkConstraint:
       def __init__(self, from_zone, to_zone, limit):
           self.limit = limit
   ```

2. **Implement Ancillary Services**
   ```python
   class AncillaryMarket(Market):
       def __init__(self):
           self.products = ['regulation', 'spinning_reserve']
   ```

3. **Add Forecasting**
   ```python
   class PriceForecast:
       def predict(self, historical_data):
           # ML model for price prediction
   ```

4. **Multi-Area Markets**
   - Define zones
   - Add transmission limits
   - Implement LMP calculation

## References

- Kirschen & Strbac: *Fundamentals of Power System Economics*
- Wood & Wollenberg: *Power Generation, Operation, and Control*
- FERC market design documentation