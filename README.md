# GRIDMIND AI — MVP

**Intelligent Energy Orchestration for the EV Network**

A working MVP demonstrating AI-powered energy optimization across 5 Indian metro clusters: Delhi NCR, Mumbai MMR, Bengaluru, Hyderabad, Chennai.

---

## Quick Start

```bash
# 1. Clone / navigate to project
cd /c/Users/Raghi/EROH

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python -m streamlit run app.py --server.port 8501

# 4. Open in browser
# http://localhost:8501
```

---

## System Parameters

| Parameter | Value |
|-----------|-------|
| **Installed Capacity** | 1,200 MW |
| **DC Fast Charging** | 720 MW |
| **Battery Swapping** | 480 MW |
| **Energy Storage** | 600 MWh |
| **Renewable Share** | 35% |
| **Reserve Requirement** | 5% (Hard Constraint) |
| **Baseline Reliability** | 78% |
| **Target Reliability** | 89.7% |
| **Baseline Losses** | 10% |
| **Target Losses** | 8% |
| **Min EBITDA Margin** | 12% |

All financial calculations in **INR**.

---

## Features Implemented

### 1. Synthetic Data Generation (5 Years)
- Hourly data for all 5 clusters (43,800 hours × 5 = 219,000 records)
- Realistic daily cycles, weekly patterns, seasonal variations
- Regional differences, weather effects, heatwave events
- Solar/wind generation with monsoon impacts
- IEX DAM prices with evening spikes + jump diffusion
- Battery SOC/SOH tracking

### 2. Forecasting Engine
- **Models**: HistGradientBoostingRegressor (Demand + Renewable)
- **Features**: Temporal (hour, day, month), lag (1,2,3,6,12,24,168h), rolling stats, weather interactions
- **Outputs**: 24-hour forecast with MAE/WAPE metrics
- **Note**: Trained on synthetic data — accuracy ≠ real-world performance

### 3. Energy Optimization Engine
- **Solver**: SciPy MILP (HiGHS) with fallback heuristic
- **Horizon**: 24 hours
- **Decision Variables**: Renewable, DISCOM, IEX procurement, battery charge/discharge, served/unserved, curtailment
- **Constraints**: Energy balance, SOC dynamics, power limits, grid limits, 5% reserve (HARD), no simultaneous charge/discharge
- **Strategies**: 
  - **Optimized**: Forecast-driven, battery arbitrage, renewable maximization
  - **Baseline**: Simple priority dispatch, minimal battery use

### 4. Battery Management
- **Stationary Storage**: LFP, 92% round-trip, 10-90% SOC, 0.5C/1C rates
- **Swap Packs**: 50kWh packs, inventory tracking (charged/partial/empty)
- **Degradation**: Throughput + DoD + calendar aging model
- **SOH Tracking**: Per-asset state of health evolution

### 5. Dynamic Pricing Engine
- **Segments**: Retail (ε=-0.4), Ride-hailing (ε=-0.65), Logistics (ε=-0.5)
- **Tariff Bounds**: ±20% of previous day average (HARD)
- **Drivers**: Procurement cost, renewable surplus, battery availability, demand level
- **Output**: 24-hour tariff curves per segment with bounds visualization

### 6. Baseline vs Optimized Simulation
- Runs both strategies side-by-side
- Compares: Reliability, Losses, Renewable Utilization, Revenue, EBITDA, EBITDA Margin
- No hardcoded targets — results from actual calculations

### 7. Scenario Simulator (4+ Scenarios)
1. **Normal Operations** — Baseline conditions
2. **Renewable -30%** — Solar/wind shortfall
3. **IEX +50%** — Evening price spike
4. **Demand +25% (Delhi, Mumbai)** — Regional surge
5. **Custom** — Interactive sliders for all parameters

### 8. Dashboard (8 Tabs)
- 📊 **Executive Overview** — System KPIs, reliability, EBITDA, procurement mix
- 🗺️ **Regional Operations** — India map, cluster details, hourly demand/supply
- 🤖 **AI Forecasting** — Model metrics, 24h demand & renewable forecasts
- ⚡ **Energy Optimization** — Procurement stack, battery dispatch, energy balance table
- 🔋 **Battery Intelligence** — SOC/SOH trends, swap inventory, degradation costs
- 💰 **Market & Pricing** — IEX/DISCOM/RE price curves, procurement recommendations, dynamic tariffs
- 🎮 **Scenario Simulator** — Interactive parameters, predefined + custom scenarios
- 📈 **Financial Performance** — Baseline vs Optimized table, target achievement, sensitivity analysis

---

## Architecture

```
app.py                 # Single-file Streamlit application
├── generate_synthetic_data()      # 5-year hourly data for 5 clusters
├── train_forecast_models()        # HistGradientBoostingRegressor training
├── generate_forecast()            # 24h rolling forecast
├── solve_energy_optimization()    # MILP + heuristic fallback
├── simulate_battery_operations()  # SOC/SOH/swap tracking
├── calculate_dynamic_tariffs()    # Segment pricing with elasticity
├── run_simulation()               # Full pipeline for both strategies
└── main()                         # Streamlit UI with 8 tabs
```

---

## Key Mathematical Rules

### Reliability
```
Reliability (%) = Energy Successfully Delivered / Total Forecast Demand × 100
```

### Energy Losses
```
Losses (%) = Total Modeled Energy Losses / Total Energy Input × 100
```
Includes: Charger conversion, battery round-trip, self-discharge, distribution (2%)

### EBITDA
```
EBITDA = Revenue - Operating Expenses (excl. depreciation, amortization, interest, taxes)
```

### EBITDA Margin
```
EBITDA Margin (%) = EBITDA / Revenue × 100
```

### Reserve Constraint (HARD)
```
SOC(t) ≥ 5% × Total Storage Capacity  ∀ t
Swap Pack Inventory ≥ 5% × Total Swap Inventory  ∀ t
```

### Tariff Bounds (HARD)
```
0.8 × Prev_Day_Avg ≤ Tariff ≤ 1.2 × Prev_Day_Avg
```

---

## Assumptions & Limitations

### Assumptions (Explicit)
- LFP battery chemistry, 92% round-trip efficiency
- 2% distribution losses
- ₹2,500/MWh degradation cost
- Customer elasticities: Retail -0.4, Ride-hailing -0.65, Logistics -0.5
- IEX base ₹4,500/MWh, peak 2.5×, off-peak 0.6×
- Renewable PPA: Solar ₹2,800, Wind ₹3,000/MWh
- DISCOM: ₹6,500-7,500/MWh by cluster
- No inter-cluster transmission (configurable but disabled)
- Synthetic data only — no live IEX/weather/API integration

### Limitations
1. **Synthetic Data**: All forecasts trained on generated data — not representative of real accuracy
2. **Single-Cluster Optimization**: No inter-cluster energy transfer modeled
3. **24h Horizon**: Rolling horizon but no multi-day lookahead
4. **Heuristic Fallback**: If MILP fails, uses rule-based dispatch (logged)
5. **Simplified Degradation**: Rainflow counting approximated
6. **No Network Constraints**: Beyond grid connection limits
7. **Fixed Segment Shares**: 50/30/20 split, no dynamic segmentation

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `streamlit: command not found` | Use `python -m streamlit run app.py` |
| Port 8501 in use | Change port: `--server.port 8502` |
| MILP solver fails | App automatically falls back to heuristic |
| Slow initial load | First run trains models (~30-60s), subsequent runs use cache |
| Memory issues | Reduce `SIM_YEARS` in config (default 5) |

---

## Development

### Extending the MVP
- Add inter-cluster transmission: Modify `solve_energy_optimization()` with transfer variables
- Real data integration: Replace `generate_synthetic_data()` with data loader
- Better solver: Install Pyomo + HiGHS for full MILP
- More scenarios: Add to `scenario_params` in sidebar
- Export results: Add CSV download buttons in each tab

### Configuration
All parameters at top of `app.py`:
- `CLUSTER_CONFIG` — Capacity, location, grid limits
- `BATTERY_CONFIG` — Chemistry, efficiency, degradation
- `PRICING_CONFIG` — PPA, DISCOM, IEX prices
- `SEGMENT_ELASTICITY` — Price sensitivity by segment
- `RANDOM_SEED` — Reproducibility (default 42)

---

## License

Proprietary — GRIDMIND AI MVP for demonstration purposes.

---

## Support

For issues or questions, check the console output for error logs. The app prints initialization status and any solver fallback notifications.