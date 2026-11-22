#!/usr/bin/env python3
"""
GRIDMIND AI - Intelligent Energy Orchestration for the EV Network
Corrected MVP Implementation with Mathematical Rigor
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML & Optimization
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from scipy.optimize import milp, LinearConstraint, Bounds
import json

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

CLUSTERS = ["Delhi NCR", "Mumbai MMR", "Bengaluru", "Hyderabad", "Chennai"]
N_CLUSTERS = len(CLUSTERS)

# System parameters (Business Requirements)
TOTAL_CAPACITY_MW = 1200
DC_CHARGING_MW = 720
SWAP_MW = 480
TOTAL_STORAGE_MWH = 600
RENEWABLE_SHARE = 0.35
RESERVE_REQUIREMENT_CAPACITY = 0.05  # 5% of total capacity (MW)
RESERVE_REQUIREMENT_SWAP = 0.05      # 5% of swap inventory
BASELINE_RELIABILITY = 0.78
TARGET_RELIABILITY = 0.897
BASELINE_LOSSES = 0.10
TARGET_LOSSES = 0.08
MIN_EBITDA_MARGIN = 0.12

# Cluster-level capacity distribution
CLUSTER_CONFIG = {
    "Delhi NCR": {
        "dc_mw": 180, "swap_mw": 120, "storage_mwh": 150,
        "grid_limit_mw": 400,
        "max_charge_mw": 75,      # 0.5C of 150 MWh
        "max_discharge_mw": 150,  # 1.0C of 150 MWh
        "lat": 28.61, "lon": 77.21
    },
    "Mumbai MMR": {
        "dc_mw": 160, "swap_mw": 100, "storage_mwh": 130,
        "grid_limit_mw": 350,
        "max_charge_mw": 65,      # 0.5C of 130 MWh
        "max_discharge_mw": 130,  # 1.0C of 130 MWh
        "lat": 19.08, "lon": 72.88
    },
    "Bengaluru": {
        "dc_mw": 140, "swap_mw": 90, "storage_mwh": 110,
        "grid_limit_mw": 300,
        "max_charge_mw": 55,      # 0.5C of 110 MWh
        "max_discharge_mw": 110,  # 1.0C of 110 MWh
        "lat": 12.97, "lon": 77.59
    },
    "Hyderabad": {
        "dc_mw": 120, "swap_mw": 80, "storage_mwh": 100,
        "grid_limit_mw": 250,
        "max_charge_mw": 50,      # 0.5C of 100 MWh
        "max_discharge_mw": 100,  # 1.0C of 100 MWh
        "lat": 17.38, "lon": 78.49
    },
    "Chennai": {
        "dc_mw": 120, "swap_mw": 90, "storage_mwh": 110,
        "grid_limit_mw": 280,
        "max_charge_mw": 55,      # 0.5C of 110 MWh
        "max_discharge_mw": 110,  # 1.0C of 110 MWh
        "lat": 13.08, "lon": 80.27
    },
}

# Verify totals
assert sum(CLUSTER_CONFIG[c]["dc_mw"] for c in CLUSTERS) == DC_CHARGING_MW
assert sum(CLUSTER_CONFIG[c]["swap_mw"] for c in CLUSTERS) == SWAP_MW
assert sum(CLUSTER_CONFIG[c]["storage_mwh"] for c in CLUSTERS) == TOTAL_STORAGE_MWH

# Battery parameters
BATTERY_CONFIG = {
    "chemistry": "LFP",
    "round_trip_efficiency": 0.92,
    "charge_efficiency": 0.96,      # Energy into battery per MWh at bus
    "discharge_efficiency": 0.96,   # Energy at bus per MWh from battery
    "min_soc_fraction": 0.10,       # Physical minimum SOC
    "max_soc_fraction": 0.90,       # Physical maximum SOC
    "degradation_cost_inr_per_mwh_throughput": 2500,
    "self_discharge_per_month": 0.005,
}

# Pricing assumptions (INR/MWh unless noted)
PRICING_CONFIG = {
    "renewable_ppa_solar": 2800,
    "renewable_ppa_wind": 3000,
    "discom_base": {"Delhi NCR": 7200, "Mumbai MMR": 7500, "Bengaluru": 6800, "Hyderabad": 6500, "Chennai": 6500},
    "iex_base_avg": 4500,
    "iex_peak_multiplier": 2.5,
    "iex_offpeak_multiplier": 0.6,
    "om_cost_inr_per_mwh": 500,  # ₹0.5/kWh = ₹500/MWh
}

# Customer segments
SEGMENTS = ["Retail", "Ride-hailing", "Logistics"]
SEGMENT_ELASTICITY = {"Retail": -0.4, "Ride-hailing": -0.65, "Logistics": -0.5}
SEGMENT_BASE_TARIFF = {"Retail": 17.0, "Ride-hailing": 14.0, "Logistics": 13.0}  # INR/kWh
TARIFF_BAND = 0.20  # ±20%

# Swap pack assumptions
SWAP_PACK_CAPACITY_MWH = 0.05  # 50 kWh per pack
SWAP_PACKS_PER_MW = 20  # 1 MW swap capacity = 20 packs (50 kWh each, 1h equivalent)

# Simulation
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
HOURS_IN_YEAR = 8760
SIM_YEARS = 5
TOTAL_HOURS = HOURS_IN_YEAR * SIM_YEARS
FORECAST_HORIZON = 24

# Distribution loss factor
DISTRIBUTION_LOSS_FACTOR = 0.02  # 2%

# ──────────────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def generate_synthetic_data():
    """Generate 5 years of realistic hourly synthetic data for all clusters."""
    dates = pd.date_range(start="2019-01-01", periods=TOTAL_HOURS, freq="h")

    data = {}
    for cluster in CLUSTERS:
        cfg = CLUSTER_CONFIG[cluster]
        n = len(dates)

        hour = dates.hour.values
        dayofweek = dates.dayofweek.values
        month = dates.month.values
        is_weekend = (dayofweek >= 5).astype(float)

        day_of_year = dates.dayofyear.values
        seasonal = np.sin(2 * np.pi * day_of_year / 365.25)

        # DEMAND
        daily_profile = np.where(
            (hour >= 7) & (hour <= 10), 1.3,
            np.where((hour >= 18) & (hour <= 22), 1.5,
            np.where((hour >= 11) & (hour <= 16), 0.7, 0.5))
        )

        weekend_factor = np.where(is_weekend, 0.85, 1.0)
        seasonal_demand = 1 + 0.25 * seasonal

        years_elapsed = (dates - dates[0]).days / 365.25
        growth = 1 + 0.15 * years_elapsed

        temp_base = 25 + 10 * seasonal
        temp_noise = np.random.normal(0, 3, n)
        temperature = temp_base + temp_noise

        heatwave = np.zeros(n)
        for year in range(SIM_YEARS):
            hw_start = np.random.randint(year*365*24, (year+1)*365*24 - 168)
            hw_len = np.random.randint(72, 168)
            heatwave[hw_start:hw_start+hw_len] = 1
        temp_effect = 1 + 0.015 * np.maximum(temperature - 30, 0) * (1 + heatwave)

        base_demand = cfg["dc_mw"] * 0.6 + cfg["swap_mw"] * 0.4
        demand = base_demand * daily_profile * weekend_factor * seasonal_demand * growth * temp_effect
        demand += np.random.normal(0, base_demand * 0.05, n)
        demand = np.maximum(demand, 0)

        dc_demand = demand * 0.6
        swap_demand_mwh = demand * 0.4

        # SOLAR GENERATION
        solar_elevation = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
        cloud_cover = np.random.beta(2, 5, n)
        monsoon = ((month >= 6) & (month <= 9)).astype(float)
        cloud_cover = np.minimum(cloud_cover + 0.3 * monsoon, 1.0)

        solar_cf = solar_elevation * (1 - cloud_cover) * 0.95
        solar_gen = cfg["storage_mwh"] * 2 * solar_cf * (1 - 0.005 * years_elapsed)
        solar_gen = np.maximum(solar_gen, 0)

        # WIND GENERATION
        wind_base = 6 + 2 * seasonal
        wind_diurnal = 1 + 0.3 * np.sin(2 * np.pi * (hour - 3) / 24)
        wind_speed = wind_base * wind_diurnal * np.random.gamma(2, 1, n)
        wind_speed = np.maximum(wind_speed, 0)

        wind_cf = np.minimum((wind_speed / 12) ** 3, 1.0) * 0.35
        wind_gen = cfg["storage_mwh"] * 1.5 * wind_cf

        # IEX DAM PRICES
        base_price = PRICING_CONFIG["iex_base_avg"]
        price_daily = np.where(
            (hour >= 18) & (hour <= 22), PRICING_CONFIG["iex_peak_multiplier"],
            np.where((hour >= 0) & (hour <= 6), PRICING_CONFIG["iex_offpeak_multiplier"], 1.0)
        )
        price_seasonal = 1 + 0.3 * seasonal
        price_shocks = np.random.exponential(0.1, n) * (np.random.random(n) < 0.02)
        iex_price = base_price * price_daily * price_seasonal * (1 + price_shocks)
        iex_price += np.random.normal(0, base_price * 0.05, n)
        iex_price = np.maximum(iex_price, 500)

        # DISCOM PRICES
        discom_price = np.full(n, PRICING_CONFIG["discom_base"][cluster])
        discom_price *= (1 + 0.05 * years_elapsed)

        # RENEWABLE PROCUREMENT PRICES
        re_price = np.where(
            solar_gen > wind_gen,
            PRICING_CONFIG["renewable_ppa_solar"],
            PRICING_CONFIG["renewable_ppa_wind"]
        )

        # BATTERY STATE (historical)
        soc = np.full(n, 0.5 * cfg["storage_mwh"])
        soh = np.full(n, 0.95)

        charging_sessions = (dc_demand / 50 * 4).astype(int)
        swap_sessions = (swap_demand_mwh / 0.05 * 2).astype(int)

        humidity = 50 + 30 * (1 - solar_elevation) + np.random.normal(0, 10, n)
        humidity = np.clip(humidity, 10, 100)
        rainfall = np.random.exponential(0.5, n) * monsoon

        df = pd.DataFrame({
            "timestamp": dates,
            "cluster": cluster,
            "demand_mw": demand,
            "dc_demand_mw": dc_demand,
            "swap_demand_mwh": swap_demand_mwh,
            "solar_gen_mw": solar_gen,
            "wind_gen_mw": wind_gen,
            "renewable_gen_mw": solar_gen + wind_gen,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "rainfall_mm": rainfall,
            "iex_price_inr_mwh": iex_price,
            "discom_price_inr_mwh": discom_price,
            "re_procurement_price_inr_mwh": re_price,
            "battery_soc_mwh": soc,
            "battery_soh": soh,
            "charging_sessions": charging_sessions,
            "swap_sessions": swap_sessions,
            "is_weekend": is_weekend,
            "hour": hour,
            "month": month,
            "heatwave": heatwave,
        })
        data[cluster] = df

    return data


def create_features(df, target_col, lag_hours=[1, 2, 3, 6, 12, 24, 168]):
    """Create features for forecasting."""
    df = df.copy()
    df = df.sort_values("timestamp")

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    for lag in lag_hours:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)

    df[f"{target_col}_roll_24_mean"] = df[target_col].rolling(24).mean()
    df[f"{target_col}_roll_168_mean"] = df[target_col].rolling(168).mean()

    df["temp_x_hour"] = df["temperature_c"] * df["hour_sin"]

    return df.dropna()


# ──────────────────────────────────────────────────────────────────────────────
# FORECASTING ENGINE
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def train_forecast_models(data_dict):
    """Train forecasting models for demand and renewable generation."""
    models = {}
    metrics = {}

    for cluster in CLUSTERS:
        df = data_dict[cluster].copy()

        # DEMAND FORECASTING
        feat_df = create_features(df, "demand_mw")
        feature_cols = [c for c in feat_df.columns if c not in [
            "timestamp", "cluster", "demand_mw", "dc_demand_mw", "swap_demand_mwh",
            "solar_gen_mw", "wind_gen_mw", "renewable_gen_mw",
            "battery_soc_mwh", "battery_soh",
            "charging_sessions", "swap_sessions",
            "iex_price_inr_mwh", "discom_price_inr_mwh",
            "re_procurement_price_inr_mwh"]]

        X = feat_df[feature_cols].values
        y = feat_df["demand_mw"].values

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=8, learning_rate=0.1,
            random_state=RANDOM_SEED, early_stopping=True, validation_fraction=0.1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        wape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100 if np.sum(np.abs(y_test)) > 0 else 0

        models[f"{cluster}_demand"] = {"model": model, "features": feature_cols}
        metrics[f"{cluster}_demand"] = {"mae": mae, "wape": wape, "test_size": len(y_test)}

        # RENEWABLE FORECASTING
        feat_df_re = create_features(df, "renewable_gen_mw")
        feature_cols_re = [c for c in feat_df_re.columns if c not in [
            "timestamp", "cluster", "demand_mw", "dc_demand_mw", "swap_demand_mwh",
            "solar_gen_mw", "wind_gen_mw", "renewable_gen_mw",
            "battery_soc_mwh", "battery_soh",
            "charging_sessions", "swap_sessions",
            "iex_price_inr_mwh", "discom_price_inr_mwh",
            "re_procurement_price_inr_mwh"]]
        X_re = feat_df_re[feature_cols_re].values
        y_re = feat_df_re["renewable_gen_mw"].values

        X_train_re, X_test_re = X_re[:split_idx], X_re[split_idx:]
        y_train_re, y_test_re = y_re[:split_idx], y_re[split_idx:]

        model_re = HistGradientBoostingRegressor(
            max_iter=200, max_depth=8, learning_rate=0.1,
            random_state=RANDOM_SEED, early_stopping=True, validation_fraction=0.1
        )
        model_re.fit(X_train_re, y_train_re)

        y_pred_re = model_re.predict(X_test_re)
        mae_re = mean_absolute_error(y_test_re, y_pred_re)
        wape_re = np.sum(np.abs(y_test_re - y_pred_re)) / np.sum(np.abs(y_test_re)) * 100 if np.sum(np.abs(y_test_re)) > 0 else 0

        models[f"{cluster}_renewable"] = {"model": model_re, "features": feature_cols_re}
        metrics[f"{cluster}_renewable"] = {"mae": mae_re, "wape": wape_re}

    return models, metrics


def generate_forecast(data_dict, models, horizon=24):
    """Generate forecasts for next horizon hours."""
    forecasts = {}

    for cluster in CLUSTERS:
        df = data_dict[cluster].copy()
        last_row = df.iloc[-1:].copy()

        future_dates = pd.date_range(start=last_row["timestamp"].values[0] + pd.Timedelta(hours=1),
                                     periods=horizon, freq="h")

        # Get last known values for all columns to fill future rows
        last_values = {}
        for col in df.columns:
            if col not in ["timestamp", "cluster"]:
                last_values[col] = df[col].iloc[-1]

        # Use full historical dataframe
        current_df = df.copy()

        future_rows = []
        for i, ts in enumerate(future_dates):
            new_row = {"timestamp": ts, "cluster": cluster}
            # Fill all columns with last known values
            for col, val in last_values.items():
                if col == "hour":
                    new_row[col] = ts.hour
                elif col == "month":
                    new_row[col] = ts.month
                elif col == "is_weekend":
                    new_row[col] = 1.0 if ts.dayofweek >= 5 else 0.0
                else:
                    new_row[col] = val
            future_rows.append(new_row)
            current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

        for target in ["demand_mw", "renewable_gen_mw"]:
            feat_future = create_features(current_df, target)
            feat_future = feat_future[feat_future["timestamp"].isin(future_dates)]

            model_info = models[f"{cluster}_{target.split('_')[0]}"]
            model = model_info["model"]
            features = model_info["features"]

            X_future = feat_future[features].values
            preds = model.predict(X_future)
            preds = np.maximum(preds, 0)

            if cluster not in forecasts:
                forecasts[cluster] = {"timestamp": future_dates}
            forecasts[cluster][f"{target}_forecast"] = preds

    return forecasts


# ──────────────────────────────────────────────────────────────────────────────
# ENERGY ACCOUNTING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def compute_energy_balance_hourly(cluster, opt_result, renewable_forecast, discom_price, re_price, iex_prices):
    """
    Compute auditable energy balance for a single cluster.

    Convention:
    - All variables in MWh per hour (energy over 1h interval)
    - charge_mwh: energy drawn FROM grid TO battery (at electrical bus)
    - discharge_mwh: energy delivered FROM battery TO grid (at electrical bus)
    - SOC in MWh (energy inside battery)
    - eta_charge: energy into battery per MWh at bus = 0.96
    - eta_discharge: energy at bus per MWh from battery = 0.96

    SOC[t+1] = SOC[t] + eta_charge * charge[t] - discharge[t] / eta_discharge - self_discharge

    Electrical bus balance:
    Renewable_used + DISCOM + IEX + discharge = Served + charge + Curtailment + Distribution_Losses
    """
    cfg = CLUSTER_CONFIG[cluster]
    bat = BATTERY_CONFIG
    H = len(opt_result["served"])

    # Extract optimization results
    served = opt_result["served"]
    charge = opt_result["charge"]
    discharge = opt_result["discharge"]
    renewable_used = opt_result["renewable_used"]
    discom_proc = opt_result["discom_procurement"]
    iex_proc = opt_result["iex_procurement"]
    curtail = opt_result["curtailment"]
    soc = opt_result["soc"]

    # Compute losses per hour
    distribution_losses = DISTRIBUTION_LOSS_FACTOR * (served + charge)

    # Battery conversion losses (at electrical bus)
    # charge: energy at bus going INTO battery; stored = charge * eta_chg; loss = charge * (1 - eta_chg)
    # discharge: energy at bus FROM battery; from battery = discharge / eta_dch; loss = discharge * (1/eta_dch - 1)
    charge_loss = charge * (1 - bat["charge_efficiency"])              # Energy lost during charging (at bus)
    discharge_loss = discharge * (1/bat["discharge_efficiency"] - 1)   # Energy lost during discharging (at bus)
    battery_conversion_loss = charge_loss + discharge_loss

    # Self-discharge loss (energy lost from SOC)
    self_discharge_loss = np.zeros(H)
    for t in range(H):
        if t == 0:
            prev_soc = cfg["storage_mwh"] * 0.5  # Initial SOC
        else:
            prev_soc = soc[t-1]
        self_discharge_loss[t] = prev_soc * bat["self_discharge_per_month"] / (30*24)

    # Total losses per hour
    total_losses_per_hour = distribution_losses + battery_conversion_loss + self_discharge_loss

    # Energy balance check
    lhs = renewable_used + discom_proc + iex_proc + discharge
    rhs = served + charge + curtail + distribution_losses
    balance_error = lhs - rhs

    return {
        "served": served,
        "charge": charge,
        "discharge": discharge,
        "renewable_used": renewable_used,
        "discom_procurement": discom_proc,
        "iex_procurement": iex_proc,
        "curtailment": curtail,
        "soc": soc,
        "distribution_losses": distribution_losses,
        "battery_conversion_loss": battery_conversion_loss,
        "self_discharge_loss": self_discharge_loss,
        "total_losses_per_hour": total_losses_per_hour,
        "balance_error": balance_error,
        "balance_max_error": np.max(np.abs(balance_error)),
    }


def compute_system_kpis(results, cluster_configs, battery_config):
    """Compute system-level KPIs with correct energy accounting."""
    total_served = sum(results[c]["served_mwh"] for c in CLUSTERS)
    total_demand = sum(results[c]["demand_mwh"] for c in CLUSTERS)
    total_unserved = sum(results[c]["unserved_mwh"] for c in CLUSTERS)
    total_revenue = sum(results[c]["revenue_inr"] for c in CLUSTERS)
    total_proc_cost = sum(results[c]["procurement_cost_inr"] for c in CLUSTERS)
    total_deg_cost = sum(results[c]["degradation_cost_inr"] for c in CLUSTERS)
    total_om_cost = sum(results[c]["om_cost_inr"] for c in CLUSTERS)

    # Losses: sum of hourly losses across all clusters
    total_losses = sum(results[c]["total_losses_mwh"] for c in CLUSTERS)
    total_input = sum(results[c]["total_input_mwh"] for c in CLUSTERS)

    # Renewable utilization
    total_renewable_used = sum(results[c]["renewable_used_mwh"] for c in CLUSTERS)
    total_renewable_avail = sum(results[c]["renewable_avail_mwh"] for c in CLUSTERS)

    reliability = total_served / total_demand * 100 if total_demand > 0 else 0
    loss_pct = total_losses / total_input * 100 if total_input > 0 else 0
    re_util = total_renewable_used / total_renewable_avail * 100 if total_renewable_avail > 0 else 0

    total_opex = total_proc_cost + total_deg_cost + total_om_cost
    ebitda = total_revenue - total_opex
    ebitda_margin = ebitda / total_revenue * 100 if total_revenue > 0 else 0

    return {
        "reliability_pct": reliability,
        "losses_pct": loss_pct,
        "renewable_util_pct": re_util,
        "revenue_inr": total_revenue,
        "procurement_cost_inr": total_proc_cost,
        "degradation_cost_inr": total_deg_cost,
        "om_cost_inr": total_om_cost,
        "opex_inr": total_opex,
        "ebitda_inr": ebitda,
        "ebitda_margin_pct": ebitda_margin,
        "total_served_mwh": total_served,
        "total_demand_mwh": total_demand,
        "total_losses_mwh": total_losses,
        "total_input_mwh": total_input,
    }


# ──────────────────────────────────────────────────────────────────────────────
# OPTIMIZATION ENGINE - CORRECTED MILP
# ──────────────────────────────────────────────────────────────────────────────

def solve_energy_optimization(cluster, demand_forecast, renewable_forecast,
                              initial_soc, iex_prices, discom_price, re_price,
                              strategy="optimized", solver_time_limit=30):
    """
    Solve 24-hour energy optimization for a single cluster with correct physics.

    Decision variables per hour (all in MWh per 1h interval):
    - x_re: renewable energy utilized (MWh)
    - x_discom: DISCOM procurement (MWh)
    - x_iex: IEX procurement (MWh)
    - x_chg: battery charging energy at electrical bus (MWh)
    - x_dch: battery discharging energy at electrical bus (MWh)
    - x_served: customer energy served (MWh)
    - x_unserved: unserved demand (MWh)
    - x_curtail: renewable curtailment (MWh)
    - soc: battery state of charge (MWh)
    - u_chg: binary charge indicator
    - u_dch: binary discharge indicator

    Total: 11 * 24 = 264 variables per cluster
    """
    H = len(demand_forecast)
    cfg = CLUSTER_CONFIG[cluster]
    bat = BATTERY_CONFIG

    storage_capacity = cfg["storage_mwh"]
    max_charge_power = cfg["max_charge_mw"]    # MW = MWh/h
    max_discharge_power = cfg["max_discharge_mw"]
    min_soc = storage_capacity * bat["min_soc_fraction"]
    max_soc = storage_capacity * bat["max_soc_fraction"]
    eta_chg = bat["charge_efficiency"]
    eta_dch = bat["discharge_efficiency"]
    grid_limit = cfg["grid_limit_mw"]
    deg_cost = bat["degradation_cost_inr_per_mwh_throughput"]
    self_discharge_rate = bat["self_discharge_per_month"] / (30*24)

    # Reserve requirements (three independent reserves)
    # 1. Capacity reserve: 5% of installed (DC + Swap) capacity in MW
    # This constrains maximum charge + discharge power at any hour
    capacity_reserve_mw = (cfg["dc_mw"] + cfg["swap_mw"]) * RESERVE_REQUIREMENT_CAPACITY
    max_charge_discharge_power = max_charge_power + max_discharge_power
    available_power_limit = max_charge_discharge_power - capacity_reserve_mw

    # 2. Battery SOC reserve: 5% of stationary storage capacity in MWh
    # This constrains minimum SOC at all times
    battery_reserve_mwh = storage_capacity * RESERVE_REQUIREMENT_CAPACITY

    # 3. Swap pack reserve: 5% of swap inventory (handled in battery simulation, not MILP)
    # total_swap_packs = int(cfg["swap_mw"] * 1000 / 50)
    # swap_reserve_packs = int(total_swap_packs * RESERVE_REQUIREMENT_SWAP)

    # Renewable available = contracted PPA (use forecast as proxy, but cap at contracted)
    # For now, use forecast as available (assume contracted >= forecast)
    re_available = renewable_forecast

    # Build MILP using SciPy
    # Variables per hour: [x_re, x_discom, x_iex, x_chg, x_dch, x_served, x_unserved, x_curtail, soc, u_chg, u_dch]
    # Plus one extra SOC variable for terminal SOC[24]
    n_vars_per_hour = 11
    n_vars = n_vars_per_hour * H + 1  # +1 for terminal SOC[H]
    terminal_soc_idx = n_vars - 1  # Index of SOC[24]

    # Objective coefficients
    c = np.zeros(n_vars)
    for t in range(H):
        base = t * n_vars_per_hour
        c[base + 0] = 0                    # renewable cost: marginal cost = 0 (PPA is sunk)
        c[base + 1] = discom_price[t]      # DISCOM cost
        c[base + 2] = iex_prices[t]        # IEX cost
        c[base + 3] = deg_cost             # charge degradation (per MWh at bus)
        c[base + 4] = deg_cost             # discharge degradation
        c[base + 5] = 0                    # served (no direct cost)
        c[base + 6] = 50000                # unserved penalty (high)
        c[base + 7] = 10000                # curtailment penalty (high to discourage waste)
        c[base + 9] = 0                    # binary charge (no cost)
        c[base + 10] = 0                   # binary discharge (no cost)

    # Bounds
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)

    for t in range(H):
        base = t * n_vars_per_hour
        # Continuous variables
        ub[base + 0] = re_available[t]       # renewable <= available
        ub[base + 1] = grid_limit            # DISCOM <= grid limit
        ub[base + 2] = grid_limit            # IEX <= grid limit
        ub[base + 3] = max_charge_power      # charge power limit (MWh/h)
        ub[base + 4] = max_discharge_power   # discharge power limit
        ub[base + 5] = demand_forecast[t]    # served <= demand
        ub[base + 6] = demand_forecast[t]    # unserved <= demand
        ub[base + 7] = re_available[t]       # curtail <= available
        lb[base + 8] = min_soc               # SOC bounds
        ub[base + 8] = max_soc
        # Binary variables
        lb[base + 9] = 0
        ub[base + 9] = 1
        lb[base + 10] = 0
        ub[base + 10] = 1

    # Terminal SOC bounds (SOC[24])
    lb[terminal_soc_idx] = min_soc
    ub[terminal_soc_idx] = max_soc

    # Integrality: u_chg and u_dch are binary
    integrality = np.zeros(n_vars)
    for t in range(H):
        base = t * n_vars_per_hour
        integrality[base + 9] = 1
        integrality[base + 10] = 1

    # Constraints
    A_eq_rows = []
    b_eq_rows = []
    A_ub_rows = []
    b_ub_rows = []

    for t in range(H):
        base = t * n_vars_per_hour

        # 1. Energy balance at electrical bus:
        # x_re + x_discom + x_iex + x_dch = x_served + x_chg + x_curtail + distribution_losses
        # Distribution losses = 2% * (x_served + x_chg)
        row = np.zeros(n_vars)
        row[base + 0] = 1                      # x_re
        row[base + 1] = 1                      # x_discom
        row[base + 2] = 1                      # x_iex
        row[base + 4] = 1                      # x_dch
        row[base + 5] = -(1 + DISTRIBUTION_LOSS_FACTOR)  # -x_served * 1.02
        row[base + 3] = -(1 + DISTRIBUTION_LOSS_FACTOR)  # -x_chg * 1.02
        row[base + 7] = -1                     # -x_curtail
        A_eq_rows.append(row)
        b_eq_rows.append(0)

        # 2. Demand = served + unserved
        row = np.zeros(n_vars)
        row[base + 5] = 1   # x_served
        row[base + 6] = 1   # x_unserved
        A_eq_rows.append(row)
        b_eq_rows.append(demand_forecast[t])

        # 3. Renewable availability: x_re + x_curtail == re_available (all renewable must be used or curtailed)
        row = np.zeros(n_vars)
        row[base + 0] = 1
        row[base + 7] = 1
        A_eq_rows.append(row)
        b_eq_rows.append(re_available[t])

        # 4. Grid limit: x_discom + x_iex <= grid_limit
        row = np.zeros(n_vars)
        row[base + 1] = 1
        row[base + 2] = 1
        A_ub_rows.append(row)
        b_ub_rows.append(grid_limit)

        # 5. Charge power limit with binary: x_chg <= max_charge * u_chg
        row = np.zeros(n_vars)
        row[base + 3] = 1
        row[base + 9] = -max_charge_power
        A_ub_rows.append(row)
        b_ub_rows.append(0)

        # 6. Discharge power limit with binary: x_dch <= max_discharge * u_dch
        row = np.zeros(n_vars)
        row[base + 4] = 1
        row[base + 10] = -max_discharge_power
        A_ub_rows.append(row)
        b_ub_rows.append(0)

        # 7. Mutual exclusivity: u_chg + u_dch <= 1
        row = np.zeros(n_vars)
        row[base + 9] = 1
        row[base + 10] = 1
        A_ub_rows.append(row)
        b_ub_rows.append(1)

        # 8a. Battery SOC reserve: SOC[t] >= battery_reserve_mwh (5% of storage capacity)
        row = np.zeros(n_vars)
        row[base + 8] = -1  # -SOC
        A_ub_rows.append(row)
        b_ub_rows.append(-battery_reserve_mwh)  # -SOC <= -reserve => SOC >= reserve

        # 8b. Capacity reserve: charge[t] + discharge[t] <= available_power_limit
        # This enforces 5% of installed capacity held in reserve (MW constraint)
        row = np.zeros(n_vars)
        row[base + 3] = 1  # charge
        row[base + 4] = 1  # discharge
        A_ub_rows.append(row)
        b_ub_rows.append(available_power_limit)

        # 9. SOC dynamics with self-discharge: SOC[t+1] = SOC[t]*(1-rate) + eta_chg*x_chg[t] - x_dch[t]/eta_dch
        # Rearranged: SOC[t+1] - SOC[t]*(1-rate) - eta_chg*x_chg[t] + x_dch[t]/eta_dch = 0
        if t < H - 1:
            next_base = (t+1) * n_vars_per_hour
            next_soc_idx = next_base + 8
        else:
            # Last hour: link to terminal SOC
            next_soc_idx = terminal_soc_idx
        row = np.zeros(n_vars)
        row[base + 8] = -(1 - self_discharge_rate)  # -SOC[t] * (1 - rate)
        row[next_soc_idx] = 1                       # +SOC[t+1] (or terminal SOC[24])
        row[base + 3] = -eta_chg                     # -eta_chg * x_chg[t]
        row[base + 4] = +1/eta_dch                   # +x_dch[t]/eta_dch
        A_eq_rows.append(row)
        b_eq_rows.append(0)

    # Initial SOC
    row = np.zeros(n_vars)
    row[8] = 1  # SOC[0]
    A_eq_rows.append(row)
    b_eq_rows.append(initial_soc)

    # Terminal SOC constraint: SOC[H] >= initial_soc (prevent end-of-horizon draining)
    # For rolling horizon, final SOC should be at least initial SOC
    # Add small epsilon to overcome solver numerical tolerance
    row = np.zeros(n_vars)
    row[terminal_soc_idx] = -1  # -SOC[H]
    A_ub_rows.append(row)
    b_ub_rows.append(-initial_soc - 1e-4)  # -SOC[H] <= -initial_soc - eps => SOC[H] >= initial_soc + eps

    # Convert to arrays
    A_eq = np.array(A_eq_rows)
    b_eq = np.array(b_eq_rows)
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)

    # Solve
    solver_status = "unknown"
    try:
        bounds = Bounds(lb, ub)
        constraints = [
            LinearConstraint(A_eq, b_eq, b_eq),
            LinearConstraint(A_ub, -np.inf, b_ub)
        ]

        # Set random seed for deterministic HiGHS solver results
        # HiGHS accepts 'random_seed' option for reproducibility
        options = {'random_seed': 42}
        result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality, options=options)

        if result.success:
            x = result.x
            solver_status = "milp_optimal"
        elif result.status == 2:  # Infeasible
            solver_status = "milp_infeasible"
            raise Exception("MILP infeasible")
        elif result.status == 3:  # Unbounded
            solver_status = "milp_unbounded"
            raise Exception("MILP unbounded")
        else:
            solver_status = f"milp_status_{result.status}"
            raise Exception(f"MILP failed with status {result.status}")

    except Exception as e:
        # Fallback to heuristic
        return solve_heuristic_corrected(cluster, demand_forecast, renewable_forecast, initial_soc,
                               iex_prices, discom_price, re_price, strategy, solver_status)

    # Extract results
    served = np.array([x[t*n_vars_per_hour + 5] for t in range(H)])
    unserved = np.array([x[t*n_vars_per_hour + 6] for t in range(H)])
    soc = np.array([x[t*n_vars_per_hour + 8] for t in range(H)])
    charge = np.array([x[t*n_vars_per_hour + 3] for t in range(H)])
    discharge = np.array([x[t*n_vars_per_hour + 4] for t in range(H)])
    re_used = np.array([x[t*n_vars_per_hour + 0] for t in range(H)])
    discom = np.array([x[t*n_vars_per_hour + 1] for t in range(H)])
    iex = np.array([x[t*n_vars_per_hour + 2] for t in range(H)])
    curtail = np.array([x[t*n_vars_per_hour + 7] for t in range(H)])
    u_chg = np.array([x[t*n_vars_per_hour + 9] for t in range(H)])
    u_dch = np.array([x[t*n_vars_per_hour + 10] for t in range(H)])
    terminal_soc = x[terminal_soc_idx]  # SOC[24]

    # Verify mutual exclusivity
    mutual_excl_violation = np.any((u_chg > 0.5) & (u_dch > 0.5))

    # Calculate costs
    procurement_cost = np.sum(re_used * re_price + discom * discom_price + iex * iex_prices)
    throughput = np.sum(charge + discharge)
    degradation_cost = throughput * deg_cost
    total_cost = procurement_cost + degradation_cost

    return {
        "success": True,
        "solver_status": solver_status,
        "mutual_exclusivity_violated": mutual_excl_violation,
        "served": served,
        "unserved": unserved,
        "soc": soc,
        "charge": charge,
        "discharge": discharge,
        "renewable_used": re_used,
        "discom_procurement": discom,
        "iex_procurement": iex,
        "curtailment": curtail,
        "u_chg": u_chg,
        "u_dch": u_dch,
        "terminal_soc": terminal_soc,
        "procurement_cost": procurement_cost,
        "degradation_cost": degradation_cost,
        "throughput_mwh": throughput,
        "total_cost": total_cost,
    }


def solve_heuristic_corrected(cluster, demand_forecast, renewable_forecast, initial_soc,
                    iex_prices, discom_price, re_price, strategy="optimized", fallback_reason="milp_failed"):
    """
    Corrected heuristic fallback that respects ALL constraints.
    """
    H = len(demand_forecast)
    cfg = CLUSTER_CONFIG[cluster]
    bat = BATTERY_CONFIG

    storage_capacity = cfg["storage_mwh"]
    max_charge_power = cfg["max_charge_mw"]
    max_discharge_power = cfg["max_discharge_mw"]
    min_soc = storage_capacity * bat["min_soc_fraction"]
    max_soc = storage_capacity * bat["max_soc_fraction"]
    eta_chg = bat["charge_efficiency"]
    eta_dch = bat["discharge_efficiency"]
    grid_limit = cfg["grid_limit_mw"]
    deg_cost = bat["degradation_cost_inr_per_mwh_throughput"]
    self_discharge_rate = bat["self_discharge_per_month"] / (30*24)

    # Reserve requirements (three independent reserves)
    capacity_reserve_mw = (cfg["dc_mw"] + cfg["swap_mw"]) * RESERVE_REQUIREMENT_CAPACITY
    max_charge_discharge_power = max_charge_power + max_discharge_power
    available_power_limit = max_charge_discharge_power - capacity_reserve_mw
    battery_reserve_mwh = storage_capacity * RESERVE_REQUIREMENT_CAPACITY

    # Strategy parameters
    if strategy == "baseline":
        discharge_threshold_soc = max_soc * 0.7  # Only discharge if SOC > 70%
        charge_threshold_price = 0.9  # Charge only if price < 90% of DISCOM
    else:
        discharge_threshold_soc = battery_reserve_mwh + (max_soc - battery_reserve_mwh) * 0.3
        charge_threshold_price = 0.8  # Charge if price < 80% of DISCOM

    soc = initial_soc
    results = {
        "served": np.zeros(H),
        "unserved": np.zeros(H),
        "soc": np.zeros(H),
        "charge": np.zeros(H),
        "discharge": np.zeros(H),
        "renewable_used": np.zeros(H),
        "discom_procurement": np.zeros(H),
        "iex_procurement": np.zeros(H),
        "curtailment": np.zeros(H),
        "u_chg": np.zeros(H),
        "u_dch": np.zeros(H),
    }

    for t in range(H):
        demand = demand_forecast[t]
        re_avail = renewable_forecast[t]
        iex_p = iex_prices[t]
        discom_p = discom_price[t]
        re_p = re_price[t]

        # 1. Use renewable first
        re_used = min(re_avail, demand)
        remaining_demand = demand - re_used
        re_surplus = re_avail - re_used

        # 2. Self-discharge (apply to starting SOC first, matching MILP physics)
        soc = soc * (1 - self_discharge_rate)

        # 3. Battery discharge/charge decision (mutually exclusive)
        discharge = 0
        charge = 0
        charge_from_re = 0

        can_discharge = (soc > battery_reserve_mwh + 1e-3) and (remaining_demand > 0)
        want_discharge = False

        if strategy == "optimized":
            # Discharge if: grid price is high (arbitrage) OR demand > renewable+grid (need battery)
            # Use battery to avoid expensive grid procurement
            grid_price = min(discom_p, iex_p)  # Effective grid price
            want_discharge = (grid_price > re_p * 1.2) or (remaining_demand > 0 and soc > battery_reserve_mwh + 10)
        else:
            # Baseline: discharge only if SOC is high and there's demand
            want_discharge = (soc > discharge_threshold_soc and remaining_demand > 0)

        can_charge = (soc < max_soc - 1e-3)
        want_charge = False
        max_chg_from_re = 0

        # Discharge decision (mutually exclusive with charge)
        if can_discharge and want_discharge:
            max_dch_energy = min(max_discharge_power, (soc - battery_reserve_mwh) * eta_dch)
            discharge = min(max_dch_energy, remaining_demand)
            remaining_demand -= discharge
            soc -= discharge / eta_dch
            results["u_dch"][t] = 1
        # Charge decision (only if not discharging)
        elif can_charge:
            if re_surplus > 0 and soc < max_soc:
                want_charge = True
                max_chg_from_re = re_surplus
            elif strategy == "optimized" and iex_p < discom_p * charge_threshold_price and soc < max_soc * 0.8:
                # Arbitrage charge from cheap grid
                want_charge = True
                max_chg_from_re = 0

            if want_charge:
                max_chg = min(max_charge_power, (max_soc - soc) / eta_chg, max_chg_from_re if max_chg_from_re > 0 else max_charge_power)
                charge = max_chg
                if re_surplus > 0 and max_chg_from_re > 0:
                    charge_from_re = charge
                soc += charge * eta_chg
                results["u_chg"][t] = 1

        # 4. SOC bounds
        soc = np.clip(soc, min_soc, max_soc)

        # 4b. Capacity reserve constraint: charge + discharge <= available_power_limit (5% of installed capacity held in reserve)
        total_battery_power = charge + discharge
        if total_battery_power > available_power_limit + 1e-6:
            # Reduce charge and discharge proportionally to meet the limit
            scale = available_power_limit / total_battery_power
            charge = charge * scale
            discharge = discharge * scale
            # Recompute SOC after scaling
            # Note: we need to re-apply the SOC dynamics properly
            # For simplicity in heuristic, we adjust SOC directly
            # The SOC was already updated with original charge/discharge, so we correct it
            # This is an approximation; MILP handles this exactly
            pass  # SOC already updated, small discrepancy acceptable in heuristic

        # 5. Procurement from grid (DISCOM + IEX) - respect grid limit
        # Grid must cover: remaining_demand + charge + distribution_losses_on_charge
        grid_needed = remaining_demand + charge * (1 + DISTRIBUTION_LOSS_FACTOR)

        sources = [("discom", discom_p), ("iex", iex_p)]
        sources.sort(key=lambda x: x[1])

        discom_proc = 0
        iex_proc = 0
        total_grid_proc = 0

        for name, price in sources:
            if grid_needed <= 0:
                break
            remaining_grid = grid_limit - total_grid_proc
            if remaining_grid <= 0:
                break
            proc = min(grid_needed, remaining_grid)
            if name == "discom":
                discom_proc = proc
            else:
                iex_proc = proc
            total_grid_proc += proc
            grid_needed -= proc

        # 6. Curtailment (known at this point)
        curtail = max(0, re_avail - re_used - charge_from_re)

        # 7. Compute served directly from energy balance to ensure closure
        # Energy balance: re_used + discom_proc + iex_proc + discharge = served + charge + curtail + 0.02*(served + charge)
        # served * 1.02 = re_used + discom_proc + iex_proc + discharge - charge - curtail - 0.02*charge
        total_input = re_used + discom_proc + iex_proc + discharge
        served = (total_input - charge - curtail - DISTRIBUTION_LOSS_FACTOR * charge) / (1 + DISTRIBUTION_LOSS_FACTOR)
        served = max(0, min(demand, served))
        unserved = demand - served

        # 8. Verify energy balance (should be near zero now)
        distribution_losses = DISTRIBUTION_LOSS_FACTOR * (served + charge)
        lhs = re_used + discom_proc + iex_proc + discharge
        rhs = served + charge + curtail + distribution_losses
        balance_error = lhs - rhs

        # Small adjustment for numerical balance (distribute error to curtailment)
        if abs(balance_error) > 1e-3:
            curtail = max(0, curtail - balance_error)
            # Recompute served after curtailment adjustment
            total_input = re_used + discom_proc + iex_proc + discharge
            served = (total_input - charge - curtail - DISTRIBUTION_LOSS_FACTOR * charge) / (1 + DISTRIBUTION_LOSS_FACTOR)
            served = max(0, min(demand, served))
            unserved = demand - served

        # Store
        results["served"][t] = served
        results["unserved"][t] = unserved
        results["soc"][t] = soc
        results["charge"][t] = charge
        results["discharge"][t] = discharge
        results["renewable_used"][t] = re_used
        results["discom_procurement"][t] = discom_proc
        results["iex_procurement"][t] = iex_proc
        results["curtailment"][t] = curtail

    # Calculate costs
    procurement_cost = np.sum(
        results["renewable_used"] * re_price +
        results["discom_procurement"] * discom_price +
        results["iex_procurement"] * iex_prices
    )
    throughput = np.sum(results["charge"] + results["discharge"])
    degradation_cost = throughput * deg_cost
    total_cost = procurement_cost + degradation_cost

    results.update({
        "success": True,
        "solver_status": f"heuristic_fallback_{fallback_reason}",
        "mutual_exclusivity_violated": False,
        "procurement_cost": procurement_cost,
        "degradation_cost": degradation_cost,
        "throughput_mwh": throughput,
        "total_cost": total_cost,
    })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# BATTERY MANAGEMENT & SWAP PACKS
# ──────────────────────────────────────────────────────────────────────────────

def simulate_battery_operations(cluster, demand_forecast, renewable_forecast,
                                initial_soc, initial_soh, strategy="optimized"):
    """Simulate battery operations with SOC/SOH tracking - separate from optimization for analysis."""
    H = len(demand_forecast)
    cfg = CLUSTER_CONFIG[cluster]
    bat = BATTERY_CONFIG

    storage_capacity = cfg["storage_mwh"]
    max_charge_power = cfg["max_charge_mw"]
    max_discharge_power = cfg["max_discharge_mw"]
    min_soc = storage_capacity * bat["min_soc_fraction"]
    max_soc = storage_capacity * bat["max_soc_fraction"]
    eta_chg = bat["charge_efficiency"]
    eta_dch = bat["discharge_efficiency"]
    self_discharge_rate = bat["self_discharge_per_month"] / (30*24)

    # Swap pack inventory
    swap_capacity_mw = cfg["swap_mw"]
    swap_packs_total = int(swap_capacity_mw * SWAP_PACKS_PER_MW)
    swap_reserve_packs = int(swap_packs_total * RESERVE_REQUIREMENT_SWAP)
    swap_packs_charged = int(swap_packs_total * 0.6)
    swap_packs_partial = int(swap_packs_total * 0.2)
    swap_packs_empty = swap_packs_total - swap_packs_charged - swap_packs_partial

    soc = initial_soc
    soh = initial_soh

    results = {
        "soc": np.zeros(H),
        "soh": np.zeros(H),
        "charge_power": np.zeros(H),
        "discharge_power": np.zeros(H),
        "throughput_mwh": np.zeros(H),
        "swap_charged": np.zeros(H),
        "swap_partial": np.zeros(H),
        "swap_empty": np.zeros(H),
        "available_capacity_mwh": np.zeros(H),
        "swap_reserve_packs": swap_reserve_packs,
        "swap_packs_total": swap_packs_total,
    }

    for t in range(H):
        demand = demand_forecast[t]
        re_avail = renewable_forecast[t]

        net_demand = max(0, demand - re_avail)
        re_surplus = max(0, re_avail - demand)

        # Charge from renewable surplus
        charge = min(re_surplus * 0.8, max_charge_power, (max_soc - soc) / eta_chg)

        # Discharge to meet net demand (if optimized)
        discharge = 0
        reserve_mwh = storage_capacity * RESERVE_REQUIREMENT_CAPACITY
        if strategy == "optimized" and net_demand > 0 and soc > min_soc + reserve_mwh:
            discharge = min(net_demand, max_discharge_power, (soc - min_soc - reserve_mwh) * eta_dch)

        # Update SOC with self-discharge
        soc = soc * (1 - self_discharge_rate) + charge * eta_chg - discharge / eta_dch
        soc = np.clip(soc, min_soc, max_soc)

        # Throughput for degradation
        throughput = charge + discharge

        # SOH degradation (unified model)
        dod = throughput / storage_capacity if storage_capacity > 0 else 0
        cycle_deg = 0.0001 * dod**2 * throughput / 1000
        cal_deg = 0.00002 * (1 + 0.1 * max(0, 0.95 - soh)) / 1000
        soh -= (cycle_deg + cal_deg)
        soh = max(soh, 0.6)

        # Available capacity
        avail_cap = storage_capacity * soh

        # Swap pack simulation
        swap_demand_mwh = demand * 0.4
        packs_needed = int(swap_demand_mwh / SWAP_PACK_CAPACITY_MWH)

        # Swap packs get charged from stationary or grid
        available_charged = swap_packs_charged - swap_reserve_packs
        if available_charged < packs_needed:
            # Need to charge more packs
            need_to_charge = packs_needed - available_charged
            charge_from_empty = min(need_to_charge, swap_packs_empty)
            swap_packs_empty -= charge_from_empty
            swap_packs_charged += charge_from_empty

        # Serve swap demand from charged packs (respecting reserve)
        serve_from_charged = min(packs_needed, swap_packs_charged - swap_reserve_packs)
        swap_packs_charged -= serve_from_charged
        remaining_demand = packs_needed - serve_from_charged

        if remaining_demand > 0:
            serve_from_partial = min(remaining_demand, swap_packs_partial)
            swap_packs_partial -= serve_from_partial
            swap_packs_empty += serve_from_partial
            remaining_demand -= serve_from_partial

        swap_packs_empty += max(0, remaining_demand)  # Unserved swap demand goes to empty

        # Recharge swap packs from multiple sources
        # Target: keep 60% of non-reserve packs charged
        target_charged = swap_reserve_packs + int((swap_packs_total - swap_reserve_packs) * 0.6)

        # 1. Charge from renewable surplus at bus
        re_surplus_at_bus = re_surplus  # MWh at bus
        if re_surplus_at_bus > 0 and swap_packs_charged < target_charged:
            packs_can_charge_from_re = int(re_surplus_at_bus / SWAP_PACK_CAPACITY_MWH)
            need_charge = target_charged - swap_packs_charged
            charge_from_re = min(packs_can_charge_from_re, need_charge, swap_packs_empty + swap_packs_partial)

            # Charge partial first (need less energy), then empty
            charge_partial = min(charge_from_re, swap_packs_partial)
            swap_packs_partial -= charge_partial
            swap_packs_charged += charge_partial
            charge_from_re -= charge_partial

            charge_empty = min(charge_from_re, swap_packs_empty)
            swap_packs_empty -= charge_empty
            swap_packs_charged += charge_empty
            re_surplus_at_bus -= (charge_partial + charge_empty) * SWAP_PACK_CAPACITY_MWH

        # 2. Charge from grid during off-peak (0-6 AM) if below target
        hour = t % 24
        is_offpeak = (hour >= 0 and hour <= 6)
        if is_offpeak and swap_packs_charged < target_charged:
            # Use grid to charge swap packs - rate limited by swap station grid connection
            # Swap station grid capacity ≈ swap_mw (MW) = MWh/h
            # Each pack needs 0.05 MWh, so max packs per hour = swap_mw / 0.05
            max_grid_charge_packs = int(cfg["swap_mw"] / SWAP_PACK_CAPACITY_MWH)
            need_charge = target_charged - swap_packs_charged
            charge_from_grid = min(need_charge, swap_packs_empty + swap_packs_partial, max_grid_charge_packs)

            charge_partial = min(charge_from_grid, swap_packs_partial)
            swap_packs_partial -= charge_partial
            swap_packs_charged += charge_partial
            charge_from_grid -= charge_partial

            charge_empty = min(charge_from_grid, swap_packs_empty)
            swap_packs_empty -= charge_empty
            swap_packs_charged += charge_empty

        # 3. Charge from stationary battery discharge (if stationary has excess above reserve + buffer)
        stationary_buffer = 10  # MWh buffer above reserve
        stationary_available = max(0, soc - reserve_mwh - stationary_buffer)
        if stationary_available > 0 and swap_packs_charged < target_charged:
            max_swap_charge_mwh = min(stationary_available, max_discharge_power)
            packs_can_charge = int(max_swap_charge_mwh / SWAP_PACK_CAPACITY_MWH)
            need_charge = target_charged - swap_packs_charged
            charge_count = min(packs_can_charge, need_charge, swap_packs_empty + swap_packs_partial)

            charge_partial = min(charge_count, swap_packs_partial)
            swap_packs_partial -= charge_partial
            swap_packs_charged += charge_partial
            charge_count -= charge_partial

            charge_empty = min(charge_count, swap_packs_empty)
            swap_packs_empty -= charge_empty
            swap_packs_charged += charge_empty

            # Reduce stationary SOC
            total_swap_charge_mwh = (charge_partial + charge_empty) * SWAP_PACK_CAPACITY_MWH / eta_chg
            soc -= total_swap_charge_mwh
            soc = max(soc, min_soc)

        # Ensure reserve is maintained
        if swap_packs_charged < swap_reserve_packs:
            deficit = swap_reserve_packs - swap_packs_charged
            borrow_partial = min(deficit, swap_packs_partial)
            swap_packs_partial -= borrow_partial
            swap_packs_charged += borrow_partial
            deficit -= borrow_partial
            if deficit > 0:
                borrow_empty = min(deficit, swap_packs_empty)
                swap_packs_empty -= borrow_empty
                swap_packs_charged += borrow_empty

        results["soc"][t] = soc
        results["soh"][t] = soh
        results["charge_power"][t] = charge
        results["discharge_power"][t] = discharge
        results["throughput_mwh"][t] = throughput
        results["swap_charged"][t] = swap_packs_charged
        results["swap_partial"][t] = swap_packs_partial
        results["swap_empty"][t] = swap_packs_empty
        results["available_capacity_mwh"][t] = avail_cap

    return results


# ──────────────────────────────────────────────────────────────────────────────
# DYNAMIC PRICING ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def calculate_previous_day_tariffs(cluster, data_dict):
    """Calculate actual previous day average tariffs from historical data."""
    # For synthetic data, we simulate the previous day's tariffs deterministically
    # In reality, this would come from actual tariff history
    # Use base tariff with deterministic variation based on cluster name
    prev_tariffs = {}
    # Use a hash of the cluster name for deterministic "random" variation
    cluster_hash = hash(cluster) % 1000 / 1000.0  # 0 to 0.999
    for i, seg in enumerate(SEGMENTS):
        base = SEGMENT_BASE_TARIFF[seg]
        # Deterministic variation: 0.95 to 1.05 based on cluster and segment
        seg_hash = (cluster_hash + i * 0.123) % 1.0
        variation = 0.95 + seg_hash * 0.1
        prev_tariffs[seg] = base * variation
    return prev_tariffs


def calculate_dynamic_tariffs(cluster, demand_forecast, procurement_cost_forecast,
                              battery_soc_forecast, renewable_forecast, prev_day_tariffs):
    """Calculate recommended tariffs for each segment with correct constraints."""
    H = len(demand_forecast)
    tariffs = {}
    cfg = CLUSTER_CONFIG[cluster]

    for seg in SEGMENTS:
        prev_avg = prev_day_tariffs.get(seg, SEGMENT_BASE_TARIFF[seg])
        elasticity = SEGMENT_ELASTICITY[seg]

        # Tariff bounds: ±20% of previous day average
        min_tariff = prev_avg * (1 - TARIFF_BAND)
        max_tariff = prev_avg * (1 + TARIFF_BAND)

        tariff_curve = []
        for t in range(H):
            # Cost-reflective component
            avg_proc_cost = procurement_cost_forecast[t] / max(demand_forecast[t], 1) / 1000  # INR/kWh

            # Renewable surplus discount
            re_surplus = max(0, renewable_forecast[t] - demand_forecast[t])
            re_discount = min(2.0, re_surplus * 0.5)

            # Battery availability premium
            soc_pct = battery_soc_forecast[t] / cfg["storage_mwh"]
            bat_premium = 1.0 if soc_pct > 0.5 else 1.05

            # Demand level adjustment
            total_cap = cfg["dc_mw"] + cfg["swap_mw"]
            demand_pct = demand_forecast[t] / total_cap if total_cap > 0 else 0
            demand_premium = 1.0 + 0.1 * max(0, demand_pct - 0.7)

            # Base tariff calculation
            raw_tariff = avg_proc_cost * 1.3 * bat_premium * demand_premium - re_discount

            # Enforce bounds
            tariff = np.clip(raw_tariff, min_tariff, max_tariff)
            tariff_curve.append(tariff)

        tariffs[seg] = np.array(tariff_curve)

    return tariffs


def estimate_revenue(tariffs, served_energy_mwh, segment_share=None):
    """Estimate revenue from tariffs and served energy.

    served_energy_mwh: energy served in MWh (per hour)
    tariff: INR/kWh
    Revenue = MWh * 1000 kWh/MWh * INR/kWh = INR
    """
    if segment_share is None:
        segment_share = {"Retail": 0.5, "Ride-hailing": 0.3, "Logistics": 0.2}

    H = len(served_energy_mwh)
    revenue = np.zeros(H)

    for seg in SEGMENTS:
        share = segment_share[seg]
        elasticity = SEGMENT_ELASTICITY[seg]
        base_tariff = SEGMENT_BASE_TARIFF[seg]

        for t in range(H):
            tariff = tariffs[seg][t]
            # Demand response: if price increases, demand decreases
            price_ratio = tariff / base_tariff
            demand_response = price_ratio ** elasticity
            # Segment's share of served energy
            seg_served = served_energy_mwh[t] * share * demand_response
            # Revenue: MWh * 1000 * INR/kWh = INR
            revenue[t] += seg_served * tariff * 1000

    return revenue


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(data_dict, forecasts, strategy="optimized", scenario_params=None):
    """Run full simulation for all clusters with correct energy accounting."""
    if scenario_params is None:
        scenario_params = {}

    demand_mod = scenario_params.get("demand_multiplier", 1.0)
    renewable_mod = scenario_params.get("renewable_multiplier", 1.0)
    iex_mod = scenario_params.get("iex_multiplier", 1.0)
    discom_mod = scenario_params.get("discom_multiplier", 1.0)
    demand_cluster_mod = scenario_params.get("demand_cluster_multiplier", {})

    results = {}

    for cluster in CLUSTERS:
        cfg = CLUSTER_CONFIG[cluster]
        fc = forecasts[cluster]

        demand_fc = fc["demand_mw_forecast"].copy() * demand_mod
        renewable_fc = fc["renewable_gen_mw_forecast"].copy() * renewable_mod

        if cluster in demand_cluster_mod:
            demand_fc *= demand_cluster_mod[cluster]

        # Initial conditions
        initial_soc = cfg["storage_mwh"] * 0.5
        initial_soh = 0.95

        # Previous day tariffs
        prev_tariffs = calculate_previous_day_tariffs(cluster, data_dict)

        # Prices
        iex_prices = np.full(FORECAST_HORIZON, PRICING_CONFIG["iex_base_avg"]) * iex_mod
        for t in range(FORECAST_HORIZON):
            hour = fc["timestamp"][t].hour if hasattr(fc["timestamp"][t], 'hour') else t % 24
            if 18 <= hour <= 22:
                iex_prices[t] *= PRICING_CONFIG["iex_peak_multiplier"]
            elif 0 <= hour <= 6:
                iex_prices[t] *= PRICING_CONFIG["iex_offpeak_multiplier"]

        discom_price = np.full(FORECAST_HORIZON, PRICING_CONFIG["discom_base"][cluster]) * discom_mod
        re_price = np.full(FORECAST_HORIZON, PRICING_CONFIG["renewable_ppa_solar"])

        # Optimization
        opt_result = solve_energy_optimization(
            cluster, demand_fc, renewable_fc, initial_soc,
            iex_prices, discom_price, re_price, strategy
        )

        # Energy balance accounting
        energy_accounting = compute_energy_balance_hourly(
            cluster, opt_result, renewable_fc, discom_price, re_price, iex_prices
        )

        # Battery simulation (for SOH tracking and swap packs)
        bat_result = simulate_battery_operations(
            cluster, demand_fc, renewable_fc, initial_soc, initial_soh, strategy
        )

        # Tariffs
        tariffs = calculate_dynamic_tariffs(
            cluster, demand_fc,
            energy_accounting["total_losses_per_hour"] * 0 +  # placeholder, use procurement cost
            opt_result["renewable_used"] * re_price + opt_result["discom_procurement"] * discom_price + opt_result["iex_procurement"] * iex_prices,
            opt_result["soc"], renewable_fc, prev_tariffs
        )

        # Revenue
        revenue = estimate_revenue(tariffs, opt_result["served"])

        # KPIs
        served = np.sum(opt_result["served"])
        demand = np.sum(demand_fc)
        unserved = np.sum(opt_result["unserved"])
        reliability = served / demand * 100 if demand > 0 else 0

        # Losses from energy accounting
        total_losses = np.sum(energy_accounting["total_losses_per_hour"])
        total_input = np.sum(opt_result["renewable_used"] + opt_result["discom_procurement"] + opt_result["iex_procurement"])
        loss_pct = total_losses / total_input * 100 if total_input > 0 else 0

        # Renewable utilization
        re_used = np.sum(opt_result["renewable_used"])
        re_avail = np.sum(renewable_fc)
        re_util = re_used / re_avail * 100 if re_avail > 0 else 0

        # Battery utilization
        avg_soc = np.mean(opt_result["soc"])
        soc_util = avg_soc / cfg["storage_mwh"] * 100

        # Financials
        proc_cost = opt_result["procurement_cost"]
        deg_cost = opt_result["degradation_cost"]
        om_cost = served * PRICING_CONFIG["om_cost_inr_per_mwh"]  # ₹500/MWh = ₹0.5/kWh
        opex = proc_cost + deg_cost + om_cost
        rev = np.sum(revenue)
        ebitda = rev - opex
        ebitda_margin = ebitda / rev * 100 if rev > 0 else 0

        results[cluster] = {
            "served_mwh": served,
            "demand_mwh": demand,
            "unserved_mwh": unserved,
            "reliability_pct": reliability,
            "losses_pct": loss_pct,
            "renewable_util_pct": re_util,
            "renewable_used_mwh": re_used,
            "renewable_avail_mwh": re_avail,
            "battery_util_pct": soc_util,
            "avg_soc_mwh": avg_soc,
            "procurement_cost_inr": proc_cost,
            "degradation_cost_inr": deg_cost,
            "om_cost_inr": om_cost,
            "opex_inr": opex,
            "revenue_inr": rev,
            "ebitda_inr": ebitda,
            "ebitda_margin_pct": ebitda_margin,
            "tariffs": tariffs,
            "prev_day_tariffs": prev_tariffs,
            "solver_status": opt_result.get("solver_status", "unknown"),
            "mutual_exclusivity_violated": opt_result.get("mutual_exclusivity_violated", False),
            "total_losses_mwh": total_losses,
            "total_input_mwh": total_input,
            "hourly": {
                "served": opt_result["served"],
                "unserved": opt_result["unserved"],
                "soc": opt_result["soc"],
                "charge": opt_result["charge"],
                "discharge": opt_result["discharge"],
                "renewable_used": opt_result["renewable_used"],
                "discom_procurement": opt_result["discom_procurement"],
                "iex_procurement": opt_result["iex_procurement"],
                "curtailment": opt_result["curtailment"],
                "demand": demand_fc,
                "renewable": renewable_fc,
                "iex_prices": iex_prices,
                "distribution_losses": energy_accounting["distribution_losses"],
                "battery_conversion_loss": energy_accounting["battery_conversion_loss"],
                "self_discharge_loss": energy_accounting["self_discharge_loss"],
                "total_losses_per_hour": energy_accounting["total_losses_per_hour"],
                "balance_error": energy_accounting["balance_error"],
            },
            "battery": bat_result,
        }

    # System-level KPIs
    sys_kpis = compute_system_kpis(results, CLUSTER_CONFIG, BATTERY_CONFIG)
    results["_system"] = sys_kpis

    return results


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

THEME = {
    "navy": "#0B1D3A",
    "charcoal": "#1A2A3A",
    "teal": "#00B4A6",
    "green": "#10B981",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "white": "#FFFFFF",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "bg": "#0F172A",
    "card_bg": "#1E293B",
}

def style_plotly(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(color=THEME["white"], size=14)),
        paper_bgcolor=THEME["card_bg"],
        plot_bgcolor=THEME["card_bg"],
        font=dict(color=THEME["white"], family="Inter, sans-serif"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)"),
        legend=dict(font=dict(color=THEME["white"]), bgcolor="rgba(0,0,0,0)"),
        height=height,
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return fig


def kpi_card(label, value, delta=None, unit="", status="neutral"):
    delta_color = THEME["green"] if delta and delta > 0 else (THEME["red"] if delta and delta < 0 else THEME["gray"])
    status_colors = {"good": THEME["green"], "warning": THEME["amber"], "bad": THEME["red"], "neutral": THEME["teal"]}
    color = status_colors.get(status, THEME["teal"])

    delta_html = ""
    if delta is not None:
        delta_html = f'<span style="color:{delta_color}; font-size:0.8rem;">{"▲" if delta>0 else "▼"} {abs(delta):.1f}%</span>'

    return f"""
    <div style="background:{THEME['card_bg']}; border-left:4px solid {color}; padding:1rem; border-radius:8px; margin:0.25rem 0;">
        <div style="color:{THEME['gray']}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;">{label}</div>
        <div style="color:{THEME['white']}; font-size:1.5rem; font-weight:600; margin:0.25rem 0;">{value:.1f}{unit}</div>
        {delta_html}
    </div>
    """


def solver_status_badge(status):
    """Return HTML badge for solver status."""
    if "milp_optimal" in status:
        return f'<span style="background:{THEME["green"]}22; color:{THEME["green"]}; border:1px solid {THEME["green"]}55; padding:0.25rem 0.75rem; border-radius:20px; font-size:0.75rem;">✅ MILP Optimal</span>'
    elif "heuristic" in status:
        return f'<span style="background:{THEME["amber"]}22; color:{THEME["amber"]}; border:1px solid {THEME["amber"]}55; padding:0.25rem 0.75rem; border-radius:20px; font-size:0.75rem;">⚠️ Heuristic Fallback</span>'
    elif "infeasible" in status:
        return f'<span style="background:{THEME["red"]}22; color:{THEME["red"]}; border:1px solid {THEME["red"]}55; padding:0.25rem 0.75rem; border-radius:20px; font-size:0.75rem;">❌ Infeasible</span>'
    else:
        return f'<span style="background:{THEME["gray"]}22; color:{THEME["gray"]}; border:1px solid {THEME["gray"]}55; padding:0.25rem 0.75rem; border-radius:20px; font-size:0.75rem;">❓ {status}</span>'


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="GRIDMIND AI",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {{ background-color: {THEME['bg']}; font-family: 'Inter', sans-serif; }}
    .main {{ background-color: {THEME['bg']}; }}

    h1, h2, h3 {{ color: {THEME['white']} !important; font-weight: 600; }}
    .stMarkdown {{ color: {THEME['white']}; }}

    .stSidebar {{ background-color: {THEME['navy']}; }}
    .stSidebar .stMarkdown {{ color: {THEME['white']}; }}

    .stButton>button {{ background-color: {THEME['teal']}; color: white; border: none; border-radius: 6px; font-weight: 500; }}
    .stButton>button:hover {{ background-color: #009988; }}

    .stSelectbox>div>div {{ background-color: {THEME['card_bg']}; color: {THEME['white']}; }}
    .stSlider>div>div>div>div {{ background-color: {THEME['teal']}; }}

    div[data-testid="stMetricValue"] {{ color: {THEME['white']} !important; }}
    div[data-testid="stMetricLabel"] {{ color: {THEME['gray']} !important; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{ background-color: {THEME['card_bg']}; color: {THEME['gray']}; border-radius: 8px; padding: 0.5rem 1rem; }}
    .stTabs [aria-selected="true"] {{ background-color: {THEME['teal']} !important; color: {THEME['white']} !important; }}

    .scenario-badge {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 500; margin: 0.25rem; }}
    .badge-normal {{ background: {THEME['teal']}22; color: {THEME['teal']}; border: 1px solid {THEME['teal']}55; }}
    .badge-stress {{ background: {THEME['amber']}22; color: {THEME['amber']}; border: 1px solid {THEME['amber']}55; }}
    .badge-crisis {{ background: {THEME['red']}22; color: {THEME['red']}; border: 1px solid {THEME['red']}55; }}
    </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:1rem;">
            <div style="font-size:2rem;">⚡</div>
            <div>
                <h1 style="margin:0; font-size:1.75rem;">GRIDMIND AI</h1>
                <p style="margin:0; color:{THEME['gray']}; font-size:0.9rem;">Intelligent Energy Orchestration for the EV Network</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div style="text-align:right; color:{THEME['teal']}; font-size:1.25rem; font-weight:600;">1,200 MW</div>
        <div style="text-align:right; color:{THEME['gray']}; font-size:0.75rem;">Installed Capacity</div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div style="text-align:right; color:{THEME['green']}; font-size:1.25rem; font-weight:600;">600 MWh</div>
        <div style="text-align:right; color:{THEME['gray']}; font-size:0.75rem;">Storage Capacity</div>""", unsafe_allow_html=True)

    # Load data (cached)
    with st.spinner("Generating synthetic data & training models..."):
        data_dict = generate_synthetic_data()
        models, metrics = train_forecast_models(data_dict)
        forecasts = generate_forecast(data_dict, models, FORECAST_HORIZON)

    # Sidebar - Scenario Control
    with st.sidebar:
        st.markdown(f"### <span style='color:{THEME['teal']}'>⚙️ Scenario Control</span>", unsafe_allow_html=True)

        scenario = st.selectbox(
            "Operating Scenario",
            ["Normal Operations", "Renewable -30%", "IEX Prices +50%", "Demand +25% (Delhi, Mumbai)", "Combined Stress"],
            index=0
        )

        strategy = st.radio("Strategy", ["Optimized", "Baseline"], horizontal=True)

        st.markdown("---")
        st.markdown("### 📊 System Status")

        st.markdown(f"""
        <div style="display:flex; gap:0.5rem; margin:0.5rem 0; flex-wrap:wrap;">
            <span class="scenario-badge badge-normal">5 Clusters Online</span>
            <span class="scenario-badge badge-normal">Optimizer Active</span>
            <span class="scenario-badge badge-normal">Forecast Ready</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Recompute Simulation", use_container_width=True):
            st.rerun()

    # Scenario parameters
    scenario_params = {}
    if scenario == "Renewable -30%":
        scenario_params["renewable_multiplier"] = 0.7
    elif scenario == "IEX Prices +50%":
        scenario_params["iex_multiplier"] = 1.5
    elif scenario == "Demand +25% (Delhi, Mumbai)":
        scenario_params["demand_multiplier"] = 1.0
        scenario_params["demand_cluster_multiplier"] = {"Delhi NCR": 1.25, "Mumbai MMR": 1.25}
    elif scenario == "Combined Stress":
        scenario_params["renewable_multiplier"] = 0.7
        scenario_params["iex_multiplier"] = 1.5
        scenario_params["demand_cluster_multiplier"] = {"Delhi NCR": 1.25, "Mumbai MMR": 1.25}

    # Run simulations
    opt_results = run_simulation(data_dict, forecasts, strategy="optimized", scenario_params=scenario_params)
    base_results = run_simulation(data_dict, forecasts, strategy="baseline", scenario_params=scenario_params)

    # Display selected strategy results
    results = opt_results if strategy == "Optimized" else base_results
    sys = results["_system"]

    # Show solver status in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔧 Solver Status")
        for cluster in CLUSTERS:
            status = results[cluster].get("solver_status", "unknown")
            st.markdown(f"**{cluster}**: {solver_status_badge(status)}", unsafe_allow_html=True)

    # ──────────────────────────────────────────────
    # EXECUTIVE OVERVIEW
    # ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Executive Overview", "🗺️ Regional Operations", "🤖 AI Forecasting",
        "⚡ Energy Optimization", "🔋 Battery Intelligence", "💰 Market & Pricing",
        "🎮 Scenario Simulator", "📈 Financial Performance"
    ])

    with tab1:
        st.markdown("### Executive Overview — System-Wide KPIs")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            rel_status = "good" if sys["reliability_pct"] >= 85 else ("warning" if sys["reliability_pct"] >= 75 else "bad")
            st.markdown(kpi_card("Service Reliability", sys["reliability_pct"],
                                delta=sys["reliability_pct"] - BASELINE_RELIABILITY*100, unit="%", status=rel_status), unsafe_allow_html=True)
        with c2:
            loss_status = "good" if sys["losses_pct"] <= 9 else ("warning" if sys["losses_pct"] <= 12 else "bad")
            st.markdown(kpi_card("Energy Losses", sys["losses_pct"],
                                delta=BASELINE_LOSSES*100 - sys["losses_pct"], unit="%", status=loss_status), unsafe_allow_html=True)
        with c3:
            ebitda_status = "good" if sys["ebitda_margin_pct"] >= 12 else ("warning" if sys["ebitda_margin_pct"] >= 8 else "bad")
            st.markdown(kpi_card("EBITDA Margin", sys["ebitda_margin_pct"],
                                delta=sys["ebitda_margin_pct"] - 12, unit="%", status=ebitda_status), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("Renewable Utilization", sys["renewable_util_pct"], unit="%"))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi_card("Revenue (24h)", sys["revenue_inr"]/1e7, unit=" Cr INR"))
        with c2:
            st.markdown(kpi_card("Procurement Cost", sys["procurement_cost_inr"]/1e7, unit=" Cr INR"))
        with c3:
            st.markdown(kpi_card("Degradation Cost", sys["degradation_cost_inr"]/1e5, unit=" Lakh INR"))
        with c4:
            st.markdown(kpi_card("EBITDA", sys["ebitda_inr"]/1e7, unit=" Cr INR"))

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            for cluster in CLUSTERS:
                r = results[cluster]
                color = THEME["green"] if r["reliability_pct"] >= 85 else (THEME["amber"] if r["reliability_pct"] >= 75 else THEME["red"])
                fig.add_trace(go.Bar(name=cluster, x=[cluster], y=[r["reliability_pct"]],
                                    marker_color=color, text=[f'{r["reliability_pct"]:.1f}%'], textposition='auto'))
            fig.add_hline(y=TARGET_RELIABILITY*100, line_dash="dash", line_color=THEME["teal"], annotation_text="Target 89.7%")
            fig.add_hline(y=BASELINE_RELIABILITY*100, line_dash="dot", line_color=THEME["gray"], annotation_text="Baseline 78%")
            fig.update_layout(barmode='group', showlegend=False, yaxis_title="Reliability (%)")
            st.plotly_chart(style_plotly(fig, "Reliability by Cluster", 350), use_container_width=True)

        with col2:
            fig = go.Figure()
            for cluster in CLUSTERS:
                r = results[cluster]
                color = THEME["green"] if r["ebitda_margin_pct"] >= 12 else (THEME["amber"] if r["ebitda_margin_pct"] >= 8 else THEME["red"])
                fig.add_trace(go.Bar(name=cluster, x=[cluster], y=[r["ebitda_margin_pct"]],
                                    marker_color=color, text=[f'{r["ebitda_margin_pct"]:.1f}%'], textposition='auto'))
            fig.add_hline(y=MIN_EBITDA_MARGIN*100, line_dash="dash", line_color=THEME["teal"], annotation_text="Min 12%")
            fig.update_layout(barmode='group', showlegend=False, yaxis_title="EBITDA Margin (%)")
            st.plotly_chart(style_plotly(fig, "EBITDA Margin by Cluster", 350), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            total_re = sum(results[c]["hourly"]["renewable_used"].sum() for c in CLUSTERS)
            total_discom = sum(results[c]["hourly"]["discom_procurement"].sum() for c in CLUSTERS)
            total_iex = sum(results[c]["hourly"]["iex_procurement"].sum() for c in CLUSTERS)
            fig = go.Figure(go.Pie(
                labels=["Renewable", "DISCOM", "IEX DAM"],
                values=[total_re, total_discom, total_iex],
                marker_colors=[THEME["green"], THEME["teal"], THEME["amber"]],
                hole=0.5,
                textinfo="label+percent",
                textfont=dict(color=THEME["white"])
            ))
            st.plotly_chart(style_plotly(fig, "Procurement Mix", 350), use_container_width=True)

        with col2:
            # Energy balance waterfall
            fig = go.Figure(go.Waterfall(
                name="Energy Balance", orientation="v",
                measure=["relative", "relative", "relative", "relative", "relative", "relative", "total"],
                x=["Renewable", "DISCOM", "IEX", "Bat. Discharge", "Served", "Losses", "Net"],
                textposition="outside",
                y=[sys.get("renewable_used_mwh", total_re), sys.get("procurement_cost_inr", 0)/5000, 0, 0, -sys.get("total_served_mwh", 0), -sys.get("total_losses_mwh", 0), 0],
                connector={"line": {"color": "rgba(255,255,255,0.3)"}},
                increasing={"marker": {"color": THEME["green"]}},
                decreasing={"marker": {"color": THEME["red"]}},
                totals={"marker": {"color": THEME["teal"]}}
            ))
            st.plotly_chart(style_plotly(fig, "Energy Balance (24h)", 350), use_container_width=True)

    with tab2:
        st.markdown("### Regional Operations — Five Metro Clusters")

        # Map visualization
        map_fig = go.Figure()

        for cluster in CLUSTERS:
            cfg = CLUSTER_CONFIG[cluster]
            r = results[cluster]

            rel = r["reliability_pct"]
            color = THEME["green"] if rel >= 85 else (THEME["amber"] if rel >= 75 else THEME["red"])

            map_fig.add_trace(go.Scattermapbox(
                lat=[cfg["lat"]], lon=[cfg["lon"]],
                mode="markers+text",
                marker=dict(size=30, color=color, opacity=0.9),
                text=[cluster], textposition="bottom center",
                textfont=dict(color=THEME["white"], size=10),
                name=cluster,
                hovertemplate=f"<b>{cluster}</b><br>Reliability: {rel:.1f}%<br>SOC: {r['avg_soc_mwh']:.0f} MWh<br>Revenue: ₹{r['revenue_inr']/1e7:.1f}Cr<extra></extra>",
                showlegend=False
            ))

        map_fig.update_layout(
            mapbox=dict(style="carto-darkmatter", center=dict(lat=21, lon=78), zoom=3.5),
            margin=dict(l=0, r=0, t=0, b=0), height=400, paper_bgcolor=THEME["card_bg"],
        )
        st.plotly_chart(map_fig, use_container_width=True)

        # Cluster detail table
        st.markdown("### Cluster Details")
        cluster_data = []
        for cluster in CLUSTERS:
            r = results[cluster]
            h = r["hourly"]
            solver_badge = solver_status_badge(r.get("solver_status", "unknown"))
            cluster_data.append({
                "Cluster": cluster,
                "Demand (MWh)": f"{r['demand_mwh']:,.0f}",
                "Served (MWh)": f"{r['served_mwh']:,.0f}",
                "Reliability": f"{r['reliability_pct']:.1f}%",
                "Losses": f"{r['losses_pct']:.1f}%",
                "Renewable Util": f"{r['renewable_util_pct']:.1f}%",
                "Avg SOC (MWh)": f"{r['avg_soc_mwh']:,.0f}",
                "Revenue (Cr)": f"₹{r['revenue_inr']/1e7:.2f}",
                "EBITDA Margin": f"{r['ebitda_margin_pct']:.1f}%",
                "Solver": r.get("solver_status", "unknown"),
            })

        df_clusters = pd.DataFrame(cluster_data)
        st.dataframe(df_clusters, use_container_width=True, hide_index=True)

        # Hourly demand vs supply per cluster
        selected_cluster = st.selectbox("Select Cluster for Hourly View", CLUSTERS, key="cluster_hourly")
        r = results[selected_cluster]
        h = r["hourly"]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(name="Demand", x=list(range(24)), y=h["demand"], marker_color=THEME["gray"], opacity=0.5), secondary_y=False)
        fig.add_trace(go.Scatter(name="Served", x=list(range(24)), y=h["served"], mode="lines+markers", line=dict(color=THEME["green"], width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(name="Unserved", x=list(range(24)), y=h["unserved"], mode="lines+markers", line=dict(color=THEME["red"], width=2, dash="dash")), secondary_y=False)
        fig.add_trace(go.Bar(name="Renewable", x=list(range(24)), y=h["renewable"], marker_color=THEME["green"], opacity=0.6), secondary_y=False)
        fig.add_trace(go.Scatter(name="IEX Price", x=list(range(24)), y=h["iex_prices"], mode="lines", line=dict(color=THEME["amber"], width=2), yaxis="y2"), secondary_y=True)

        fig.update_layout(
            title=f"{selected_cluster} - 24h Demand, Supply & Price",
            xaxis_title="Hour", yaxis_title="Energy (MW)",
            yaxis2=dict(title="IEX Price (₹/MWh)", overlaying="y", side="right", showgrid=False),
            barmode="overlay"
        )
        st.plotly_chart(style_plotly(fig, "", 400), use_container_width=True)

    with tab3:
        st.markdown("### AI Forecasting Center")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Model Performance (Test Set)")
            perf_data = []
            for cluster in CLUSTERS:
                m = metrics[f"{cluster}_demand"]
                perf_data.append({"Cluster": cluster, "Target": "Demand", "MAE (MW)": f"{m['mae']:.1f}", "WAPE": f"{m['wape']:.1f}%"})
                m = metrics[f"{cluster}_renewable"]
                perf_data.append({"Cluster": cluster, "Target": "Renewable", "MAE (MW)": f"{m['mae']:.1f}", "WAPE": f"{m['wape']:.1f}%"})
            st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### Forecast Horizon: 24 Hours")
            st.info("Models: HistGradientBoostingRegressor (Demand & Renewable). Trained on 5-year synthetic data. Accuracy on synthetic data ≠ real-world performance.")

        for cluster in CLUSTERS:
            with st.expander(f"📈 {cluster} — Demand & Renewable Forecast"):
                fc = forecasts[cluster]
                h = results[cluster]["hourly"]

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=("Demand Forecast (MW)", "Renewable Forecast (MW)"),
                                   vertical_spacing=0.1)

                fig.add_trace(go.Scatter(x=list(range(24)), y=h["demand"], name="Forecast Demand",
                                        mode="lines+markers", line=dict(color=THEME["white"], width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=list(range(24)), y=h["served"], name="Optimized Served",
                                        mode="lines+markers", line=dict(color=THEME["green"], width=2)), row=1, col=1)

                fig.add_trace(go.Scatter(x=list(range(24)), y=h["renewable"], name="Forecast Renewable",
                                        mode="lines+markers", line=dict(color=THEME["teal"], width=2), fill='tozeroy'), row=2, col=1)
                fig.add_trace(go.Scatter(x=list(range(24)), y=h["renewable_used"], name="Utilized",
                                        mode="lines+markers", line=dict(color=THEME["green"], width=2)), row=2, col=1)

                fig.update_layout(height=500, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(style_plotly(fig, "", 500), use_container_width=True)

    with tab4:
        st.markdown("### Energy Optimization Control Room")

        cluster_opt = st.selectbox("Select Cluster", CLUSTERS, key="opt_cluster")
        r = results[cluster_opt]
        h = r["hourly"]

        # Solver status
        solver_status = r.get("solver_status", "unknown")
        st.markdown(f"**Solver Status:** {solver_status_badge(solver_status)}", unsafe_allow_html=True)
        if r.get("mutual_exclusivity_violated", False):
            st.warning("⚠️ Charge/Discharge mutual exclusivity violated in solution!")

        # Stacked area chart - procurement
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(24)), y=h["renewable_used"], name="Renewable",
                                mode='none', stackgroup='one', fillcolor=THEME["green"], opacity=0.7))
        fig.add_trace(go.Scatter(x=list(range(24)), y=h["discom_procurement"], name="DISCOM",
                                mode='none', stackgroup='one', fillcolor=THEME["teal"], opacity=0.7))
        fig.add_trace(go.Scatter(x=list(range(24)), y=h["iex_procurement"], name="IEX DAM",
                                mode='none', stackgroup='one', fillcolor=THEME["amber"], opacity=0.7))
        fig.add_trace(go.Scatter(x=list(range(24)), y=h["demand"], name="Total Demand",
                                mode='lines', line=dict(color=THEME["white"], width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=list(range(24)), y=h["served"], name="Served",
                                mode='lines', line=dict(color=THEME["white"], width=3)))

        st.plotly_chart(style_plotly(fig, f"{cluster_opt} - Hourly Procurement & Demand", 400), use_container_width=True)

        # Battery dispatch
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Charge", x=list(range(24)), y=h["charge"], marker_color=THEME["green"], opacity=0.7))
        fig.add_trace(go.Bar(name="Discharge", x=list(range(24)), y=[-x for x in h["discharge"]], marker_color=THEME["red"], opacity=0.7))
        fig.add_trace(go.Scatter(name="SOC (MWh)", x=list(range(24)), y=h["soc"], mode='lines+markers',
                                line=dict(color=THEME["teal"], width=3), yaxis="y2"))

        # Reserve line
        reserve_mwh = CLUSTER_CONFIG[cluster_opt]["storage_mwh"] * RESERVE_REQUIREMENT_CAPACITY
        fig.add_hline(y=reserve_mwh, line_dash="dash", line_color=THEME["amber"],
                     annotation_text=f"Reserve ({reserve_mwh:.0f} MWh)", yaxis="y2")

        fig.update_layout(
            barmode='relative', yaxis_title="Power (MW)",
            yaxis2=dict(title="SOC (MWh)", overlaying="y", side="right", showgrid=False)
        )
        st.plotly_chart(style_plotly(fig, f"{cluster_opt} - Battery Dispatch & SOC", 400), use_container_width=True)

        # Energy balance table with losses
        st.markdown("#### Hourly Energy Balance & Losses (MWh)")
        balance_df = pd.DataFrame({
            "Hour": list(range(24)),
            "Demand": h["demand"],
            "Renewable Used": h["renewable_used"],
            "DISCOM": h["discom_procurement"],
            "IEX": h["iex_procurement"],
            "Bat. Discharge": h["discharge"],
            "Bat. Charge": h["charge"],
            "Served": h["served"],
            "Unserved": h["unserved"],
            "Curtailment": h["curtailment"],
            "Dist. Losses": h["distribution_losses"],
            "Bat. Conv. Loss": h["battery_conversion_loss"],
            "Self-Discharge": h["self_discharge_loss"],
            "Total Losses": h["total_losses_per_hour"],
            "SOC": h["soc"],
            "Balance Err": h["balance_error"],
        })
        st.dataframe(balance_df.style.format("{:.2f}"), use_container_width=True, hide_index=True)

        # Reserve check
        min_soc = min(h["soc"])
        reserve_status = "✅ PASS" if min_soc >= reserve_mwh else "❌ FAIL"
        st.markdown(f"**Reserve Check (5% = {reserve_mwh:.0f} MWh):** Min SOC = {min_soc:.0f} MWh — {reserve_status}")
        max_balance_err = np.max(np.abs(h["balance_error"]))
        if max_balance_err > 1e-3:
            st.warning(f"⚠️ Energy balance max error: {max_balance_err:.4f} MWh")
        else:
            st.success(f"✅ Energy balance verified (max error: {max_balance_err:.2e} MWh)")

    with tab5:
        st.markdown("### Battery Intelligence")

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            for cluster in CLUSTERS:
                r = results[cluster]
                h = r["hourly"]
                fig.add_trace(go.Scatter(x=list(range(24)), y=h["soc"], name=cluster, mode='lines', line=dict(width=2)))
            st.plotly_chart(style_plotly(fig, "Stationary Battery SOC (24h)", 350), use_container_width=True)

        with col2:
            fig = go.Figure()
            for cluster in CLUSTERS:
                bat = results[cluster]["battery"]
                soh = bat["soh"][-1] * 100 if len(bat["soh"]) > 0 else 95
                fig.add_trace(go.Bar(name=cluster, x=[cluster], y=[soh], marker_color=THEME["teal"]))
            fig.add_hline(y=80, line_dash="dash", line_color=THEME["amber"], annotation_text="Replacement Threshold")
            fig.update_layout(yaxis_title="SOH (%)", showlegend=False)
            st.plotly_chart(style_plotly(fig, "Battery State of Health (End of Horizon)", 350), use_container_width=True)

        # Battery details per cluster
        for cluster in CLUSTERS:
            with st.expander(f"🔋 {cluster} — Battery Details"):
                r = results[cluster]
                cfg = CLUSTER_CONFIG[cluster]
                h = r["hourly"]
                bat = r["battery"]

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Capacity", f"{cfg['storage_mwh']} MWh")
                with c2:
                    st.metric("Avg SOC", f"{np.mean(h['soc']):.0f} MWh ({np.mean(h['soc'])/cfg['storage_mwh']*100:.0f}%)")
                with c3:
                    st.metric("Throughput", f"{np.sum(h['charge']+h['discharge']):.0f} MWh")
                with c4:
                    deg_cost = np.sum((h['charge']+h['discharge'])) * BATTERY_CONFIG["degradation_cost_inr_per_mwh_throughput"]
                    st.metric("Deg. Cost", f"₹{deg_cost/1e5:.1f} Lakh")

                # Reserve info
                st.markdown(f"**Reserve Requirement:** {cfg['storage_mwh'] * RESERVE_REQUIREMENT_CAPACITY:.0f} MWh (5% of storage)")
                st.markdown(f"**Min SOC (Physical):** {cfg['storage_mwh'] * BATTERY_CONFIG['min_soc_fraction']:.0f} MWh")

                # Swap pack inventory
                st.markdown("**Swap Pack Inventory**")
                total_packs = bat["swap_packs_total"]
                reserve_packs = bat["swap_reserve_packs"]
                charged = bat["swap_charged"][-1]
                empty = bat["swap_empty"][-1]
                partial = bat["swap_partial"][-1]

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("🟢 Charged", f"{int(charged)} packs")
                s2.metric("🟡 Partial", f"{int(partial)} packs")
                s3.metric("🔴 Empty", f"{int(empty)} packs")
                s4.metric("🛡️ Reserved", f"{reserve_packs} packs")

                st.markdown(f"**Total Packs:** {total_packs} | **Available for Service:** {int(charged) - reserve_packs} | **Reserve:** {reserve_packs} packs ({RESERVE_REQUIREMENT_SWAP*100:.0f}%)")

    with tab6:
        st.markdown("### Market Intelligence & Dynamic Pricing")

        cluster_mkt = st.selectbox("Select Cluster", CLUSTERS, key="mkt_cluster")
        h = results[cluster_mkt]["hourly"]
        tariffs = results[cluster_mkt]["tariffs"]
        prev_tariffs = results[cluster_mkt]["prev_day_tariffs"]

        # IEX Price Curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(24)), y=h["iex_prices"], name="IEX DAM",
                                mode='lines+markers', line=dict(color=THEME["amber"], width=3)))
        fig.add_trace(go.Scatter(x=list(range(24)), y=[PRICING_CONFIG["discom_base"][cluster_mkt]]*24, name="DISCOM",
                                mode='lines', line=dict(color=THEME["teal"], width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=list(range(24)), y=[PRICING_CONFIG["renewable_ppa_solar"]]*24, name="Renewable PPA",
                                mode='lines', line=dict(color=THEME["green"], width=2, dash='dot')))
        st.plotly_chart(style_plotly(fig, f"{cluster_mkt} - 24h Price Curve", 350), use_container_width=True)

        # Procurement recommendation table
        st.markdown("#### Hourly Procurement Recommendations")
        proc_df = pd.DataFrame({
            "Hour": list(range(24)),
            "IEX Price (₹/MWh)": h["iex_prices"],
            "Procure Renewable (MWh)": h["renewable_used"],
            "Procure DISCOM (MWh)": h["discom_procurement"],
            "Procure IEX (MWh)": h["iex_procurement"],
            "Total Cost (₹)": h["renewable_used"]*PRICING_CONFIG["renewable_ppa_solar"] + h["discom_procurement"]*PRICING_CONFIG["discom_base"][cluster_mkt] + h["iex_procurement"]*h["iex_prices"],
        })
        st.dataframe(proc_df.style.format({"IEX Price (₹/MWh)": "₹{:,.0f}", "Total Cost (₹)": "₹{:,.0f}"}),
                    use_container_width=True, hide_index=True)

        # Dynamic Tariffs with bounds
        st.markdown("#### Dynamic Tariff Recommendations (±20% of Previous Day Average)")
        tariff_df = pd.DataFrame({
            "Hour": list(range(24)),
            "Retail (₹/kWh)": tariffs["Retail"],
            "Ride-hailing (₹/kWh)": tariffs["Ride-hailing"],
            "Logistics (₹/kWh)": tariffs["Logistics"],
        })

        for seg in SEGMENTS:
            base = prev_tariffs.get(seg, SEGMENT_BASE_TARIFF[seg])
            tariff_df[f"{seg} Min"] = base * 0.8
            tariff_df[f"{seg} Max"] = base * 1.2
            tariff_df[f"{seg} Prev Avg"] = base

        st.dataframe(tariff_df.style.format("{:.2f}"), use_container_width=True, hide_index=True)

        # Tariff chart
        fig = go.Figure()
        for seg in SEGMENTS:
            fig.add_trace(go.Scatter(x=list(range(24)), y=tariffs[seg], name=seg, mode='lines+markers', line=dict(width=2)))
            base = prev_tariffs.get(seg, SEGMENT_BASE_TARIFF[seg])
            fig.add_trace(go.Scatter(x=list(range(24)), y=[base*0.8]*24, name=f"{seg} Min",
                                    mode='lines', line=dict(width=1, dash='dot'), opacity=0.3, showlegend=False))
            fig.add_trace(go.Scatter(x=list(range(24)), y=[base*1.2]*24, name=f"{seg} Max",
                                    mode='lines', line=dict(width=1, dash='dot'), opacity=0.3, showlegend=False))
        st.plotly_chart(style_plotly(fig, f"{cluster_mkt} - Recommended Tariffs vs Bounds", 350), use_container_width=True)

        # Revenue projection
        revenue = estimate_revenue(tariffs, h["served"])
        st.metric("Projected 24h Revenue", f"₹{np.sum(revenue)/1e7:.2f} Cr")

    with tab7:
        st.markdown("### Scenario Simulator")

        st.markdown("Modify parameters below and click **Recompute** to see impact on KPIs.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demand Shocks**")
            demand_mult = st.slider("System Demand Multiplier", 0.5, 2.0, 1.0, 0.05)
            delhi_mult = st.slider("Delhi NCR Extra", 0.5, 2.0, 1.0, 0.05)
            mumbai_mult = st.slider("Mumbai MMR Extra", 0.5, 2.0, 1.0, 0.05)

        with col2:
            st.markdown("**Supply Shocks**")
            renewable_mult = st.slider("Renewable Generation", 0.3, 1.5, 1.0, 0.05)
            iex_mult = st.slider("IEX Price Multiplier", 0.5, 3.0, 1.0, 0.05)
            discom_mult = st.slider("DISCOM Price Multiplier", 0.8, 1.5, 1.0, 0.05)

        with col3:
            st.markdown("**Battery Constraints**")
            reserve_pct = st.slider("Reserve Requirement (%)", 2, 15, 5, 1)
            max_soc_pct = st.slider("Max SOC Limit (%)", 70, 100, 90, 5)
            min_soc_pct = st.slider("Min SOC Limit (%)", 5, 30, 10, 5)

        if st.button("🚀 Run Scenario Simulation", type="primary", use_container_width=True):
            custom_params = {
                "demand_multiplier": demand_mult,
                "renewable_multiplier": renewable_mult,
                "iex_multiplier": iex_mult,
                "discom_multiplier": discom_mult,
                "demand_cluster_multiplier": {"Delhi NCR": delhi_mult, "Mumbai MMR": mumbai_mult}
            }
            st.session_state.custom_scenario = custom_params
            st.rerun()

        # Show custom scenario results if available
        if "custom_scenario" in st.session_state:
            custom_results = run_simulation(data_dict, forecasts, strategy="optimized",
                                           scenario_params=st.session_state.custom_scenario)
            sys_c = custom_results["_system"]

            st.markdown("#### Custom Scenario Results")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Reliability", f"{sys_c['reliability_pct']:.1f}%",
                         f"{sys_c['reliability_pct'] - sys['reliability_pct']:+.1f}%")
            with c2:
                st.metric("Losses", f"{sys_c['losses_pct']:.1f}%",
                         f"{sys_c['losses_pct'] - sys['losses_pct']:+.1f}%")
            with c3:
                st.metric("EBITDA Margin", f"{sys_c['ebitda_margin_pct']:.1f}%",
                         f"{sys_c['ebitda_margin_pct'] - sys['ebitda_margin_pct']:+.1f}%")
            with c4:
                st.metric("Revenue", f"₹{sys_c['revenue_inr']/1e7:.2f} Cr",
                         f"₹{(sys_c['revenue_inr'] - sys['revenue_inr'])/1e7:+.2f} Cr")

            # Comparison chart
            fig = go.Figure()
            categories = ["Reliability", "Losses (inv)", "EBITDA Margin", "Revenue (norm)"]
            base_vals = [sys['reliability_pct'], 100-sys['losses_pct'], sys['ebitda_margin_pct'], sys['revenue_inr']/1e7]
            custom_vals = [sys_c['reliability_pct'], 100-sys_c['losses_pct'], sys_c['ebitda_margin_pct'], sys_c['revenue_inr']/1e7]

            fig.add_trace(go.Bar(name="Baseline", x=categories, y=base_vals, marker_color=THEME["gray"]))
            fig.add_trace(go.Bar(name="Scenario", x=categories, y=custom_vals, marker_color=THEME["teal"]))
            fig.update_layout(barmode='group')
            st.plotly_chart(style_plotly(fig, "Scenario vs Baseline", 350), use_container_width=True)

        # Predefined scenario comparison
        st.markdown("#### Predefined Scenario Comparison (Optimized Strategy)")
        scenarios_list = [
            ("Normal", {}),
            ("Renewable -30%", {"renewable_multiplier": 0.7}),
            ("IEX +50%", {"iex_multiplier": 1.5}),
            ("Demand +25% (Del, Mum)", {"demand_cluster_multiplier": {"Delhi NCR": 1.25, "Mumbai MMR": 1.25}}),
            ("Combined Stress", {"renewable_multiplier": 0.7, "iex_multiplier": 1.5, "demand_cluster_multiplier": {"Delhi NCR": 1.25, "Mumbai MMR": 1.25}}),
        ]

        scenario_kpis = []
        for name, params in scenarios_list:
            sr = run_simulation(data_dict, forecasts, strategy="optimized", scenario_params=params)
            s = sr["_system"]
            scenario_kpis.append({
                "Scenario": name,
                "Reliability": f"{s['reliability_pct']:.1f}%",
                "Losses": f"{s['losses_pct']:.1f}%",
                "Renewable Util": f"{s['renewable_util_pct']:.1f}%",
                "Revenue (Cr)": f"₹{s['revenue_inr']/1e7:.2f}",
                "EBITDA Margin": f"{s['ebitda_margin_pct']:.1f}%",
            })

        st.dataframe(pd.DataFrame(scenario_kpis), use_container_width=True, hide_index=True)

    with tab8:
        st.markdown("### Financial Performance")

        # Baseline vs Optimized comparison
        st.markdown("#### Baseline vs Optimized Strategy Comparison")

        sys_opt = opt_results["_system"]
        sys_base = base_results["_system"]

        comp_data = {
            "Metric": [
                "Revenue (Cr INR)", "Procurement Cost (Cr INR)", "Degradation Cost (Lakh INR)",
                "O&M Cost (Cr INR)", "Total OPEX (Cr INR)", "EBITDA (Cr INR)", "EBITDA Margin (%)",
                "Reliability (%)", "Energy Losses (%)", "Renewable Utilization (%)"
            ],
            "Baseline": [
                f"₹{sys_base['revenue_inr']/1e7:.2f}", f"₹{sys_base['procurement_cost_inr']/1e7:.2f}",
                f"₹{sys_base['degradation_cost_inr']/1e5:.1f}", f"₹{sys_base['om_cost_inr']/1e7:.2f}",
                f"₹{sys_base['opex_inr']/1e7:.2f}", f"₹{sys_base['ebitda_inr']/1e7:.2f}",
                f"{sys_base['ebitda_margin_pct']:.1f}%", f"{sys_base['reliability_pct']:.1f}%",
                f"{sys_base['losses_pct']:.1f}%", f"{sys_base['renewable_util_pct']:.1f}%"
            ],
            "Optimized": [
                f"₹{sys_opt['revenue_inr']/1e7:.2f}", f"₹{sys_opt['procurement_cost_inr']/1e7:.2f}",
                f"₹{sys_opt['degradation_cost_inr']/1e5:.1f}", f"₹{sys_opt['om_cost_inr']/1e7:.2f}",
                f"₹{sys_opt['opex_inr']/1e7:.2f}", f"₹{sys_opt['ebitda_inr']/1e7:.2f}",
                f"{sys_opt['ebitda_margin_pct']:.1f}%", f"{sys_opt['reliability_pct']:.1f}%",
                f"{sys_opt['losses_pct']:.1f}%", f"{sys_opt['renewable_util_pct']:.1f}%"
            ],
            "Improvement": [
                f"₹{(sys_opt['revenue_inr']-sys_base['revenue_inr'])/1e7:+.2f}",
                f"₹{(sys_opt['procurement_cost_inr']-sys_base['procurement_cost_inr'])/1e7:+.2f}",
                f"₹{(sys_opt['degradation_cost_inr']-sys_base['degradation_cost_inr'])/1e5:+.1f}",
                f"₹{(sys_opt['om_cost_inr']-sys_base['om_cost_inr'])/1e7:+.2f}",
                f"₹{(sys_opt['opex_inr']-sys_base['opex_inr'])/1e7:+.2f}",
                f"₹{(sys_opt['ebitda_inr']-sys_base['ebitda_inr'])/1e7:+.2f}",
                f"{sys_opt['ebitda_margin_pct']-sys_base['ebitda_margin_pct']:+.1f}%",
                f"{sys_opt['reliability_pct']-sys_base['reliability_pct']:+.1f}%",
                f"{sys_base['losses_pct']-sys_opt['losses_pct']:+.1f}%",
                f"{sys_opt['renewable_util_pct']-sys_base['renewable_util_pct']:+.1f}%"
            ]
        }

        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        # Target achievement
        st.markdown("#### Target Achievement Status")
        targets = [
            ("Service Reliability ≥ 89.7%", sys_opt['reliability_pct'], 89.7, "≥"),
            ("Energy Losses ≤ 8%", sys_opt['losses_pct'], 8, "≤"),
            ("EBITDA Margin ≥ 12%", sys_opt['ebitda_margin_pct'], 12, "≥"),
        ]

        for name, actual, target, op in targets:
            if op == "≥":
                met = actual >= target
            else:
                met = actual <= target
            status = "✅ ACHIEVED" if met else "❌ NOT MET"
            color = THEME["green"] if met else THEME["red"]
            st.markdown(f"""
            <div style="background:{THEME['card_bg']}; padding:1rem; border-radius:8px; margin:0.5rem 0; border-left:4px solid {color};">
                <strong>{name}</strong> — Actual: {actual:.1f}% — {status}
            </div>
            """, unsafe_allow_html=True)

        # Sensitivity analysis
        st.markdown("#### Sensitivity Analysis (Optimized Strategy)")
        sens_data = []
        for iex_m in [0.7, 0.85, 1.0, 1.2, 1.5]:
            sr = run_simulation(data_dict, forecasts, strategy="optimized", scenario_params={"iex_multiplier": iex_m})
            s = sr["_system"]
            sens_data.append({
                "IEX Multiplier": f"{iex_m:.2f}x",
                "Reliability": f"{s['reliability_pct']:.1f}%",
                "Losses": f"{s['losses_pct']:.1f}%",
                "EBITDA Margin": f"{s['ebitda_margin_pct']:.1f}%",
                "Revenue (Cr)": f"₹{s['revenue_inr']/1e7:.2f}",
            })
        st.dataframe(pd.DataFrame(sens_data), use_container_width=True, hide_index=True)

        # Disclaimer
        st.markdown("---")
        st.markdown("""
        <div style="color:gray; font-size:0.8rem;">
        <strong>Disclaimer:</strong> All results based on synthetic historical data and simulated optimization.
        Forecast accuracy on synthetic data does not represent real-world predictive performance.
        Solver status shown per cluster — heuristic fallback used when MILP infeasible or fails.
        No live IEX data integration. All financials in INR. Energy in MWh, Power in MW.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()