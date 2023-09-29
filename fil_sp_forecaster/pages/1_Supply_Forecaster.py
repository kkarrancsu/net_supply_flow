#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

from datetime import date, timedelta

import time

import numpy as np
import pandas as pd
import jax.numpy as jnp

import streamlit as st
import streamlit.components.v1 as components
import altair as alt

import mechafil_jax.data as data
import mechafil_jax.sim as sim
import mechafil_jax.constants as C
import mechafil_jax.minting as minting
import mechafil_jax.date_utils as du

import scenario_generator.utils as u

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# local_css("debug.css")

@st.cache_data
def get_offline_data(start_date, current_date, end_date):
    PUBLIC_AUTH_TOKEN='Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ'
    offline_data = data.get_simulation_data(PUBLIC_AUTH_TOKEN, start_date, current_date, end_date)

    _, hist_rbp = u.get_historical_daily_onboarded_power(current_date-timedelta(days=180), current_date)
    _, hist_rr = u.get_historical_renewal_rate(current_date-timedelta(days=180), current_date)
    _, hist_fpr = u.get_historical_filplus_rate(current_date-timedelta(days=180), current_date)

    smoothed_last_historical_rbp = float(np.median(hist_rbp[-30:]))
    smoothed_last_historical_rr = float(np.median(hist_rr[-30:]))
    smoothed_last_historical_fpr = float(np.median(hist_fpr[-30:]))

    return offline_data, smoothed_last_historical_rbp, smoothed_last_historical_rr, smoothed_last_historical_fpr

def xr(locking_pct_change, xr_locking_sensitivity):
    return 1 + locking_pct_change * xr_locking_sensitivity / 100
    
def pledge(pledge0, locking_pct_change):
    return pledge0 * (1 + locking_pct_change / 100)
    
def locking_pct_change(TL):
    return 100 * (TL / 30 - 1)
    
def ROI(reward, pledge, cost, xr, pct_fiat_cost):
    return 100 * (xr * reward - 0.01 * cost * reward * (pct_fiat_cost / 100 + (1 - pct_fiat_cost / 100) * xr)) / (xr * pledge)
    
TL_values = np.linspace(30, 90, 100)  # Define the range of TL values for plotting


def plot_panel(scenario_results, baseline, start_date, current_date, end_date, simulation_start_idx):
    # convert results dictionary into a dataframe so that we can use altair to make nice plots
    status_quo_results = scenario_results['status-quo']
    
    power_dff = pd.DataFrame()
    power_dff['RBP'] = status_quo_results['rb_total_power_eib']
    power_dff['QAP'] = status_quo_results['qa_total_power_eib']
    power_dff['Baseline'] = baseline
    power_dff['date'] = pd.to_datetime(du.get_t(start_date, end_date=end_date))

    minting_dff = pd.DataFrame()
    minting_dff['StatusQuo'] = status_quo_results['block_reward']
    minting_dff['date'] = pd.to_datetime(du.get_t(start_date, end_date=end_date))

    pledge_dff = pd.DataFrame()
    pledge_dff['StatusQuo'] = status_quo_results['day_pledge_per_QAP']
    pledge_dff['date'] = pd.to_datetime(du.get_t(start_date, forecast_length=pledge_dff.shape[0]))

    supplyflow_dff = pd.DataFrame()
    supplyflow_dff['StatusQuo'] = status_quo_results['circ_supply']
    supplyflow_dff['StatusQuo'] = supplyflow_dff['StatusQuo'].diff().rolling(28).median().dropna() / 1_000_000
    supplyflow_dff['date'] = pd.to_datetime(du.get_t(start_date, forecast_length=supplyflow_dff.shape[0]))
    
    CostPCTofRewards = st.session_state['cost_pct_rewards']
    XRLockSensitivity = st.session_state['xr_locking_sensitivity']
    PCTCostInFiat = st.session_state['pct_fiat_cost']

    rps_at_sim_start = float(status_quo_results['1y_return_per_sector'][simulation_start_idx])
    pledge_at_sim_start = float(status_quo_results['day_pledge_per_QAP'][simulation_start_idx])

    ROI_values_2 = [ROI(rps_at_sim_start, pledge(pledge_at_sim_start, locking_pct_change(TL)), CostPCTofRewards, xr(locking_pct_change(TL), XRLockSensitivity), PCTCostInFiat)
                    - ROI(rps_at_sim_start, pledge(pledge_at_sim_start, locking_pct_change(30)), CostPCTofRewards, xr(locking_pct_change(30), XRLockSensitivity), PCTCostInFiat)
                    for TL in TL_values]
    
    plot_df = pd.DataFrame()
    plot_df['TL'] = TL_values
    plot_df['ROI (cfg)'] = ROI_values_2
    
    supplyflow_df = pd.melt(supplyflow_dff, id_vars=["date"],
                                    value_vars=["StatusQuo"],
                                    var_name='Scenario', value_name='M-FIL')
    plot_df = plot_df.melt('TL', var_name='ROI', value_name='Value')

    col1, col2 = st.columns(2)
    with col1:
        supplyflow = (
            alt.Chart(supplyflow_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(labelAngle=-45)), 
                    y=alt.Y("M-FIL"), color=alt.Color('Scenario', legend=None))
            .properties(title="Net Supply Flow")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(supplyflow.interactive(), use_container_width=True)

        power_df = pd.melt(power_dff, id_vars=["date"], 
                           value_vars=[
                               "Baseline", 
                               "RBP", "QAP",],
                           var_name='Power', 
                           value_name='EIB')
        power_df['EIB'] = power_df['EIB']
        power = (
            alt.Chart(power_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(labelAngle=-45)), 
                    y=alt.Y("EIB"), 
                    color=alt.Color('Power', legend=alt.Legend(orient="top", title=None)))
            # .encode(x=alt.X("date", title="", axis=alt.Axis(labelAngle=-45)), 
            #         y=alt.Y("EIB"))
            .properties(title="Network Power")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(power.interactive(), use_container_width=True) 

    with col2:
        chart = (
            alt.Chart(plot_df)
            .mark_line()
            .encode(
                x=alt.X('TL', title='Locked Supply (%)'),
                y=alt.Y('Value', title='%'))
            .properties(title="Fiat ROI boost")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(chart.interactive(), use_container_width=True)


def run_sim(rbp, rr, fpr, lock_target, start_date, current_date, forecast_length_days, sector_duration_days, burn_boost, offline_data):
    # apply burn boost
    offline_data['daily_burnt_fil'] = offline_data['daily_burnt_fil'] * burn_boost
    simulation_results = sim.run_sim(
        rbp,
        rr,
        fpr,
        lock_target,

        start_date,
        current_date,
        forecast_length_days,
        sector_duration_days,
        offline_data
    )
    simulation_results['block_reward'] = simulation_results['day_network_reward'] / float(5*2880)
    
    return simulation_results

def forecast_economy(start_date=None, current_date=None, end_date=None, forecast_length_days=365*6, simulation_start_idx=0):
    t1 = time.time()
    
    rb_onboard_power_pib_day =  st.session_state['rbp_slider']
    renewal_rate_pct = st.session_state['rr_slider']
    fil_plus_rate_pct = st.session_state['fpr_slider']

    lock_target = st.session_state['lock_target_slider']
    sector_duration_days = st.session_state['av_dur_slider']

    burn_boost = st.session_state['burn_factor_slider']

    # get offline data
    t2 = time.time()
    offline_data, _, _, _ = get_offline_data(start_date, current_date, end_date)
    t3 = time.time()

    # run simulation for the configured scenario, and for a pessimsitc and optimistic version of it
    scenario_scalers = [1.0]
    scenario_strings = ['status-quo']
    scenario_results = {}
    for ii, scenario_scaler in enumerate(scenario_scalers):
        rbp_val = rb_onboard_power_pib_day * scenario_scaler
        rr_val = max(0.0, min(1.0, renewal_rate_pct / 100. * scenario_scaler))
        fpr_val = max(0.0, min(1.0, fil_plus_rate_pct / 100. * scenario_scaler))

        rbp = jnp.ones(forecast_length_days) * rbp_val
        rr = jnp.ones(forecast_length_days) * rr_val
        fpr = jnp.ones(forecast_length_days) * fpr_val
        
        simulation_results = run_sim(rbp, rr, fpr, lock_target, start_date, current_date, forecast_length_days, sector_duration_days,burn_boost, offline_data) 
        scenario_results[scenario_strings[ii]] = simulation_results

    baseline = minting.compute_baseline_power_array(
        np.datetime64(start_date), np.datetime64(end_date), offline_data['init_baseline_eib'],
    )

    # plot
    plot_panel(scenario_results, baseline, start_date, current_date, end_date, simulation_start_idx)
    t4 = time.time()

def forecast_len():
    forecast_length_days=st.session_state['forecast_length_slider']
    return forecast_length_days

def main():
    st.set_page_config(
        page_title="Filecoin Net Supply Explorer",
        layout="wide",
    )
    current_date = date.today() - timedelta(days=3)
    mo_start = max(current_date.month - 1 % 12, 1)
    start_date = date(current_date.year, mo_start, 1)

    forecast_length_days=(365*1)
    end_date = current_date + timedelta(days=forecast_length_days)
    simulation_start_idx = (current_date - start_date).days
    
    forecast_kwargs = {
        'start_date': start_date,
        'current_date': current_date,
        'end_date': end_date,
        'forecast_length_days': forecast_length_days,
        'simulation_start_idx': simulation_start_idx,
    }

    _, smoothed_last_historical_rbp, smoothed_last_historical_rr, smoothed_last_historical_fpr = get_offline_data(start_date, current_date, end_date)
    smoothed_last_historical_renewal_pct = int(smoothed_last_historical_rr * 100)
    smoothed_last_historical_fil_plus_pct = int(smoothed_last_historical_fpr * 100)

    with st.sidebar:
        with st.expander('Macro Configuration'):
            st.slider("Raw Byte Onboarding (PiB/day)", min_value=3., max_value=50., value=smoothed_last_historical_rbp, step=.1, format='%0.02f', key="rbp_slider",
                    on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
            st.slider("Renewal Rate (Percentage)", min_value=10, max_value=99, value=95, step=1, format='%d', key="rr_slider",
                    on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
            st.slider("FIL+ Rate (Percentage)", min_value=10, max_value=99, value=smoothed_last_historical_fil_plus_pct, step=1, format='%d', key="fpr_slider",
                    on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
            st.slider("Lock Target (Percentage)", min_value=0.1, max_value=0.9, value=0.3, step=0.01, format='%.2f', key="lock_target_slider",
                    on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
            st.slider("Average Sector Duration (Days)", min_value=180, max_value=540, value=360, step=10, format='%d', key="av_dur_slider",
                    on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
            st.slider("Burn Factor (Multiplicative)", min_value=0.1, max_value=10., value=1., step=0.1, format='%.2f', key="burn_factor_slider",
                    on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")

        with st.expander('Optimal Locking'):
            st.slider(
                "SP costs as a \% of rewards", min_value=50, max_value=95, value=90, step=1, key="cost_pct_rewards",
                on_change=forecast_economy, kwargs=forecast_kwargs
            )
            st.slider(
                "\% of costs in paid in fiat", min_value=50, max_value=95, value=80, step=1, key="pct_fiat_cost",
                on_change=forecast_economy, kwargs=forecast_kwargs
            )
            st.slider(
                "XR locking sensitivity", min_value=1, max_value=25, value=10, step=1, key="xr_locking_sensitivity",
                on_change=forecast_economy, kwargs=forecast_kwargs
            )
        
        st.button("Forecast", on_click=forecast_economy, kwargs=forecast_kwargs, key="forecast_button")

if __name__ == '__main__':
    main()
