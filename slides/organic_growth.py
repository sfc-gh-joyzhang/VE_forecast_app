import plotly.graph_objects as go
import numpy as np
import pandas as pd
import io
import streamlit as st
from datetime import datetime, timedelta

def get_session_defaults():
    """Return default sessionz state values for organic growth"""
    return {
        'og_start_month': '2023-05-01',  # Default to 24 months ago
        'og_end_month': '2025-05-01',    # Default to latest month
        'og_compute_cmgr': 0.0,
        'og_storage_cmgr': 0.0,
        'og_other_cmgr': 0.0,
        'og_data_transfer_cmgr': 0.0,
        'og_priority_support_cmgr': 0.0,
        'og_total_cmgr': 0.0
    }

def render(connection, selected_customer_schema, validated_info, connection_type='snowpark'):
    """
    Render the Organic Growth analysis slide
    
    Args:
        connection: Snowflake connection (session for Snowpark, cursor for browser)
        selected_customer_schema: Customer schema (e.g., 'FINOPS_OUTPUTS.SIEMENS_AG')
        validated_info: Dictionary with Salesforce account validation information
        connection_type: Type of connection ('snowpark' or 'cursor')
    """
    
    st.markdown("*Select the Beginning & End Month that most represents Steady State for the account (minimal use cases added)*")
    
    # Build SQL query using Salesforce Account ID from validation
    salesforce_id = validated_info['salesforce_id']
    salesforce_name = validated_info['salesforce_name']
    
    # Use schema from validation (already discovered in main app)
    if 'potential_schema' in validated_info:
        uem_schema = validated_info['potential_schema']
        st.info(f"Using schema: {uem_schema}")
    else:
        # Fallback: derive from customer name
        customer_name_clean = customer_name.replace(' ', '_').upper()
        uem_schema = f"FINOPS_OUTPUTS.{customer_name_clean}_UE_20250606"  # Fallback pattern
        st.warning(f"Using fallback schema: {uem_schema}")
        
    # SQL query replicating Excel SUMIFS logic
    
    billing_sql = f"""
    WITH billing_data AS (
        SELECT 
            DATE_TRUNC('MONTH', MONTH::DATE) AS MONTH,
            REVENUE_CATEGORY,
            REVENUE_GROUP,
            SUM(REVENUE) AS REVENUE,
            SUM(CREDITS) AS CREDITS,
            SUM(STORAGE_TB) AS STORAGE_TB,
            SUM(TRANSFER_TB) AS TRANSFER_TB
        FROM {uem_schema}.BILLING
        WHERE MONTH IS NOT NULL
            AND MONTH >= DATEADD(YEAR, -2, CURRENT_DATE())
            AND MONTH < DATE_TRUNC('MONTH', CURRENT_DATE())
        GROUP BY MONTH, REVENUE_CATEGORY, REVENUE_GROUP
    ),
    monthly_summary AS (
        SELECT 
            MONTH,
            SUM(CASE WHEN REVENUE_CATEGORY = 'Compute' THEN REVENUE ELSE 0 END) AS COMPUTE,
            SUM(CASE WHEN REVENUE_CATEGORY = 'Storage' THEN REVENUE ELSE 0 END) AS STORAGE,
            SUM(CASE WHEN REVENUE_CATEGORY = 'Data Transfer' THEN REVENUE ELSE 0 END) AS DATA_TRANSFER,
            SUM(CASE WHEN REVENUE_CATEGORY = 'Priority Support' THEN REVENUE ELSE 0 END) AS PRIORITY_SUPPORT,
            SUM(CASE WHEN REVENUE_CATEGORY NOT IN ('Compute', 'Storage', 'Data Transfer', 'Priority Support') THEN REVENUE ELSE 0 END) AS OTHER,
            SUM(REVENUE) AS TOTAL
            
        FROM billing_data
        GROUP BY MONTH
        ORDER BY MONTH
    )
    SELECT * FROM monthly_summary
    WHERE TOTAL > 0
    """
    
    # Execute billing query
    try:
        if connection_type == 'snowpark':
            df = connection.sql(billing_sql).to_pandas()
        else:
            from browser_connection import execute_sql
            df = execute_sql(connection, billing_sql)
            
        if len(df) == 0:
            st.error("No billing data found for this customer schema")
            # Debug: Check what's actually in the raw BILLING table
            debug_sql = f"SELECT TOP 5 * FROM {uem_schema}.BILLING ORDER BY MONTH DESC"
            try:
                if connection_type == 'snowpark':
                    debug_df = connection.sql(debug_sql).to_pandas()
                else:
                    debug_df = execute_sql(connection, debug_sql)
                st.write("Sample raw data from BILLING table:")
                st.dataframe(debug_df)
            except Exception as debug_error:
                st.error(f"Cannot access BILLING table: {debug_error}")
            return
        # Query successful - proceed with analysis
        pass
            
    except Exception as billing_error:
        st.error(f"Billing query failed: {str(billing_error)}")
        return
    
    st.markdown(f"**Analyzing consumption data for:** {salesforce_name} ({salesforce_id})")
    
    # Continue with data processing
    if df.empty:
        st.warning(f"No consumption data found for {salesforce_name} ({salesforce_id}) in the last 2 years")
        st.markdown("**Possible reasons:**")
        st.markdown("- No Snowflake consumption activity during this period")
        st.markdown("- Data not yet processed in billing table") 
        st.markdown("- Customer schema mapping issue")
        return
    
    # Convert MONTH to datetime for proper sorting
    df['MONTH'] = pd.to_datetime(df['MONTH'])
    df = df.sort_values('MONTH')
    # Expose monthly totals for downstream Forecast Trend chart
    try:
        st.session_state['og_monthly_df'] = df[['MONTH', 'TOTAL']].copy()
    except Exception:
        pass
    
    # Expose latest complete-month type split percentages for defaults in Use Case Forecaster
    try:
        latest_row = df.iloc[-1]
        latest_total = float(latest_row['TOTAL']) if float(latest_row['TOTAL']) != 0 else 0.0
        if latest_total > 0:
            # Expose latest month total for 5-year baseline in Use Case Forecaster
            st.session_state['og_latest_month_total'] = latest_total
            latest_compute_pct = float(latest_row['COMPUTE']) / latest_total * 100.0
            latest_storage_pct = float(latest_row['STORAGE']) / latest_total * 100.0
            latest_other_pct = float(latest_row['OTHER']) / latest_total * 100.0
            latest_dt_pct = float(latest_row['DATA_TRANSFER']) / latest_total * 100.0
            # Store raw splits and combined Compute+Other per your Excel mapping (I29 + K29)
            st.session_state['og_latest_split_compute_pct'] = latest_compute_pct
            st.session_state['og_latest_split_storage_pct'] = latest_storage_pct
            st.session_state['og_latest_split_other_pct'] = latest_other_pct
            st.session_state['og_latest_split_data_transfer_pct'] = latest_dt_pct
            st.session_state['og_default_split_compute_pct'] = latest_compute_pct + latest_other_pct
            st.session_state['og_default_split_storage_pct'] = latest_storage_pct
            st.session_state['og_default_split_data_transfer_pct'] = latest_dt_pct
    except Exception:
        pass

    # Remove global START/END selectors; use full available range
    filtered_df = df.copy()
    start_month = filtered_df['MONTH'].min().strftime('%Y-%m-%d')
    end_month = filtered_df['MONTH'].max().strftime('%Y-%m-%d')
        
    if len(filtered_df) < 2:
        st.error(f"Insufficient data for growth calculation. Available dates: {df['MONTH'].min()} to {df['MONTH'].max()}")
        return
    
    # --- Pre-compute Growth Rate Analysis (used in expander below) ---
    def calculate_cmgr(start_value, end_value, months):
        if start_value <= 0 or end_value <= 0 or months <= 0:
            return 0.0
        return (end_value / start_value) ** (1 / months) - 1
    def calculate_cagr(start_value, end_value, years):
        if start_value <= 0 or end_value <= 0 or years <= 0:
            return 0.0
        return (end_value / start_value) ** (1 / years) - 1
    display_categories = ['COMPUTE', 'STORAGE', 'OTHER', 'DATA_TRANSFER', 'COMPUTE (TOTAL)']
    total_months = len(filtered_df) - 1
    years = total_months / 12
    periods = [
        ('CAGR', years, 'years'),
        ('12 months', 12, 'months'),
        ('6 months', 6, 'months'),
        ('3 months', 3, 'months'),
        ('1 month', 1, 'months')
    ]
    growth_data = {}
    for category in ['COMPUTE', 'STORAGE', 'OTHER', 'DATA_TRANSFER', 'PRIORITY_SUPPORT', 'TOTAL']:
        growth_data[category] = {}
        for period_name, period_length, period_type in periods:
            if period_type == 'years' and total_months >= 12:
                latest_date = filtered_df['MONTH'].iloc[-1]
                year_ago_date = latest_date - pd.DateOffset(years=1)
                year_ago_row = filtered_df[filtered_df['MONTH'] <= year_ago_date].iloc[-1] if len(filtered_df[filtered_df['MONTH'] <= year_ago_date]) > 0 else filtered_df.iloc[0]
                latest_row = filtered_df.iloc[-1]
                start_val = year_ago_row[category]
                end_val = latest_row[category]
                growth_rate = (end_val / start_val - 1) * 100 if start_val > 0 else 0.0
            elif period_type == 'months' and total_months >= period_length:
                if period_length <= total_months:
                    start_idx = max(0, len(filtered_df) - period_length - 1)
                    first_val = filtered_df.iloc[start_idx][category]
                    last_val = filtered_df.iloc[-1][category]
                    growth_rate = calculate_cmgr(first_val, last_val, period_length) * 100
                else:
                    growth_rate = 0.0
            else:
                growth_rate = 0.0
            growth_data[category][period_name] = growth_rate
    # Store 12-months growth in session for other slides and fallbacks
    st.session_state['og_compute_cmgr'] = growth_data['COMPUTE'].get('12 months', 0.0)
    st.session_state['og_storage_cmgr'] = growth_data['STORAGE'].get('12 months', 0.0)
    st.session_state['og_other_cmgr'] = growth_data['OTHER'].get('12 months', 0.0)
    st.session_state['og_data_transfer_cmgr'] = growth_data['DATA_TRANSFER'].get('12 months', 0.0)
    st.session_state['og_priority_support_cmgr'] = growth_data['PRIORITY_SUPPORT'].get('12 months', 0.0)
    st.session_state['og_total_cmgr'] = growth_data['TOTAL'].get('12 months', 0.0)
    try:
        if len(filtered_df) >= 13:
            start_idx = len(filtered_df) - 12 - 1
            compute_total_start = float(filtered_df.iloc[start_idx]['COMPUTE'] + filtered_df.iloc[start_idx]['OTHER'])
            compute_total_end = float(filtered_df.iloc[-1]['COMPUTE'] + filtered_df.iloc[-1]['OTHER'])
            if compute_total_start > 0:
                compute_total_cmgr_12m = (compute_total_end / compute_total_start) ** (1 / 12) - 1
                st.session_state['og_y1_compute_total_cmgr_fallback'] = compute_total_cmgr_12m * 100
        st.session_state['og_y1_storage_cmgr_fallback'] = growth_data['STORAGE'].get('12 months', 0.0)
        st.session_state['og_y1_data_transfer_cmgr_fallback'] = growth_data['DATA_TRANSFER'].get('12 months', 0.0)
    except Exception:
        pass

    # ---- Custom Period CMGR Calculator (top) ----
    st.markdown("---")
    st.markdown("### Custom Period CMGR Calculator")

    col1, col2 = st.columns(2)
    with col1:
        start_month = st.selectbox(
            "Start Month:",
            options=filtered_df['MONTH'].dt.strftime('%Y-%m-%d').tolist(),
            index=0,
            key='custom_start_month_top'
        )
    with col2:
        end_month = st.selectbox(
            "End Month:",
            options=filtered_df['MONTH'].dt.strftime('%Y-%m-%d').tolist(),
            index=len(filtered_df)-1,
            key='custom_end_month_top'
        )

    if start_month and end_month:
        start_date = pd.to_datetime(start_month)
        end_date = pd.to_datetime(end_month)
        
        if start_date < end_date:
            start_row = filtered_df[filtered_df['MONTH'] == start_date]
            end_row = filtered_df[filtered_df['MONTH'] == end_date]

            if len(start_row) > 0 and len(end_row) > 0:
                months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

                st.markdown(f"**CMGR(%) ({months_diff} month(s))**")

                # Calculate and store custom CMGR values
                custom_cmgr_values = {}
                custom_cols = st.columns(5)

                for i, category in enumerate(['COMPUTE', 'STORAGE', 'OTHER', 'DATA_TRANSFER', 'TOTAL']):
                    with custom_cols[i]:
                        # For "COMPUTE (TOTAL)", Excel folds OTHER into COMPUTE
                        if category == 'TOTAL':
                            start_val = float(start_row['COMPUTE'].iloc[0] + start_row['OTHER'].iloc[0])
                            end_val = float(end_row['COMPUTE'].iloc[0] + end_row['OTHER'].iloc[0])
                        else:
                            start_val = start_row[category].iloc[0]
                            end_val = end_row[category].iloc[0]

                        if start_val > 0 and months_diff > 0:
                            custom_cmgr = ((end_val / start_val) ** (1 / months_diff) - 1) * 100
                        else:
                            custom_cmgr = 0.0

                        custom_cmgr_values[category] = custom_cmgr

                        color = "#28a745" if custom_cmgr > 0 else "#dc3545" if custom_cmgr < 0 else "#6c757d"
                        display_name = "COMPUTE (TOTAL)" if category == 'TOTAL' else category

                        st.markdown(f"""
                        <div style="text-align: center; padding: 0.5rem; background-color: #f8f9fa; border-radius: 4px; border-left: 3px solid {color};">
                            <div style="font-weight: bold; color: {color}; font-size: 1.1rem;">{custom_cmgr:.2f}%</div>
                            <div style="font-size: 0.8rem; color: #6c757d;">{display_name}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Store in session state for use case defaults
                st.session_state['og_y1_storage_cmgr'] = custom_cmgr_values.get('STORAGE', 0.0)
                st.session_state['og_y1_data_transfer_cmgr'] = custom_cmgr_values.get('DATA_TRANSFER', 0.0)
                # Initialize Y2–Y5 sensible defaults once so the rest of the app has non-zero rates by default
                if 'og_rates_initialized' not in st.session_state:
                    # Compute TOTAL uses compute+other for Y1; for Y2–Y5 use typical presets
                    st.session_state['og_y2_compute_cmgr'] = 0.8
                    st.session_state['og_y3_compute_cmgr'] = 1.0
                    st.session_state['og_y4_compute_cmgr'] = 1.0
                    st.session_state['og_y5_compute_cmgr'] = 1.0
                    st.session_state['og_y2_storage_cmgr'] = 0.8
                    st.session_state['og_y3_storage_cmgr'] = 1.2
                    st.session_state['og_y4_storage_cmgr'] = 1.2
                    st.session_state['og_y5_storage_cmgr'] = 1.2
                    st.session_state['og_y2_data_transfer_cmgr'] = 0.0
                    st.session_state['og_y3_data_transfer_cmgr'] = 0.0
                    st.session_state['og_y4_data_transfer_cmgr'] = 0.0
                    st.session_state['og_y5_data_transfer_cmgr'] = 0.0
                    st.session_state['og_rates_initialized'] = True
                try:
                    compute_total_start = float(start_row['COMPUTE'].iloc[0] + start_row['OTHER'].iloc[0])
                    compute_total_end = float(end_row['COMPUTE'].iloc[0] + end_row['OTHER'].iloc[0])
                    compute_total_cmgr = ((compute_total_end / compute_total_start) ** (1 / months_diff) - 1) * 100 if compute_total_start > 0 else 0.0
                except Exception:
                    compute_total_cmgr = 0.0
                st.session_state['og_y1_compute_total_cmgr'] = compute_total_cmgr

                # Apply button (Y1 only)
                if st.button(
                    f"Apply Calculated CMGR (Y1)",
                    help="Set Y1 monthly CMGR defaults for Use Case Forecaster",
                    type="primary",
                    key="apply_custom_cmgr_y1"
                ):
                    st.success("Applied to Y1 defaults for Use Case Forecaster.")
                    st.rerun()

                # Analysis dropdowns placed right after calculator
                with st.expander("Growth Rate Analysis", expanded=False):
                    col_headers = ['Metric'] + display_categories
                    cols = st.columns(len(col_headers))
                    for i, header in enumerate(col_headers):
                        with cols[i]:
                            st.markdown(f"**{header}**")
                    for period_name, _, _ in periods:
                        cols = st.columns(len(col_headers))
                        with cols[0]:
                            metric_label = f"{period_name} (%)" if period_name != 'CAGR' else "CAGR (%)"
                            st.markdown(f"**{metric_label}**")
                        for i, category in enumerate(['COMPUTE', 'STORAGE', 'OTHER', 'DATA_TRANSFER', 'TOTAL']):
                            with cols[i + 1]:
                                display_cat = category
                                growth_rate = growth_data[display_cat].get(period_name, 0.0)
                                color = "#28a745" if growth_rate > 0 else "#dc3545" if growth_rate < 0 else "#6c757d"
                                st.markdown(f'<span style="color: {color}; font-weight: bold;">{growth_rate:.1f}%</span>', unsafe_allow_html=True)

                with st.expander("View Monthly Data", expanded=False):
                    display_df = filtered_df.copy()
                    display_df['MONTH'] = display_df['MONTH'].dt.strftime('%b %Y')
                    currency_cols = ['COMPUTE', 'STORAGE', 'OTHER', 'DATA_TRANSFER', 'PRIORITY_SUPPORT', 'TOTAL']
                    for col in currency_cols:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
                    st.dataframe(display_df, use_container_width=True)

    # Manual Y1–Y5 CMGR inputs for Compute, Storage, Data Transfer
    st.markdown("#### Organic Growth CMGR Presets and Y1–Y5 Overrides")
    # Build recommended defaults
    rec_compute = [
        float(st.session_state.get('og_y1_compute_total_cmgr', st.session_state.get('og_y1_compute_total_cmgr_fallback', 0.0))),
        0.8, 1.0, 1.0, 1.0,
    ]
    rec_storage = [
        float(st.session_state.get('og_y1_storage_cmgr', st.session_state.get('og_y1_storage_cmgr_fallback', 0.0))),
        0.8, 1.2, 1.2, 1.2,
    ]
    rec_dt = [
        float(st.session_state.get('og_y1_data_transfer_cmgr', st.session_state.get('og_y1_data_transfer_cmgr_fallback', 0.0))),
        0.0, 0.0, 0.0, 0.0,
    ]
    preset = st.selectbox(
        "Preset",
        ["Recommended", "Flat 0%", "Flat at Y1"],
        index=0,
        help="Recommended = calculated Y1 then typical Y2–Y5. Flat at Y1 = copy Y1 to all years."
    )
    if st.button("Reset to preset"):
        if preset == "Recommended":
            st.session_state['og_y1_compute_total_cmgr'], st.session_state['og_y2_compute_cmgr'], st.session_state['og_y3_compute_cmgr'], st.session_state['og_y4_compute_cmgr'], st.session_state['og_y5_compute_cmgr'] = rec_compute
            st.session_state['og_y1_storage_cmgr'], st.session_state['og_y2_storage_cmgr'], st.session_state['og_y3_storage_cmgr'], st.session_state['og_y4_storage_cmgr'], st.session_state['og_y5_storage_cmgr'] = rec_storage
            st.session_state['og_y1_data_transfer_cmgr'], st.session_state['og_y2_data_transfer_cmgr'], st.session_state['og_y3_data_transfer_cmgr'], st.session_state['og_y4_data_transfer_cmgr'], st.session_state['og_y5_data_transfer_cmgr'] = rec_dt
        elif preset == "Flat 0%":
            # Compute uses 'compute_total_cmgr' for Y1, but 'compute_cmgr' for Y2–Y5
            st.session_state['og_y1_compute_total_cmgr'] = 0.0
            for n in [2, 3, 4, 5]:
                st.session_state[f'og_y{n}_compute_cmgr'] = 0.0
            for k in ['og_y1_', 'og_y2_', 'og_y3_', 'og_y4_', 'og_y5_']:
                st.session_state[k + 'storage_cmgr'] = 0.0
                st.session_state[k + 'data_transfer_cmgr'] = 0.0
        elif preset == "Flat at Y1":
            c1 = float(st.session_state.get('og_y1_compute_total_cmgr', 0.0))
            s1 = float(st.session_state.get('og_y1_storage_cmgr', 0.0))
            d1 = float(st.session_state.get('og_y1_data_transfer_cmgr', 0.0))
            st.session_state['og_y2_compute_cmgr'] = st.session_state['og_y3_compute_cmgr'] = st.session_state['og_y4_compute_cmgr'] = st.session_state['og_y5_compute_cmgr'] = c1
            st.session_state['og_y2_storage_cmgr'] = st.session_state['og_y3_storage_cmgr'] = st.session_state['og_y4_storage_cmgr'] = st.session_state['og_y5_storage_cmgr'] = s1
            st.session_state['og_y2_data_transfer_cmgr'] = st.session_state['og_y3_data_transfer_cmgr'] = st.session_state['og_y4_data_transfer_cmgr'] = st.session_state['og_y5_data_transfer_cmgr'] = d1
        st.rerun()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("Compute (TOTAL)")
        c_y1 = float(st.session_state.get('og_y1_compute_total_cmgr', 0.0))
        c_y2d = float(st.session_state.get('og_y2_compute_cmgr', rec_compute[1]))
        c_y3d = float(st.session_state.get('og_y3_compute_cmgr', rec_compute[2]))
        c_y4d = float(st.session_state.get('og_y4_compute_cmgr', rec_compute[3]))
        c_y5d = float(st.session_state.get('og_y5_compute_cmgr', rec_compute[4]))
        st.session_state['og_y1_compute_total_cmgr'] = st.number_input("Y1", value=c_y1, step=0.01, key="og_custom_c_cy1")
        st.session_state['og_y2_compute_cmgr'] = st.number_input("Y2", value=c_y2d, step=0.01, key="og_custom_c_cy2")
        st.session_state['og_y3_compute_cmgr'] = st.number_input("Y3", value=c_y3d, step=0.01, key="og_custom_c_cy3")
        st.session_state['og_y4_compute_cmgr'] = st.number_input("Y4", value=c_y4d, step=0.01, key="og_custom_c_cy4")
        st.session_state['og_y5_compute_cmgr'] = st.number_input("Y5", value=c_y5d, step=0.01, key="og_custom_c_cy5")
    with c2:
        st.write("Storage")
        s_y1 = float(st.session_state.get('og_y1_storage_cmgr', 0.0))
        s_y2d = float(st.session_state.get('og_y2_storage_cmgr', rec_storage[1]))
        s_y3d = float(st.session_state.get('og_y3_storage_cmgr', rec_storage[2]))
        s_y4d = float(st.session_state.get('og_y4_storage_cmgr', rec_storage[3]))
        s_y5d = float(st.session_state.get('og_y5_storage_cmgr', rec_storage[4]))
        st.session_state['og_y1_storage_cmgr'] = st.number_input("Y1 ", value=s_y1, step=0.01, key="og_custom_s_cy1")
        st.session_state['og_y2_storage_cmgr'] = st.number_input("Y2 ", value=s_y2d, step=0.01, key="og_custom_s_cy2")
        st.session_state['og_y3_storage_cmgr'] = st.number_input("Y3 ", value=s_y3d, step=0.01, key="og_custom_s_cy3")
        st.session_state['og_y4_storage_cmgr'] = st.number_input("Y4 ", value=s_y4d, step=0.01, key="og_custom_s_cy4")
        st.session_state['og_y5_storage_cmgr'] = st.number_input("Y5 ", value=s_y5d, step=0.01, key="og_custom_s_cy5")
    with c3:
        st.write("Data Transfer")
        d_y1 = float(st.session_state.get('og_y1_data_transfer_cmgr', 0.0))
        d_y2d = float(st.session_state.get('og_y2_data_transfer_cmgr', rec_dt[1]))
        d_y3d = float(st.session_state.get('og_y3_data_transfer_cmgr', rec_dt[2]))
        d_y4d = float(st.session_state.get('og_y4_data_transfer_cmgr', rec_dt[3]))
        d_y5d = float(st.session_state.get('og_y5_data_transfer_cmgr', rec_dt[4]))
        st.session_state['og_y1_data_transfer_cmgr'] = st.number_input("Y1  ", value=d_y1, step=0.01, key="og_custom_dt_cy1")
        st.session_state['og_y2_data_transfer_cmgr'] = st.number_input("Y2  ", value=d_y2d, step=0.01, key="og_custom_dt_cy2")
        st.session_state['og_y3_data_transfer_cmgr'] = st.number_input("Y3  ", value=d_y3d, step=0.01, key="og_custom_dt_cy3")
        st.session_state['og_y4_data_transfer_cmgr'] = st.number_input("Y4  ", value=d_y4d, step=0.01, key="og_custom_dt_cy4")
        st.session_state['og_y5_data_transfer_cmgr'] = st.number_input("Y5  ", value=d_y5d, step=0.01, key="og_custom_dt_cy5")

    st.caption("Y1 uses your calculated CMGR when you press Apply; Y2–Y5 are manual unless you tick 'Copy Y1 to Y2–Y5'. These are monthly rates; baseline and forecasts apply them piecewise by year (months 1–12 use Y1, 13–24 use Y2, etc.).")
    if st.button("Apply Organic Growth Y1–Y5", key="og_apply_y1_y5"):
        st.success("Applied. Baseline and charts now use these Y1–Y5 rates.")
        
    # Create stacked bar chart visualization
    fig = go.Figure()
    categories = ['COMPUTE', 'STORAGE', 'OTHER', 'DATA_TRANSFER', 'PRIORITY_SUPPORT']
    colors = ['#11567F', '#29B5E8', '#71D3DC', '#FF9F36', '#7D44CF']
    for category, color in zip(categories, colors):
        fig.add_trace(go.Bar(x=filtered_df['MONTH'], y=filtered_df[category], name=category.replace('_', ' ').title(), marker_color=color, showlegend=True))
    fig.update_layout(title=dict(text=f"Monthly Revenue by Category ({start_month} to {end_month})", font=dict(size=18, color="#11567F"), x=0.5, xanchor='center'), xaxis_title="Month", yaxis_title="Revenue ($)", barmode='stack', xaxis_tickformat="%b %Y", xaxis_tickangle=-45, plot_bgcolor='white', paper_bgcolor='white', font=dict(color="#11567F"), height=500)
    st.plotly_chart(fig, use_container_width=True)
    # (Duplicate calculator removed to avoid redundancy; top calculator retained)
    # Store in session state for forecast overview
    st.session_state['og_compute_cmgr'] = growth_data['COMPUTE'].get('12 months', 0.0)
    st.session_state['og_storage_cmgr'] = growth_data['STORAGE'].get('12 months', 0.0)
    st.session_state['og_other_cmgr'] = growth_data['OTHER'].get('12 months', 0.0)
    st.session_state['og_data_transfer_cmgr'] = growth_data['DATA_TRANSFER'].get('12 months', 0.0)
    st.session_state['og_priority_support_cmgr'] = growth_data['PRIORITY_SUPPORT'].get('12 months', 0.0)
    st.session_state['og_total_cmgr'] = growth_data['TOTAL'].get('12 months', 0.0)
    
    # Removed duplicate analysis/table sections