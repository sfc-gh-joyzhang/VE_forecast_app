import plotly.graph_objects as go
import numpy as np
import pandas as pd
import io
import streamlit as st
from datetime import datetime, timedelta

def get_session_defaults():
    """Return default session state values for organic growth"""
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
    
    # Create month options for dropdowns
    available_months = sorted(df['MONTH'].dt.strftime('%Y-%m-01').unique())
    
    # Date range selector UI
    col1, col2 = st.columns(2)
    
    with col1:
        start_month = st.selectbox(
            "START MONTH",
            options=available_months,
            index=0,  # Default to earliest month (full 2 years)
            key="og_start_month_selector"
        )
        st.session_state.og_start_month = start_month
        
    with col2:
        end_month = st.selectbox(
            "END MONTH", 
            options=available_months,
            index=len(available_months) - 1,  # Default to latest month
            key="og_end_month_selector"
        )
        st.session_state.og_end_month = end_month
    
    # Filter data to selected range
    start_date = pd.to_datetime(start_month)
    end_date = pd.to_datetime(end_month)
    
    filtered_df = df[
        (df['MONTH'] >= start_date) & 
        (df['MONTH'] <= end_date)
    ].copy()
        
    if len(filtered_df) < 2:
        st.error(f"Insufficient data for growth calculation. Available dates: {df['MONTH'].min()} to {df['MONTH'].max()}")
        return
        
    # Create stacked bar chart visualization
    fig = go.Figure()
        
    # Add stacked bars for each category
    categories = ['COMPUTE', 'STORAGE', 'OTHER', 'DATA_TRANSFER', 'PRIORITY_SUPPORT']
    colors = ['#11567F', '#29B5E8', '#71D3DC', '#FF9F36', '#7D44CF']  # Snowflake colors
    
    for category, color in zip(categories, colors):
        fig.add_trace(go.Bar(
            x=filtered_df['MONTH'],
            y=filtered_df[category],
            name=category.replace('_', ' ').title(),
            marker_color=color,
            showlegend=True
        ))
        
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Monthly Revenue by Category ({start_month} to {end_month})",
            font=dict(size=18, color="#11567F"),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        barmode='stack',
        xaxis_tickformat="%b %Y",
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color="#11567F"),
        height=500
    )
        
    st.plotly_chart(fig, use_container_width=True)
        
    # Calculate growth rates
    def calculate_cmgr(start_value, end_value, months):
        """Calculate CMGR: (End/Start)^(1/months) - 1"""
        if start_value <= 0 or end_value <= 0 or months <= 0:
            return 0.0
        return (end_value / start_value) ** (1 / months) - 1
    
    def calculate_cagr(start_value, end_value, years):
        """Calculate CAGR: (End/Start)^(1/years) - 1"""
        if start_value <= 0 or end_value <= 0 or years <= 0:
            return 0.0
        return (end_value / start_value) ** (1 / years) - 1
    
    # Your requested categories (5 total)
    display_categories = ['COMPUTE', 'STORAGE', 'OTHER', 'DATA_TRANSFER', 'COMPUTE (TOTAL)']
    
    # Calculate different time period growth rates
    total_months = len(filtered_df) - 1
    years = total_months / 12
    
    # Growth rate periods you requested
    periods = [
        ('CAGR', years, 'years'),
        ('12 months', 12, 'months'), 
        ('6 months', 6, 'months'),
        ('3 months', 3, 'months'),
        ('1 month', 1, 'months')
    ]
    
    # Calculate growth rates for each category and period
    growth_data = {}
    for category in ['COMPUTE', 'STORAGE', 'OTHER', 'DATA_TRANSFER', 'PRIORITY_SUPPORT', 'TOTAL']:
        growth_data[category] = {}
        
        for period_name, period_length, period_type in periods:
            if period_type == 'years' and total_months >= 12:
                # Match Excel logic: simple percentage change over exactly 1 year
                # Find data for exactly 1 year ago from latest month
                latest_date = filtered_df['MONTH'].iloc[-1]
                year_ago_date = latest_date - pd.DateOffset(years=1)
                
                # Find closest match to 1 year ago
                year_ago_row = filtered_df[filtered_df['MONTH'] <= year_ago_date].iloc[-1] if len(filtered_df[filtered_df['MONTH'] <= year_ago_date]) > 0 else filtered_df.iloc[0]
                latest_row = filtered_df.iloc[-1]
                
                start_val = year_ago_row[category]
                end_val = latest_row[category]
                
                # Excel formula: =End/Start - 1 (simple percentage change)
                if start_val > 0:
                    growth_rate = (end_val / start_val - 1) * 100
                else:
                    growth_rate = 0.0
            elif period_type == 'months' and total_months >= period_length:
                # Use last N months for CMGR calculation
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
    
    # Display growth rates table
    st.markdown("### Growth Rate Analysis")
    
    # Create table
    col_headers = ['Metric'] + display_categories
    cols = st.columns(len(col_headers))
    
    # Header row
    for i, header in enumerate(col_headers):
        with cols[i]:
            st.markdown(f"**{header}**")
    
    # Data rows  
    for period_name, _, _ in periods:
        cols = st.columns(len(col_headers))
        with cols[0]:
            metric_label = f"{period_name} (%)" if period_name != 'CAGR' else "CAGR (%)"
            st.markdown(f"**{metric_label}**")
        
        # Data columns
        for i, category in enumerate(['COMPUTE', 'STORAGE', 'OTHER', 'DATA_TRANSFER', 'TOTAL']):
            with cols[i + 1]:
                # Map TOTAL to display as "COMPUTE (TOTAL)" but calculate using "TOTAL"
                display_cat = category
                    
                growth_rate = growth_data[display_cat].get(period_name, 0.0)
                color = "#28a745" if growth_rate > 0 else "#dc3545" if growth_rate < 0 else "#6c757d"
                
                st.markdown(f'<span style="color: {color}; font-weight: bold;">{growth_rate:.1f}%</span>', 
                           unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Custom Period CMGR Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        start_month = st.selectbox(
            "Start Month (Excel D22):",
            options=filtered_df['MONTH'].dt.strftime('%Y-%m-%d').tolist(),
            index=0,
            key='custom_start_month'
        )
    with col2:
        end_month = st.selectbox(
            "End Month (Excel F22):",
            options=filtered_df['MONTH'].dt.strftime('%Y-%m-%d').tolist(),
            index=len(filtered_df)-1,
            key='custom_end_month'
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
                custom_cols = st.columns(len(display_categories))
                
                for i, category in enumerate(['COMPUTE', 'STORAGE', 'OTHER', 'DATA_TRANSFER', 'TOTAL']):
                    with custom_cols[i]:
                        start_val = start_row[category].iloc[0]
                        end_val = end_row[category].iloc[0]
                        
                        if start_val > 0 and months_diff > 0:
                            custom_cmgr = ((end_val / start_val) ** (1 / months_diff) - 1) * 100
                        else:
                            custom_cmgr = 0.0
                        
                        # Store the calculated value
                        custom_cmgr_values[category] = custom_cmgr
                        
                        color = "#28a745" if custom_cmgr > 0 else "#dc3545" if custom_cmgr < 0 else "#6c757d"
                        display_name = "COMPUTE (TOTAL)" if category == 'TOTAL' else category
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 0.5rem; background-color: #f8f9fa; border-radius: 4px; border-left: 3px solid {color};">
                            <div style="font-weight: bold; color: {color}; font-size: 1.1rem;">{custom_cmgr:.2f}%</div>
                            <div style="font-size: 0.8rem; color: #6c757d;">{display_name}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Store in session state for the apply button
                st.session_state.custom_cmgr_values = custom_cmgr_values
                st.session_state.custom_cmgr_values['PRIORITY_SUPPORT'] = 0.0  # Not calculated for custom periods
                
            # Custom Period Apply Button - aligned like standard section
            st.markdown("")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button(
                    f"Apply Custom Period CMGR ({months_diff} months)",
                    help=f"Transfer custom period CMGR values from {start_month} to {end_month}",
                    use_container_width=True,
                    type="secondary",
                    key="apply_custom_cmgr"
                ):
                    # Use the calculated values directly
                    st.session_state.forecast_year1_compute_growth = custom_cmgr_values.get('COMPUTE', 0.0)
                    st.session_state.forecast_year1_storage_growth = custom_cmgr_values.get('STORAGE', 0.0)
                    st.session_state.forecast_year1_other_growth = custom_cmgr_values.get('OTHER', 0.0)
                    st.session_state.forecast_year1_data_transfer_growth = custom_cmgr_values.get('DATA_TRANSFER', 0.0)
                    st.session_state.forecast_year1_priority_support_growth = 0.0  # Not calculated for custom periods
                    
                    st.success(f"""
                    Applied Custom Period CMGR ({months_diff} months) to Forecast Overview:
                    - Compute: {custom_cmgr_values.get('COMPUTE', 0.0):.2f}% monthly
                    - Storage: {custom_cmgr_values.get('STORAGE', 0.0):.2f}% monthly  
                    - Other: {custom_cmgr_values.get('OTHER', 0.0):.2f}% monthly
                    - Data Transfer: {custom_cmgr_values.get('DATA_TRANSFER', 0.0):.2f}% monthly
                    - Priority Support: 0.00% monthly (not calculated for custom periods)
                    
                    *Period: {start_month} to {end_month} ({months_diff} months)*
                    """)
                
        else:
            st.warning("End month must be after start month")
    
    # Store in session state for forecast overview
    st.session_state['og_compute_cmgr'] = growth_data['COMPUTE'].get('12 months', 0.0)
    st.session_state['og_storage_cmgr'] = growth_data['STORAGE'].get('12 months', 0.0)
    st.session_state['og_other_cmgr'] = growth_data['OTHER'].get('12 months', 0.0)
    st.session_state['og_data_transfer_cmgr'] = growth_data['DATA_TRANSFER'].get('12 months', 0.0)
    st.session_state['og_priority_support_cmgr'] = growth_data['PRIORITY_SUPPORT'].get('12 months', 0.0)
    st.session_state['og_total_cmgr'] = growth_data['TOTAL'].get('12 months', 0.0)
    
    # Transfer to Forecast Overview button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button(
            "Apply Standard CMGR (12 months)",
            help="Transfer 12-month CMGR values from the analysis table above",
            use_container_width=True,
            type="primary"
        ):
            # Store calculated values for use in main forecast (using 12-month CMGR)
            st.session_state.forecast_year1_compute_growth = growth_data['COMPUTE'].get('12 months', 0.0)
            st.session_state.forecast_year1_storage_growth = growth_data['STORAGE'].get('12 months', 0.0)
            st.session_state.forecast_year1_other_growth = growth_data['OTHER'].get('12 months', 0.0)
            st.session_state.forecast_year1_data_transfer_growth = growth_data['DATA_TRANSFER'].get('12 months', 0.0)
            st.session_state.forecast_year1_priority_support_growth = growth_data['PRIORITY_SUPPORT'].get('12 months', 0.0)
            
            st.success(f"""
            Applied Standard CMGR (12 months) to Forecast Overview:
            - Compute: {growth_data['COMPUTE'].get('12 months', 0.0):.2f}% monthly
            - Storage: {growth_data['STORAGE'].get('12 months', 0.0):.2f}% monthly  
            - Other: {growth_data['OTHER'].get('12 months', 0.0):.2f}% monthly
            - Data Transfer: {growth_data['DATA_TRANSFER'].get('12 months', 0.0):.2f}% monthly
            - Priority Support: {growth_data['PRIORITY_SUPPORT'].get('12 months', 0.0):.2f}% monthly
            
            *Standard 12-month CMGR values will drive Year 1 of your main forecast.*
            """)
        
    # Show data table
    with st.expander("View Monthly Data"):
        display_df = filtered_df.copy()
        display_df['MONTH'] = display_df['MONTH'].dt.strftime('%b %Y')
        
        # Format currency columns
        currency_cols = ['COMPUTE', 'STORAGE', 'OTHER', 'DATA_TRANSFER', 'PRIORITY_SUPPORT', 'TOTAL']
        for col in currency_cols:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
            
        st.dataframe(display_df, use_container_width=True)