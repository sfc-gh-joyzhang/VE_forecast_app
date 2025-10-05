import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime


def get_session_defaults():
    return {
        "fo_start_month": None,
        "fo_end_month": None,
        "fo_pricing": {},
    }


def _fetch_billing_df(connection, schema: str, connection_type: str):
    sql = f"""
    SELECT DATE_TRUNC('MONTH', MONTH::DATE) AS MONTH,
           REVENUE_CATEGORY,
           REVENUE_GROUP,
           SUM(REVENUE)     AS REVENUE,
           SUM(CREDITS)     AS CREDITS,
           SUM(STORAGE_TB)  AS STORAGE_TB,
           SUM(TRANSFER_TB) AS TRANSFER_TB
    FROM {schema}.BILLING
    WHERE MONTH IS NOT NULL
      AND MONTH >= DATEADD('month', -24, DATE_TRUNC('month', CURRENT_DATE()))
      AND MONTH <  DATE_TRUNC('month', CURRENT_DATE())
    GROUP BY MONTH, REVENUE_CATEGORY, REVENUE_GROUP
    ORDER BY MONTH
    """
    if connection_type == "snowpark":
        bill = connection.sql(sql).to_pandas()
    else:
        from browser_connection import execute_sql
        bill = execute_sql(connection, sql)
    if not bill.empty:
        bill["MONTH"] = pd.to_datetime(bill["MONTH"]).dt.to_period("M").dt.to_timestamp()
    return bill


def _build_overview_components():
    # Organic Growth baseline from latest month
    latest_total = float(st.session_state.get("og_latest_month_total", 0.0))
    comp_pct = float(st.session_state.get("og_default_split_compute_pct", 0.0)) / 100.0
    stor_pct = float(st.session_state.get("og_default_split_storage_pct", 0.0)) / 100.0
    dt_pct = float(st.session_state.get("og_default_split_data_transfer_pct", 0.0)) / 100.0
    comp_val = latest_total * comp_pct
    stor_val = latest_total * stor_pct
    dt_val = latest_total * dt_pct

    # Allow overrides from the Overview yellow cells if present
    og_cmgr_compute = [
        float(st.session_state.get("fo_og_c_y1", st.session_state.get("og_y1_compute_total_cmgr", st.session_state.get("og_y1_compute_total_cmgr_fallback", 0.0)))),
        float(st.session_state.get("fo_og_c_y2", st.session_state.get("og_y2_compute_cmgr", 0.0))),
        float(st.session_state.get("fo_og_c_y3", st.session_state.get("og_y3_compute_cmgr", 1.0))),
        float(st.session_state.get("fo_og_c_y4", st.session_state.get("og_y4_compute_cmgr", 1.0))),
        float(st.session_state.get("fo_og_c_y5", st.session_state.get("og_y5_compute_cmgr", 1.0))),
    ]
    og_cmgr_storage = [
        float(st.session_state.get("fo_og_s_y1", st.session_state.get("og_y1_storage_cmgr", st.session_state.get("og_y1_storage_cmgr_fallback", 0.0)))),
        float(st.session_state.get("fo_og_s_y2", st.session_state.get("og_y2_storage_cmgr", 0.0))),
        float(st.session_state.get("fo_og_s_y3", st.session_state.get("og_y3_storage_cmgr", 1.2))),
        float(st.session_state.get("fo_og_s_y4", st.session_state.get("og_y4_storage_cmgr", 1.2))),
        float(st.session_state.get("fo_og_s_y5", st.session_state.get("og_y5_storage_cmgr", 1.2))),
    ]

    # Deal length (years) -> months
    deal_years = int(st.session_state.get("fo_deal_years", 5))
    horizon_months = max(12, min(60, deal_years * 12))
    baseline_growth_months = []
    comp_series = []
    stor_series = []
    for m in range(horizon_months):
        year_idx = min(m // 12, 4)
        if m > 0:
            comp_val *= 1 + og_cmgr_compute[year_idx] / 100.0
            stor_val *= 1 + og_cmgr_storage[year_idx] / 100.0
        baseline_growth_months.append(comp_val + stor_val + dt_val)
        comp_series.append(comp_val)
        stor_series.append(stor_val)

    # Expose series and horizon for downstream scenario calculations
    st.session_state["fo_horizon_months"] = horizon_months
    st.session_state["fo_comp_series"] = comp_series
    st.session_state["fo_stor_series"] = stor_series
    st.session_state["fo_base_series"] = baseline_growth_months

    baseline_with_growth_5y = sum(baseline_growth_months)
    baseline_steady_5y = latest_total * 60.0
    organic_growth_5y = baseline_with_growth_5y - baseline_steady_5y

    # New Use Cases monthly from Use Case Forecaster (fallback compute from rows if not rendered yet)
    ucf_monthly = st.session_state.get("ucf_monthly_df", pd.DataFrame())
    if ucf_monthly is None or ucf_monthly.empty:
        try:
            rows = st.session_state.get("ucf_use_cases", [])
            if rows:
                from slides.use_case_forecaster import _compute_forecast
                df_rows = pd.DataFrame(rows)
                monthly_df, _ = _compute_forecast(df_rows, horizon_months)
                st.session_state["ucf_monthly_df"] = monthly_df.copy()
                new_use_cases_5y = float(monthly_df["TOTAL"].sum())
            else:
                new_use_cases_5y = 0.0
        except Exception:
            new_use_cases_5y = 0.0
    else:
        new_use_cases_5y = float(ucf_monthly["TOTAL"].head(horizon_months).sum())

    # Optimizations monthly from Optimization Forecast
    # Helper to compute optimizations 5y if monthly not yet built
    def _first_of_month(d: datetime) -> datetime:
        return datetime(d.year, d.month, 1)

    def _months_between(start: datetime, end: datetime) -> int:
        return (end.year - start.year) * 12 + (end.month - start.month)

    def _curve_ramp_factor(curve: str, months_since_start: int, ramp_months: int) -> float:
        if ramp_months <= 0:
            return 1.0
        m = max(0, min(months_since_start, ramp_months))
        x = m / float(ramp_months)
        if curve == "Slowest Ramp":
            return x ** 3
        if curve == "Slow Ramp":
            return x ** 2
        if curve == "Fast Ramp":
            return 1 - (1 - x) ** 2
        if curve == "Fastest Ramp":
            return 1 - (1 - x) ** 3
        return x  # Linear/Manual

    def _compute_opt_5y_from_rows(rows: list) -> float:
        months = []
        start = _first_of_month(datetime.today())
        for i in range(60):
            y = start.year + (start.month - 1 + i) // 12
            m = (start.month - 1 + i) % 12 + 1
            months.append(datetime(y, m, 1))
        values = [0.0 for _ in months]
        for r in rows:
            try:
                s = _first_of_month(pd.to_datetime(r.get("implementation_start")).to_pydatetime())
            except Exception:
                continue
            ramp_months = int(r.get("implementation_months", 0) or 0)
            curve = str(r.get("ramp_curve", "Linear Ramp"))
            annual = float(r.get("annualized_savings", 0.0) or 0.0)
            base = annual / 12.0
            split_total = (
                float(r.get("split_compute_pct", 0.0) or 0.0)
                + float(r.get("split_storage_pct", 0.0) or 0.0)
                + float(r.get("split_dt_pct", 0.0) or 0.0)
            ) / 100.0
            for i, m in enumerate(months):
                if m < s:
                    continue
                ms = _months_between(s, m)
                ramp = _curve_ramp_factor(curve, ms, ramp_months)
                values[i] += -(base * ramp * split_total)
        return float(abs(sum(values)))

    opt_monthly = st.session_state.get("opt_monthly_df", pd.DataFrame())
    rows = st.session_state.get("opt_rows", [])
    if (opt_monthly is None or opt_monthly.empty) and rows:
        optimizations_5y = _compute_opt_5y_from_rows(rows[:]) * (horizon_months / 60.0)
    elif opt_monthly is not None and not opt_monthly.empty:
        col = "SAVINGS" if "SAVINGS" in opt_monthly.columns else ("TOTAL" if "TOTAL" in opt_monthly.columns else None)
        optimizations_5y = float(abs(opt_monthly[col].head(horizon_months).sum())) if col else 0.0
    else:
        optimizations_5y = float(st.session_state.get("opt_5y_selected", 0.0)) * (horizon_months / 60.0)

    # No manual discount input in Overview; keep as 0 unless set elsewhere
    # Calculate default discount amount over 5y (list minus actual) so it's visible by default
    fp = st.session_state.get("fo_pricing", {})
    before_c = float(fp.get("compute_discount", 0.0))  # decimal
    before_s = float(fp.get("storage_discount", 0.0))
    discount_current_total = 0.0
    for i in range(horizon_months):
        c_rev = float(comp_series[i])
        s_rev = float(stor_series[i])
        c_list = c_rev / (1.0 - before_c) if (1.0 - before_c) > 1e-9 else c_rev
        s_list = s_rev / (1.0 - before_s) if (1.0 - before_s) > 1e-9 else s_rev
        discount_current_total += (c_list - c_rev) + (s_list - s_rev)
    # Negative in waterfall to represent discount reducing contract value
    discount_5y = -abs(discount_current_total)

    contract_value_5y = baseline_steady_5y + organic_growth_5y + new_use_cases_5y - optimizations_5y - discount_5y

    return baseline_steady_5y, organic_growth_5y, new_use_cases_5y, optimizations_5y, discount_5y, contract_value_5y


# Helper to compute optimization savings from LOW/HIGH rows, summed to horizon
def compute_opt_5y_from_rows(rows: list, horizon_months: int) -> float:
    def _first_of_month(d: datetime) -> datetime:
        return datetime(d.year, d.month, 1)
    def _months_between(start: datetime, end: datetime) -> int:
        return (end.year - start.year) * 12 + (end.month - start.month)
    def _curve_ramp_factor(curve: str, months_since_start: int, ramp_months: int) -> float:
        if ramp_months <= 0:
            return 1.0
        m = max(0, min(months_since_start, ramp_months))
        x = m / float(ramp_months)
        return x
    months = []
    start = _first_of_month(datetime.today())
    for i in range(horizon_months):
        y = start.year + (start.month - 1 + i) // 12
        m = (start.month - 1 + i) % 12 + 1
        months.append(datetime(y, m, 1))
    values = [0.0 for _ in months]
    for r in rows or []:
        try:
            s = _first_of_month(pd.to_datetime(r.get("implementation_start")).to_pydatetime())
        except Exception:
            continue
        ramp_months = int(r.get("implementation_months", 0) or 0)
        curve = str(r.get("ramp_curve", "Linear Ramp"))
        annual = float(r.get("annualized_savings", 0.0) or 0.0)
        base = annual / 12.0
        split_total = (
            float(r.get("split_compute_pct", 0.0) or 0.0)
            + float(r.get("split_storage_pct", 0.0) or 0.0)
            + float(r.get("split_dt_pct", 0.0) or 0.0)
        ) / 100.0
        for i, m in enumerate(months):
            if m < s:
                continue
            ms = _months_between(s, m)
            ramp = _curve_ramp_factor(curve, ms, ramp_months)
            values[i] += -(base * ramp * split_total)
    return float(abs(sum(values)))


# Helper to compute optimization monthly series from rows (negative monthly savings)
def compute_opt_monthly_from_rows(rows: list, horizon_months: int) -> list:
    def _first_of_month(d: datetime) -> datetime:
        return datetime(d.year, d.month, 1)
    def _months_between(start: datetime, end: datetime) -> int:
        return (end.year - start.year) * 12 + (end.month - start.month)
    def _curve_ramp_factor(curve: str, months_since_start: int, ramp_months: int) -> float:
        if ramp_months <= 0:
            return 1.0
        m = max(0, min(months_since_start, ramp_months))
        x = m / float(ramp_months)
        return x
    months = []
    start = _first_of_month(datetime.today())
    for i in range(horizon_months):
        y = start.year + (start.month - 1 + i) // 12
        m = (start.month - 1 + i) % 12 + 1
        months.append(datetime(y, m, 1))
    values = [0.0 for _ in months]
    for r in rows or []:
        try:
            s = _first_of_month(pd.to_datetime(r.get("implementation_start")).to_pydatetime())
        except Exception:
            continue
        ramp_months = int(r.get("implementation_months", 0) or 0)
        curve = str(r.get("ramp_curve", "Linear Ramp"))
        annual = float(r.get("annualized_savings", 0.0) or 0.0)
        base = annual / 12.0
        split_total = (
            float(r.get("split_compute_pct", 0.0) or 0.0)
            + float(r.get("split_storage_pct", 0.0) or 0.0)
            + float(r.get("split_dt_pct", 0.0) or 0.0)
        ) / 100.0
        for i, m in enumerate(months):
            if m < s:
                continue
            ms = _months_between(s, m)
            ramp = _curve_ramp_factor(curve, ms, ramp_months)
            values[i] += -(base * ramp * split_total)
    return values

def render(connection, selected_customer_schema, validated_info, connection_type="snowpark"):
    st.markdown("*High-level summary of forecast components across modules.*")

    # Pricing/Discount inputs from billing table
    try:
        schema = selected_customer_schema
        bill = _fetch_billing_df(connection, schema, connection_type)
        if not bill.empty:
            months = sorted(bill["MONTH"].unique())
            col1, col2 = st.columns(2)
            with col1:
                start_m = st.selectbox("Start Month", options=[m.strftime('%Y-%m-01') for m in months], index=0, key="fo_start_m")
            with col2:
                end_m = st.selectbox("End Month", options=[m.strftime('%Y-%m-01') for m in months], index=len(months)-1, key="fo_end_m")
            sm = pd.to_datetime(start_m).to_period("M").to_timestamp()
            em = pd.to_datetime(end_m).to_period("M").to_timestamp()
            mask = (bill["MONTH"] >= sm) & (bill["MONTH"] <= em)
            window = bill.loc[mask].copy()

            # Aggregate actuals by category
            actual_compute = float(window.loc[window["REVENUE_CATEGORY"] == "Compute", "REVENUE"].sum())
            actual_storage = float(window.loc[window["REVENUE_CATEGORY"] == "Storage", "REVENUE"].sum())
            actual_dt = float(window.loc[window["REVENUE_CATEGORY"] == "Data Transfer", "REVENUE"].sum())
            actual_ps = float(window.loc[window["REVENUE_CATEGORY"] == "Priority Support", "REVENUE"].sum())

            # List totals (fallback to CREDITS for Compute, Storage list to STORAGE_LIST_PRICE if present)
            if "CREDIT_LIST_PRICE" in window.columns:
                list_compute = float(window.loc[window["REVENUE_CATEGORY"] == "Compute", "CREDIT_LIST_PRICE"].sum())
            elif "CREDITS" in window.columns:
                list_compute = float(window.loc[window["REVENUE_CATEGORY"] == "Compute", "CREDITS"].sum())
            else:
                list_compute = 0.0

            if "STORAGE_LIST_PRICE" in window.columns:
                list_storage = float(window.loc[window["REVENUE_CATEGORY"] == "Storage", "STORAGE_LIST_PRICE"].sum())
            else:
                # As a conservative fallback, treat actual as list → 0% discount
                list_storage = actual_storage

            # Data transfer mixed rate ($/TB) over window
            transfer_tb = 0.0
            if "TRANSFER_TB" in window.columns:
                transfer_tb = float(window.loc[:, "TRANSFER_TB"].sum())
            dt_rate = (actual_dt / transfer_tb) if transfer_tb > 0 else 0.0

            # Ensure lists are comparable to revenue; if missing, default to actual → 0% discount
            if list_compute <= 0:
                list_compute = actual_compute
            if list_storage <= 0:
                list_storage = actual_storage
            # Discounts (1 - actual/list)
            compute_disc = 1.0 - (actual_compute / list_compute) if list_compute > 0 else 0.0
            storage_disc = 1.0 - (actual_storage / list_storage) if list_storage > 0 else 0.0

            st.session_state["fo_pricing"] = {
                "compute_discount": compute_disc,
                "storage_discount": storage_disc,
                "data_transfer_rate": dt_rate,
                "priority_support": actual_ps,
                "compute_actual": actual_compute,
                "storage_actual": actual_storage,
                "dt_actual": actual_dt,
                "compute_list": list_compute,
                "storage_list": list_storage,
                "transfer_tb": transfer_tb,
                "seed_transfer_tb": float(window.loc[window["MONTH"] == sm, "TRANSFER_TB"].sum()) if "TRANSFER_TB" in window.columns else 0.0,
                "window_months": int(len(window["MONTH"].unique())) if not window.empty else 0,
                "start": sm,
                "end": em,
            }
            # Expose BILL history for Trend fallback
            try:
                bill_hist = (
                    bill.groupby("MONTH")["REVENUE"].sum().reset_index().rename(columns={"REVENUE": "TOTAL"})
                )
                st.session_state["fo_bill_hist"] = bill_hist
            except Exception:
                pass

            # Initialize AFTER defaults once (no UI yet)
            if "fo_after_compute_discount" not in st.session_state:
                st.session_state["fo_after_compute_discount"] = float(compute_disc) * 100.0
            if "fo_after_storage_discount" not in st.session_state:
                st.session_state["fo_after_storage_discount"] = float(storage_disc) * 100.0
            if "fo_after_dt_rate" not in st.session_state:
                st.session_state["fo_after_dt_rate"] = float(dt_rate)
            if "fo_after_priority_support" not in st.session_state:
                st.session_state["fo_after_priority_support"] = float(actual_ps)

            # Seed Organic Growth defaults if missing so Overview always has a baseline
            try:
                latest_m = bill["MONTH"].max()
                latest_rows = bill.loc[bill["MONTH"] == latest_m]
                total_latest = float(latest_rows["REVENUE"].sum())
                if st.session_state.get("og_latest_month_total", 0.0) == 0.0 and total_latest > 0:
                    st.session_state["og_latest_month_total"] = total_latest
                    comp_latest = float(latest_rows.loc[latest_rows["REVENUE_CATEGORY"] == "Compute", "REVENUE"].sum())
                    stor_latest = float(latest_rows.loc[latest_rows["REVENUE_CATEGORY"] == "Storage", "REVENUE"].sum())
                    other_latest = float(latest_rows.loc[~latest_rows["REVENUE_CATEGORY"].isin(["Compute","Storage","Data Transfer","Priority Support"]), "REVENUE"].sum())
                    dt_latest = float(latest_rows.loc[latest_rows["REVENUE_CATEGORY"] == "Data Transfer", "REVENUE"].sum())
                    # Compute (TOTAL) = Compute + Other
                    comp_pct = (comp_latest + other_latest) / total_latest * 100.0 if total_latest > 0 else 0.0
                    stor_pct = stor_latest / total_latest * 100.0 if total_latest > 0 else 0.0
                    dt_pct = dt_latest / total_latest * 100.0 if total_latest > 0 else 0.0
                    st.session_state['og_default_split_compute_pct'] = comp_pct
                    st.session_state['og_default_split_storage_pct'] = stor_pct
                    st.session_state['og_default_split_data_transfer_pct'] = dt_pct
                # If Y1 baseline growth not set, initialize from computed/fallbacks
                if 'og_rates_initialized' not in st.session_state:
                    st.session_state['og_y1_compute_total_cmgr'] = float(st.session_state.get('og_y1_compute_total_cmgr', st.session_state.get('og_y1_compute_total_cmgr_fallback', 0.0)))
                    st.session_state['og_y1_storage_cmgr'] = float(st.session_state.get('og_y1_storage_cmgr', st.session_state.get('og_y1_storage_cmgr_fallback', 0.0)))
                    st.session_state['og_y1_data_transfer_cmgr'] = float(st.session_state.get('og_y1_data_transfer_cmgr', st.session_state.get('og_y1_data_transfer_cmgr_fallback', 0.0)))
                    st.session_state['og_y2_compute_cmgr'] = st.session_state.get('og_y2_compute_cmgr', 0.8)
                    st.session_state['og_y3_compute_cmgr'] = st.session_state.get('og_y3_compute_cmgr', 1.0)
                    st.session_state['og_y4_compute_cmgr'] = st.session_state.get('og_y4_compute_cmgr', 1.0)
                    st.session_state['og_y5_compute_cmgr'] = st.session_state.get('og_y5_compute_cmgr', 1.0)
                    st.session_state['og_y2_storage_cmgr'] = st.session_state.get('og_y2_storage_cmgr', 0.8)
                    st.session_state['og_y3_storage_cmgr'] = st.session_state.get('og_y3_storage_cmgr', 1.2)
                    st.session_state['og_y4_storage_cmgr'] = st.session_state.get('og_y4_storage_cmgr', 1.2)
                    st.session_state['og_y5_storage_cmgr'] = st.session_state.get('og_y5_storage_cmgr', 1.2)
                    st.session_state['og_y2_data_transfer_cmgr'] = st.session_state.get('og_y2_data_transfer_cmgr', 0.0)
                    st.session_state['og_y3_data_transfer_cmgr'] = st.session_state.get('og_y3_data_transfer_cmgr', 0.0)
                    st.session_state['og_y4_data_transfer_cmgr'] = st.session_state.get('og_y4_data_transfer_cmgr', 0.0)
                    st.session_state['og_y5_data_transfer_cmgr'] = st.session_state.get('og_y5_data_transfer_cmgr', 0.0)
                    st.session_state['og_rates_initialized'] = True
            except Exception:
                pass

            # (UI moved below)

            # No 'Further' inputs; Discount uses AFTER−BEFORE directly
    except Exception as e:
        st.caption(f"Pricing source not available: {str(e)[:120]}")

    base, org_uplift, new_cases, opts, disc, total = _build_overview_components()

    # Discount effect using AFTER − BEFORE (no 'Further' inputs)
    try:
        horizon_months = int(st.session_state.get("fo_horizon_months", 60))
        comp_series = st.session_state.get("fo_comp_series", [])
        stor_series = st.session_state.get("fo_stor_series", [])
        fp = st.session_state.get("fo_pricing", {})
        before_c = float(fp.get("compute_discount", 0.0))
        before_s = float(fp.get("storage_discount", 0.0))
        dt_rate_before = float(fp.get("data_transfer_rate", 0.0))
        seed_tb = float(fp.get("seed_transfer_tb", 0.0))

        after_c = float(st.session_state.get("fo_after_compute_discount", before_c*100.0)) / 100.0
        after_s = float(st.session_state.get("fo_after_storage_discount", before_s*100.0)) / 100.0
        after_dt = float(st.session_state.get("fo_after_dt_rate", dt_rate_before))
        delta_c = after_c - before_c
        delta_s = after_s - before_s
        delta_dt = after_dt - dt_rate_before

        # Build monthly DT TB series: Kt = Kt-1 + newUseCaseDT$/dt_rate + Kt-1*dt_growth
        dt_tb_series = []
        dt_prev = seed_tb
        dt_cmgr = [
            float(st.session_state.get("og_y1_data_transfer_cmgr", st.session_state.get("og_y1_data_transfer_cmgr_fallback", 0.0)))/100.0,
            float(st.session_state.get("og_y2_data_transfer_cmgr", 0.0))/100.0,
            float(st.session_state.get("og_y3_data_transfer_cmgr", 0.0))/100.0,
            float(st.session_state.get("og_y4_data_transfer_cmgr", 0.0))/100.0,
            float(st.session_state.get("og_y5_data_transfer_cmgr", 0.0))/100.0,
        ]
        ucf_dt_series = []
        try:
            if "DATA_TRANSFER" in st.session_state.get("ucf_monthly_df", pd.DataFrame()).columns:
                ucf_dt_series = list(st.session_state["ucf_monthly_df"]["DATA_TRANSFER"].head(horizon_months))
            else:
                ucf_dt_series = [0.0 for _ in range(horizon_months)]
        except Exception:
            ucf_dt_series = [0.0 for _ in range(horizon_months)]
        for i in range(horizon_months):
            year_idx = min(i // 12, 4)
            growth = dt_prev * dt_cmgr[year_idx]
            new_dt_tb = float(ucf_dt_series[i]) / (dt_rate_before if dt_rate_before > 0 else 1.0)
            dt_curr = dt_prev + growth + new_dt_tb
            dt_tb_series.append(dt_curr)
            dt_prev = dt_curr

        # Compute discount effect for selected scenario
        discount_total = 0.0
        for i in range(min(horizon_months, len(comp_series), len(stor_series))):
            c_rev = float(comp_series[i])
            s_rev = float(stor_series[i])
            c_list = c_rev / (1.0 - before_c) if (1.0 - before_c) > 1e-9 else c_rev
            s_list = s_rev / (1.0 - before_s) if (1.0 - before_s) > 1e-9 else s_rev
            discount_total += c_list * delta_c + s_list * delta_s + dt_tb_series[i] * delta_dt
        disc = -abs(discount_total)

        # Scenario-specific optimizations
        # Keep optimizations equal to LOW by default here; Low/High range charts below will show both
        rows = st.session_state.get("opt_rows", [])
        if rows:
            low_rows = [r for r in rows if str(r.get("range", "")).upper() == "LOW"]
            opts = compute_opt_5y_from_rows(low_rows, horizon_months) if low_rows else 0.0
    except Exception:
        pass

    # Charts first: Forecast Trend + Low/High waterfalls
    # Low/High composition waterfalls (top)
    try:
        with st.spinner("Computing overview charts..."):
            horizon_months = int(st.session_state.get("fo_horizon_months", 60))
        comp_series = st.session_state.get("fo_comp_series", [])
        stor_series = st.session_state.get("fo_stor_series", [])
        fp = st.session_state.get("fo_pricing", {})
        before_c = float(fp.get("compute_discount", 0.0))
        before_s = float(fp.get("storage_discount", 0.0))
        dt_rate_before = float(fp.get("data_transfer_rate", 0.0))
        seed_tb = float(fp.get("seed_transfer_tb", 0.0))

        def _fo_build_dt_series():
            dt_prev = seed_tb
            series = []
            dt_cmgr = [
                float(st.session_state.get("og_y1_data_transfer_cmgr", st.session_state.get("og_y1_data_transfer_cmgr_fallback", 0.0)))/100.0,
                float(st.session_state.get("og_y2_data_transfer_cmgr", 0.0))/100.0,
                float(st.session_state.get("og_y3_data_transfer_cmgr", 0.0))/100.0,
                float(st.session_state.get("og_y4_data_transfer_cmgr", 0.0))/100.0,
                float(st.session_state.get("og_y5_data_transfer_cmgr", 0.0))/100.0,
            ]
            ucf_dt_series = []
            try:
                if "DATA_TRANSFER" in st.session_state.get("ucf_monthly_df", pd.DataFrame()).columns:
                    ucf_dt_series = list(st.session_state["ucf_monthly_df"]["DATA_TRANSFER"].head(horizon_months))
                else:
                    ucf_dt_series = [0.0 for _ in range(horizon_months)]
            except Exception:
                ucf_dt_series = [0.0 for _ in range(horizon_months)]
            for i in range(horizon_months):
                year_idx = min(i // 12, 4)
                growth = dt_prev * dt_cmgr[year_idx]
                new_dt_tb = float(ucf_dt_series[i]) / (dt_rate_before if dt_rate_before > 0 else 1.0)
                dt_curr = dt_prev + growth + new_dt_tb
                series.append(dt_curr)
                dt_prev = dt_curr
            return series

        def _fo_discount_total_for(delta_c, delta_s, delta_dt):
            dt_tb_series = _fo_build_dt_series()
            total = 0.0
            for i in range(min(horizon_months, len(comp_series), len(stor_series))):
                c_rev = float(comp_series[i])
                s_rev = float(stor_series[i])
                c_list = c_rev / (1.0 - before_c) if (1.0 - before_c) > 1e-9 else c_rev
                s_list = s_rev / (1.0 - before_s) if (1.0 - before_s) > 1e-9 else s_rev
                total += c_list * delta_c + s_list * delta_s + dt_tb_series[i] * delta_dt
            return -abs(total)

        # Use AFTER − BEFORE deltas for discount effect consistently across charts
        after_c = float(st.session_state.get("fo_after_compute_discount", before_c * 100.0)) / 100.0
        after_s = float(st.session_state.get("fo_after_storage_discount", before_s * 100.0)) / 100.0
        after_dt = float(st.session_state.get("fo_after_dt_rate", dt_rate_before))
        delta_c = after_c - before_c
        delta_s = after_s - before_s
        delta_dt = after_dt - dt_rate_before
        disc_low = _fo_discount_total_for(delta_c, delta_s, delta_dt)
        disc_high = disc_low

        # Optimizations Low/High from rows
        rows = st.session_state.get("opt_rows", [])
        low_rows = [r for r in rows if str(r.get("range", "")).upper() == "LOW"]
        high_rows = [r for r in rows if str(r.get("range", "")).upper() == "HIGH"]
        opt_low = compute_opt_5y_from_rows(low_rows, horizon_months) if low_rows else 0.0
        opt_high = compute_opt_5y_from_rows(high_rows, horizon_months) if high_rows else 0.0

        # Two waterfalls side-by-side
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Low Range")
            wf_low = go.Figure(go.Waterfall(
                name="",
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "relative", "total"],
                x=["Baseline", "Organic Growth", "New Use Cases", "Optimizations", "Discount", "Contract Value"],
                textposition="outside",
                y=[base, org_uplift, new_cases, -abs(opt_low), disc_low, base + org_uplift + new_cases - abs(opt_low) + disc_low],
                connector={"line": {"color": "#11567F"}},
            ))
            wf_low.update_layout(height=300)
            st.plotly_chart(wf_low, use_container_width=True)
        with c2:
            st.markdown("#### High Range")
            wf_high = go.Figure(go.Waterfall(
                name="",
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "relative", "total"],
                x=["Baseline", "Organic Growth", "New Use Cases", "Optimizations", "Discount", "Contract Value"],
                textposition="outside",
                y=[base, org_uplift, new_cases, -abs(opt_high), disc_high, base + org_uplift + new_cases - abs(opt_high) + disc_high],
                connector={"line": {"color": "#11567F"}},
            ))
            wf_high.update_layout(height=300)
            st.plotly_chart(wf_high, use_container_width=True)
    except Exception:
        pass

    # Forecast Trend (top)
    try:
        with st.spinner("Computing trend..."):
            og_hist = st.session_state.get("og_monthly_df", pd.DataFrame()).copy()
            if og_hist is None or og_hist.empty:
                og_hist = st.session_state.get("fo_bill_hist", pd.DataFrame()).copy()
            if not og_hist.empty:
                og_hist = og_hist.sort_values("MONTH")
                hist_tail = og_hist.tail(12).copy()
                hist_tail.rename(columns={"TOTAL": "BILL"}, inplace=True)
                last_hist_month = hist_tail["MONTH"].max()

                def _add_month(d: datetime, n: int) -> datetime:
                    y = d.year + (d.month - 1 + n) // 12
                    m = (d.month - 1 + n) % 12 + 1
                    return datetime(y, m, 1)

                horizon_months = int(st.session_state.get("fo_horizon_months", 60))
                forecast_months = [_add_month(last_hist_month, i) for i in range(1, horizon_months+1)]

                base_series = st.session_state.get("fo_base_series", [])
                nuc_df = st.session_state.get("ucf_monthly_df", pd.DataFrame())
                new_uc_series = list(nuc_df["TOTAL"].head(horizon_months)) if (not nuc_df.empty and "TOTAL" in nuc_df.columns) else [0.0 for _ in range(horizon_months)]
                rows = st.session_state.get("opt_rows", [])
                low_rows = [r for r in rows if str(r.get("range", "")).upper() == "LOW"]
                opt_monthly = compute_opt_monthly_from_rows(low_rows, horizon_months) if low_rows else [0.0 for _ in range(horizon_months)]

                fp = st.session_state.get("fo_pricing", {})
                before_c = float(fp.get("compute_discount", 0.0))
                before_s = float(fp.get("storage_discount", 0.0))
                dt_rate_before = float(fp.get("data_transfer_rate", 0.0))
                seed_tb = float(fp.get("seed_transfer_tb", 0.0))
                after_c = float(st.session_state.get("fo_after_compute_discount", before_c*100.0)) / 100.0
                after_s = float(st.session_state.get("fo_after_storage_discount", before_s*100.0)) / 100.0
                after_dt = float(st.session_state.get("fo_after_dt_rate", dt_rate_before))
                delta_c = after_c - before_c
                delta_s = after_s - before_s
                delta_dt = after_dt - dt_rate_before

                dt_prev = seed_tb
                dt_tb_series = []
                dt_cmgr = [
                    float(st.session_state.get("og_y1_data_transfer_cmgr", st.session_state.get("og_y1_data_transfer_cmgr_fallback", 0.0)))/100.0,
                    float(st.session_state.get("og_y2_data_transfer_cmgr", 0.0))/100.0,
                    float(st.session_state.get("og_y3_data_transfer_cmgr", 0.0))/100.0,
                    float(st.session_state.get("og_y4_data_transfer_cmgr", 0.0))/100.0,
                    float(st.session_state.get("og_y5_data_transfer_cmgr", 0.0))/100.0,
                ]
                ucf_dt_series = []
                try:
                    if "DATA_TRANSFER" in nuc_df.columns:
                        ucf_dt_series = list(nuc_df["DATA_TRANSFER"].head(horizon_months))
                    else:
                        ucf_dt_series = [0.0 for _ in range(horizon_months)]
                except Exception:
                    ucf_dt_series = [0.0 for _ in range(horizon_months)]
                for i in range(horizon_months):
                    year_idx = min(i // 12, 4)
                    growth = dt_prev * dt_cmgr[year_idx]
                    new_dt_tb = float(ucf_dt_series[i]) / (dt_rate_before if dt_rate_before > 0 else 1.0)
                    dt_curr = dt_prev + growth + new_dt_tb
                    dt_tb_series.append(dt_curr)
                    dt_prev = dt_curr

                disc_monthly = []
                for i in range(horizon_months):
                    c_rev = float(comp_series[i])
                    s_rev = float(stor_series[i])
                    c_list = c_rev / (1.0 - before_c) if (1.0 - before_c) > 1e-9 else c_rev
                    s_list = s_rev / (1.0 - before_s) if (1.0 - before_s) > 1e-9 else s_rev
                    disc_monthly.append(-(c_list * delta_c + s_list * delta_s + dt_tb_series[i] * delta_dt))

                forecast_amt = [base_series[i] + new_uc_series[i] + opt_monthly[i] + disc_monthly[i] for i in range(horizon_months)]

                trend_df = pd.DataFrame({
                    "MONTH": list(hist_tail["MONTH"]) + forecast_months,
                    "BILL": list(hist_tail["BILL"]) + [None] * horizon_months,
                    "FORECAST AMT": [None] * len(hist_tail) + forecast_amt,
                })

                fig_trend = go.Figure()
                fig_trend.add_bar(x=trend_df["MONTH"], y=trend_df["BILL"], name="BILL", marker_color="#888888")
                fig_trend.add_bar(x=trend_df["MONTH"], y=trend_df["FORECAST AMT"], name="FORECAST AMT", marker_color="#29B5E8")
                fig_trend.update_layout(title="Forecast Trend", barmode="group", height=300)
                st.plotly_chart(fig_trend, use_container_width=True)
    except Exception:
        pass

    st.markdown("### Forecast Overview (Deal Settings)")
    st.caption("Optimizations auto-sync from Optimization Forecast; Use Cases auto-sync from Use Case Forecaster.")
    st.session_state["fo_deal_years"] = st.selectbox("Deal Length (years)", options=[1,2,3,4,5], index=[1,2,3,4,5].index(int(st.session_state.get("fo_deal_years", 5))), key="fo_deal_len")

    # Pricing and Growth Inputs (now below charts)
    try:
        before = st.session_state.get("fo_pricing", {})
        cbefore, cafter = st.columns(2)
        with cbefore:
            st.markdown("**BEFORE (from billing window)**")
            st.number_input("Compute Discount %", value=float(before.get("compute_discount", 0.0)) * 100.0, step=0.01, disabled=True, key="fo_before_compute_disc")
            st.number_input("Storage Discount %", value=float(before.get("storage_discount", 0.0)) * 100.0, step=0.01, disabled=True, key="fo_before_storage_disc")
            st.number_input("Data Transfer $/TB", value=float(before.get("data_transfer_rate", 0.0)), step=0.01, disabled=True, key="fo_before_dt_rate")
            st.number_input("Priority Support ($, window)", value=float(before.get("priority_support", 0.0)), step=100.0, disabled=True, key="fo_before_ps")
        with cafter:
            st.markdown("**AFTER (editable)**")
            st.number_input("Compute Discount %  ", value=float(st.session_state.get("fo_after_compute_discount", 0.0)), step=0.01, key="fo_after_compute_discount")
            st.number_input("Storage Discount %  ", value=float(st.session_state.get("fo_after_storage_discount", 0.0)), step=0.01, key="fo_after_storage_discount")
            st.number_input("Data Transfer $/TB  ", value=float(st.session_state.get("fo_after_dt_rate", 0.0)), step=0.01, key="fo_after_dt_rate")
            st.number_input("Priority Support ($, window)  ", value=float(st.session_state.get("fo_after_priority_support", 0.0)), step=100.0, key="fo_after_priority_support")

        st.markdown("#### Organic Growth CMGR (%)")
        ogc1, ogc2 = st.columns(2)
        with ogc1:
            st.markdown("Compute")
            for y in range(1,6):
                key = f"fo_og_c_y{y}"
                default = float(st.session_state.get(f"og_y{y}_compute_total_cmgr", st.session_state.get(f"og_y{y}_compute_total_cmgr_fallback", 0.0))) if y==1 else float(st.session_state.get(f"og_y{y}_compute_cmgr", 0.0 if y==2 else 1.0))
                st.number_input(f"Y{y}", value=float(st.session_state.get(key, default)), step=0.01, key=key)
        with ogc2:
            st.markdown("Storage")
            for y in range(1,6):
                key = f"fo_og_s_y{y}"
                default = float(st.session_state.get(f"og_y{y}_storage_cmgr", st.session_state.get(f"og_y{y}_storage_cmgr_fallback", 0.0))) if y==1 else float(st.session_state.get(f"og_y{y}_storage_cmgr", 0.0 if y==2 else 1.2))
                st.number_input(f"Y{y} ", value=float(st.session_state.get(key, default)), step=0.01, key=key)

        with st.expander("Show your math (list vs actual)", expanded=False):
            math_df = pd.DataFrame({
                "Metric": ["Compute Actual", "Compute List", "Storage Actual", "Storage List", "DT $", "Transfer TB", "Priority Support"],
                "Value": [
                    before.get("compute_actual", 0.0), before.get("compute_list", 0.0), before.get("storage_actual", 0.0), before.get("storage_list", 0.0), before.get("dt_actual", 0.0), before.get("transfer_tb", 0.0), before.get("priority_support", 0.0),
                ],
            })
            math_df["Value"] = math_df["Value"].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
            st.dataframe(math_df, use_container_width=True)
    except Exception:
        pass

    # (Summary table removed; charts drive the overview)

    # Growth Summary panel
    with st.expander("Growth Summary (applied rates)", expanded=False):
        try:
            def cmgr_to_cagr(m):
                return ((1 + m / 100.0) ** 12 - 1) * 100.0
            growth_df = pd.DataFrame({
                "Metric": [
                    "Compute Y1 CMGR%","Compute Y2 CMGR%","Compute Y3 CMGR%","Compute Y4 CMGR%","Compute Y5 CMGR%",
                    "Storage Y1 CMGR%","Storage Y2 CMGR%","Storage Y3 CMGR%","Storage Y4 CMGR%","Storage Y5 CMGR%",
                ],
                "Monthly": [
                    st.session_state.get("og_y1_compute_total_cmgr", st.session_state.get("og_y1_compute_total_cmgr_fallback", 0.0)),
                    st.session_state.get("og_y2_compute_cmgr", 0.0),
                    st.session_state.get("og_y3_compute_cmgr", 1.0),
                    st.session_state.get("og_y4_compute_cmgr", 1.0),
                    st.session_state.get("og_y5_compute_cmgr", 1.0),
                    st.session_state.get("og_y1_storage_cmgr", st.session_state.get("og_y1_storage_cmgr_fallback", 0.0)),
                    st.session_state.get("og_y2_storage_cmgr", 0.0),
                    st.session_state.get("og_y3_storage_cmgr", 1.2),
                    st.session_state.get("og_y4_storage_cmgr", 1.2),
                    st.session_state.get("og_y5_storage_cmgr", 1.2),
                ]
            })
            growth_df["Annualized (CAGR %)"] = growth_df["Monthly"].apply(cmgr_to_cagr)
            st.dataframe(growth_df, use_container_width=True)
        except Exception:
            st.caption("Growth summary unavailable.")


    # (Removed duplicate bottom Low/High and Trend blocks)


