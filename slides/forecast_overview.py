import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime


def get_session_defaults():
    return {}


def _build_overview_components():
    # Organic Growth baseline from latest month
    latest_total = float(st.session_state.get("og_latest_month_total", 0.0))
    comp_pct = float(st.session_state.get("og_default_split_compute_pct", 0.0)) / 100.0
    stor_pct = float(st.session_state.get("og_default_split_storage_pct", 0.0)) / 100.0
    dt_pct = float(st.session_state.get("og_default_split_data_transfer_pct", 0.0)) / 100.0
    comp_val = latest_total * comp_pct
    stor_val = latest_total * stor_pct
    dt_val = latest_total * dt_pct

    og_cmgr_compute = [
        float(st.session_state.get("og_y1_compute_total_cmgr", st.session_state.get("og_y1_compute_total_cmgr_fallback", 0.0))),
        st.session_state.get("og_y2_compute_cmgr", 0.0),
        st.session_state.get("og_y3_compute_cmgr", 1.0),
        st.session_state.get("og_y4_compute_cmgr", 1.0),
        st.session_state.get("og_y5_compute_cmgr", 1.0),
    ]
    og_cmgr_storage = [
        float(st.session_state.get("og_y1_storage_cmgr", st.session_state.get("og_y1_storage_cmgr_fallback", 0.0))),
        st.session_state.get("og_y2_storage_cmgr", 0.0),
        st.session_state.get("og_y3_storage_cmgr", 1.2),
        st.session_state.get("og_y4_storage_cmgr", 1.2),
        st.session_state.get("og_y5_storage_cmgr", 1.2),
    ]

    baseline_growth_months = []
    for m in range(60):
        year_idx = min(m // 12, 4)
        if m > 0:
            comp_val *= 1 + og_cmgr_compute[year_idx] / 100.0
            stor_val *= 1 + og_cmgr_storage[year_idx] / 100.0
        baseline_growth_months.append(comp_val + stor_val + dt_val)

    baseline_with_growth_5y = sum(baseline_growth_months)
    baseline_steady_5y = latest_total * 60.0
    organic_growth_5y = baseline_with_growth_5y - baseline_steady_5y

    # New Use Cases monthly from Use Case Forecaster
    ucf_monthly = st.session_state.get("ucf_monthly_df", pd.DataFrame())
    new_use_cases_5y = float(ucf_monthly["TOTAL"].sum()) if not ucf_monthly.empty else 0.0

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
    optimizations_5y = 0.0
    try:
        if not opt_monthly.empty:
            col = "SAVINGS" if "SAVINGS" in opt_monthly.columns else ("TOTAL" if "TOTAL" in opt_monthly.columns else None)
            if col:
                optimizations_5y = float(abs(opt_monthly[col].sum()))
        else:
            rows = st.session_state.get("opt_rows", [])
            optimizations_5y = _compute_opt_5y_from_rows(rows) if rows else float(st.session_state.get("opt_5y_selected", 0.0))
    except Exception:
        rows = st.session_state.get("opt_rows", [])
        optimizations_5y = _compute_opt_5y_from_rows(rows) if rows else float(st.session_state.get("opt_5y_selected", 0.0))

    discount_5y = float(st.session_state.get("overview_discount_5y", 0.0))

    contract_value_5y = baseline_steady_5y + organic_growth_5y + new_use_cases_5y - optimizations_5y - discount_5y

    return baseline_steady_5y, organic_growth_5y, new_use_cases_5y, optimizations_5y, discount_5y, contract_value_5y


def render(connection, selected_customer_schema, validated_info, connection_type="snowpark"):
    st.markdown("*High-level summary of forecast components across modules.*")

    base, org_uplift, new_cases, opts, disc, total = _build_overview_components()

    st.markdown("### Forecast Overview (5-year)")
    # Show quick diagnostics for sync
    try:
        uc_rows = 0 if ucf_monthly is None or ucf_monthly.empty else len(ucf_monthly)
        opt_rows = 0 if opt_monthly is None or opt_monthly.empty else len(opt_monthly)
        st.caption(f"Sync: use cases rows={uc_rows}, optimizations rows={opt_rows}")
    except Exception:
        pass
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Discount (5y)", value=float(disc), step=1000.0, key="overview_discount_5y")
    with col2:
        st.caption("Optimizations auto-sync from Optimization Forecast; Use Cases auto-sync from Use Case Forecaster.")

    # Refresh computed totals after possible discount change
    base, org_uplift, new_cases, opts, disc, total = _build_overview_components()

    wf = go.Figure(go.Waterfall(
        name="",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        x=["Baseline", "Organic Growth", "New Use Cases", "Optimizations", "Discount", "Contract Value"],
        textposition="outside",
        y=[base, org_uplift, new_cases, -abs(opts), -abs(disc), total],
        connector={"line": {"color": "#11567F"}},
    ))
    wf.update_layout(height=320)
    st.plotly_chart(wf, use_container_width=True)


