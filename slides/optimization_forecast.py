import math
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime


def _first_of_month(date_obj: datetime) -> datetime:
    return datetime(date_obj.year, date_obj.month, 1)


def _months_between(start: datetime, end: datetime) -> int:
    """Inclusive difference in months where start is first-of-month."""
    return (end.year - start.year) * 12 + (end.month - start.month)


def get_session_defaults() -> Dict[str, object]:
    today = datetime.today()
    return {
        "opt_rows": [],
        "opt_planning_horizon_months": 60,
        "opt_5y_selected": 0.0,
    }


def _ensure_session_defaults() -> None:
    for k, v in get_session_defaults().items():
        if k not in st.session_state:
            st.session_state[k] = v


def _fetch_known_optimizations(connection, selected_customer_schema: str, connection_type: str) -> pd.DataFrame:
    """Fetch optimizations from Rocks Summary, aggregating to LOW/HIGH annualized savings."""
    sql = f"""
    SELECT REVENUE_GROUP, ROCK_CATEGORY, ROCK,
           TOTAL_LOW_ANNUALIZED_DOLLAR_SAVINGS  AS LOW_SAVINGS,
           TOTAL_HIGH_ANNUALIZED_DOLLAR_SAVINGS AS HIGH_SAVINGS
    FROM {selected_customer_schema}.SCOPED_ACCOUNTS_ROCKS_SUMMARY
    WHERE (COALESCE(TOTAL_LOW_ANNUALIZED_DOLLAR_SAVINGS,0) <> 0
        OR COALESCE(TOTAL_HIGH_ANNUALIZED_DOLLAR_SAVINGS,0) <> 0)
    """
    try:
        if connection_type == "snowpark":
            return connection.sql(sql).to_pandas()
        else:
            from browser_connection import execute_sql
            return execute_sql(connection, sql)
    except Exception as e:
        try:
            st.warning(f"Optimization fetch failed: {str(e)[:120]}")
        except Exception:
            pass
        return pd.DataFrame()


def _curve_ramp_factor(curve: str, months_since_start: int, ramp_months: int) -> float:
    """Excel-style ramp using exponents from Forecast Lookups.

    Uses x = (t - s + 1) / (e - s + 1) with exponent mapping:
      Slowest=4.0, Slow=2.0, Linear=1.0, Fast=0.75, Fastest=0.5
    Manual returns 0 (allowing manual overrides upstream if desired).
    """
    if ramp_months <= 0:
        return 1.0
    # Clamp to [0, ramp_months]
    m = max(0, min(months_since_start, ramp_months))
    # Excel uses 1-based index inside the window
    x = (m + 1) / float(ramp_months + 1)
    exp_map = {
        "Slowest Ramp": 4.0,
        "Slow Ramp": 2.0,
        "Linear Ramp": 1.0,
        "Fast Ramp": 0.75,
        "Fastest Ramp": 0.5,
    }
    if curve == "Manual":
        return 0.0
    exponent = exp_map.get(curve, 1.0)
    return x ** exponent


def _build_month_range(rows_df: pd.DataFrame, horizon_months: int) -> List[datetime]:
    start = _first_of_month(datetime.today())
    try:
        impl_starts = pd.to_datetime(rows_df["implementation_start"]).dropna()
        if not impl_starts.empty:
            start = min(start, _first_of_month(impl_starts.min().to_pydatetime()))
    except Exception:
        pass
    months: List[datetime] = []
    for i in range(horizon_months):
        year = start.year + (start.month - 1 + i) // 12
        month = (start.month - 1 + i) % 12 + 1
        months.append(datetime(year, month, 1))
    return months


def _editor_dataframe() -> pd.DataFrame:
    ramp_options = [
        "Slowest Ramp",
        "Slow Ramp",
        "Linear Ramp",
        "Fast Ramp",
        "Fastest Ramp",
        "Manual",
    ]
    range_options = ["LOW", "HIGH"]

    df = pd.DataFrame(st.session_state["opt_rows"]) if st.session_state["opt_rows"] else pd.DataFrame()
    # Default precise splits based on optimization type when splits missing
    if not df.empty:
        def _default_splits(name: str) -> tuple:
            n = (name or "").lower()
            # Map common optimization names to spend category
            if any(k in n for k in ["storage", "autocluster", "unused storage", "inactive storage"]):
                return 0.0, 100.0, 0.0
            if any(k in n for k in ["transfer", "egress", "data transfer"]):
                return 0.0, 0.0, 100.0
            return 100.0, 0.0, 0.0  # default to compute
        for i, r in df.iterrows():
            if pd.isna(r.get("split_compute_pct")) and pd.isna(r.get("split_storage_pct")) and pd.isna(r.get("split_data_transfer_pct")):
                c, s, dt = _default_splits(str(r.get("optimization", "")))
                df.loc[i, "split_compute_pct"] = c
                df.loc[i, "split_storage_pct"] = s
                df.loc[i, "split_data_transfer_pct"] = dt
    if not df.empty:
        for date_col in ["implementation_start"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    col_config = {
        "optimization": st.column_config.TextColumn("OPTIMIZATION", required=True),
        "implementation_start": st.column_config.DateColumn("IMPLEMENTATION START DATE", format="YYYY-MM-DD", required=True),
        "implementation_months": st.column_config.NumberColumn("TOTAL IMPLEMENTATION MONTHS", min_value=0, max_value=60, step=1),
        "ramp_curve": st.column_config.SelectboxColumn("RAMP UP CURVE", options=ramp_options, required=True, default="Linear Ramp"),
        "range_type": st.column_config.SelectboxColumn("RANGE", options=range_options, required=True),
        "annualized_savings": st.column_config.NumberColumn("ANNUALIZED SAVINGS ($)", min_value=0.0),
        "split_compute_pct": st.column_config.NumberColumn("COMPUTE %", min_value=0.0, max_value=100.0, step=0.01, format="%.2f", default=100.0),
        "split_storage_pct": st.column_config.NumberColumn("STORAGE %", min_value=0.0, max_value=100.0, step=0.01, format="%.2f", default=0.0),
        "split_data_transfer_pct": st.column_config.NumberColumn("DATA TRANSFER %", min_value=0.0, max_value=100.0, step=0.01, format="%.2f", default=0.0),
        "savings_y1_pct": st.column_config.NumberColumn("Savings MoM Change Y1 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "savings_y2_pct": st.column_config.NumberColumn("Savings MoM Change Y2 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "savings_y3_pct": st.column_config.NumberColumn("Savings MoM Change Y3 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "savings_y4_pct": st.column_config.NumberColumn("Savings MoM Change Y4 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "savings_y5_pct": st.column_config.NumberColumn("Savings MoM Change Y5 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
    }

    st.markdown("#### Add Optimization")
    c1, c2 = st.columns(2)
    with c1:
        opt_name = st.text_input("Optimization name", value="")
        impl_start = st.date_input("Implementation start date", key="opt_form_impl_start")
        impl_months = st.number_input("Total implementation months", min_value=0, max_value=60, step=1, value=3)
        ramp_curve = st.selectbox("Ramp up curve", options=ramp_options, index=2)
        range_type = st.selectbox("Range", options=range_options, index=0)
    with c2:
        annualized_savings = st.number_input("Annualized savings ($)", min_value=0.0, value=0.0, step=1000.0, format="%.2f", key="opt_form_annual")
        split_c = st.number_input("Compute %", min_value=0.0, max_value=100.0, value=100.0, step=0.01, key="opt_form_compute_pct")
        split_s = st.number_input("Storage %", min_value=0.0, max_value=100.0, value=0.0, step=0.01, key="opt_form_storage_pct")
        split_dt = st.number_input("Data Transfer %", min_value=0.0, max_value=100.0, value=0.0, step=0.01, key="opt_form_dt_pct")
        st.markdown("**Optional MoM change in savings (Y1–Y5)**")
        y1 = st.number_input("Y1 %", value=0.0, step=0.01, key="opt_form_y1")
        y2 = st.number_input("Y2 %", value=0.0, step=0.01, key="opt_form_y2")
        y3 = st.number_input("Y3 %", value=0.0, step=0.01, key="opt_form_y3")
        y4 = st.number_input("Y4 %", value=0.0, step=0.01, key="opt_form_y4")
        y5 = st.number_input("Y5 %", value=0.0, step=0.01, key="opt_form_y5")

    if st.button("Add optimization", type="primary"):
        if opt_name:
            row = {
                "optimization": opt_name,
                "implementation_start": pd.to_datetime(impl_start),
                "implementation_months": int(impl_months),
                "ramp_curve": ramp_curve,
                "range_type": range_type,
                "annualized_savings": float(annualized_savings),
                "split_compute_pct": float(split_c),
                "split_storage_pct": float(split_s),
                "split_data_transfer_pct": float(split_dt),
                "savings_y1_pct": float(y1),
                "savings_y2_pct": float(y2),
                "savings_y3_pct": float(y3),
                "savings_y4_pct": float(y4),
                "savings_y5_pct": float(y5),
            }
            st.session_state.opt_rows = st.session_state.opt_rows + [row]
            st.success("Optimization added.")
            st.rerun()
        else:
            st.error("Please provide an optimization name.")

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        key="opt_editor",
        column_config=col_config,
        use_container_width=True,
    )

    st.session_state.opt_rows = edited_df.to_dict(orient="records")
    return edited_df


def _compute_forecast(opt_df: pd.DataFrame, horizon_months: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    months = _build_month_range(opt_df, horizon_months)
    timeline_df = pd.DataFrame({"MONTH": months})
    totals = {"COMPUTE": [0.0 for _ in months], "STORAGE": [0.0 for _ in months], "DATA_TRANSFER": [0.0 for _ in months]}

    for _, row in opt_df.iterrows():
        try:
            start = _first_of_month(pd.to_datetime(row["implementation_start"]).to_pydatetime())
        except Exception:
            continue
        ramp_months = int(row.get("implementation_months", 0) or 0)
        base_monthly = -abs(float(row.get("annualized_savings", 0.0) or 0.0)) / 12.0
        compute_split = float(row.get("split_compute_pct", 0.0) or 0.0) / 100.0
        storage_split = float(row.get("split_storage_pct", 0.0) or 0.0) / 100.0
        dt_split = float(row.get("split_data_transfer_pct", 0.0) or 0.0) / 100.0
        savings_rates = [
            float(row.get("savings_y1_pct", 0.0) or 0.0),
            float(row.get("savings_y2_pct", 0.0) or 0.0),
            float(row.get("savings_y3_pct", 0.0) or 0.0),
            float(row.get("savings_y4_pct", 0.0) or 0.0),
            float(row.get("savings_y5_pct", 0.0) or 0.0),
        ]

        for idx, m in enumerate(months):
            if m < start:
                continue
            months_since_start = _months_between(start, m)
            ramp_factor = _curve_ramp_factor(str(row.get("ramp_curve", "Linear Ramp")), months_since_start, ramp_months)
            year_idx = min(months_since_start // 12, 4)
            rate = savings_rates[year_idx] / 100.0
            change_factor = (1.0 + rate) ** (months_since_start if months_since_start > 0 else 0)
            value = base_monthly * ramp_factor * change_factor
            totals["COMPUTE"][idx] += value * compute_split
            totals["STORAGE"][idx] += value * storage_split
            totals["DATA_TRANSFER"][idx] += value * dt_split

    result_df = timeline_df.copy()
    result_df["COMPUTE"] = totals["COMPUTE"]
    result_df["STORAGE"] = totals["STORAGE"]
    result_df["DATA_TRANSFER"] = totals["DATA_TRANSFER"]
    result_df["TOTAL"] = result_df[["COMPUTE", "STORAGE", "DATA_TRANSFER"]].sum(axis=1)
    summary = {"total_5y": -sum(result_df["TOTAL"]) }  # positive magnitude of savings
    return result_df, summary


def render(connection, selected_customer_schema, validated_info, connection_type="snowpark") -> None:
    _ensure_session_defaults()
    st.markdown("*Plan and track optimizations. Savings are applied monthly with ramp and optional Y1–Y5 changes.*")

    # Auto-load known optimizations once if none present
    if not st.session_state.opt_rows:
        df_known = _fetch_known_optimizations(connection, selected_customer_schema, connection_type)
        rows = []
        for _, r in df_known.iterrows():
            name = str(r.get("OPTIMIZATION", "")).strip()
            low = float(r.get("LOW_SAVINGS", 0.0) or 0.0)
            high = float(r.get("HIGH_SAVINGS", 0.0) or 0.0)
            if not name:
                continue
            row = {
                "optimization": name,
                "implementation_start": _first_of_month(datetime.today()),
                "implementation_months": 3,
                "ramp_curve": "Linear Ramp",
                "range_type": "LOW",
                "annualized_savings": low if low > 0 else high,
                "split_compute_pct": 100.0,
                "split_storage_pct": 0.0,
                "split_data_transfer_pct": 0.0,
                "savings_y1_pct": 0.0,
                "savings_y2_pct": 0.0,
                "savings_y3_pct": 0.0,
                "savings_y4_pct": 0.0,
                "savings_y5_pct": 0.0,
            }
            rows.append(row)
        if rows:
            st.session_state.opt_rows = rows

    # Editor
    opt_df = _editor_dataframe()

    # Forecast
    monthly_df, summary = _compute_forecast(opt_df, int(st.session_state.opt_planning_horizon_months))

    # Charts
    yearly = monthly_df.copy()
    yearly["YEAR"] = yearly["MONTH"].dt.year
    yearly_sum = yearly.groupby("YEAR")["TOTAL"].sum().reset_index()
    fig_year = go.Figure(data=[go.Bar(x=yearly_sum["YEAR"], y=yearly_sum["TOTAL"], marker_color="#11567F")])
    fig_year.update_layout(title="Estimated Savings by Year", xaxis_title="Year", yaxis_title="Savings ($)", height=280)
    type_totals = [monthly_df["COMPUTE"].sum(), monthly_df["STORAGE"].sum(), monthly_df["DATA_TRANSFER"].sum()]
    fig_type = go.Figure(data=[go.Pie(labels=["Compute", "Storage", "Data Transfer"], values=[abs(x) for x in type_totals], hole=0.6)])
    fig_type.update_layout(title="Savings by Type", height=280)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_year, use_container_width=True)
    with c2:
        st.plotly_chart(fig_type, use_container_width=True)

    # Expose for Overview/Waterfall and for live updates
    st.session_state["opt_5y_selected"] = float(abs(summary.get("total_5y", 0.0)))
    st.session_state["opt_monthly_df"] = monthly_df.copy()
    try:
        token = f"{len(monthly_df)}_{float(monthly_df['TOTAL'].sum()):.2f}"
        if st.session_state.get('opt_sync_token') != token:
            st.session_state['opt_sync_token'] = token
            st.rerun()
    except Exception:
        pass
    st.caption(f"Optimizations (5y total): ${st.session_state['opt_5y_selected']:,.2f}")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from typing import Dict, List


def _first_of_month(date_obj: datetime) -> datetime:
    return datetime(date_obj.year, date_obj.month, 1)


def _months_between(start: datetime, end: datetime) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def get_session_defaults() -> Dict[str, object]:
    return {
        "opt_default_compute_pct": 100.0,
        "opt_default_storage_pct": 0.0,
        "opt_default_dt_pct": 0.0,
        "opt_planning_horizon_months": 60,
        "opt_rows": [],
    }


def _ensure_defaults():
    for k, v in get_session_defaults().items():
        if k not in st.session_state:
            st.session_state[k] = v


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
    return x


def _build_months(horizon: int) -> List[datetime]:
    start = _first_of_month(datetime.today())
    months = []
    for i in range(horizon):
        y = start.year + (start.month - 1 + i) // 12
        m = (start.month - 1 + i) % 12 + 1
        months.append(datetime(y, m, 1))
    return months


def _fetch_known_optimizations(connection, selected_customer_schema: str, connection_type: str) -> pd.DataFrame:
    """Fetch known optimizations using Rocks Summary as a seed list.
    Creates LOW/HIGH variants per rock with annualized savings.
    """
    rocks_summary = f"{selected_customer_schema}.SCOPED_ACCOUNTS_ROCKS_SUMMARY"
    sql = f"""
    SELECT REVENUE_GROUP, ROCK_CATEGORY, ROCK,
           TOTAL_LOW_ANNUALIZED_DOLLAR_SAVINGS  AS LOW_SAVINGS,
           TOTAL_HIGH_ANNUALIZED_DOLLAR_SAVINGS AS HIGH_SAVINGS
    FROM {rocks_summary}
    WHERE (COALESCE(TOTAL_LOW_ANNUALIZED_DOLLAR_SAVINGS,0) <> 0
        OR COALESCE(TOTAL_HIGH_ANNUALIZED_DOLLAR_SAVINGS,0) <> 0)
    """
    try:
        if connection_type == "snowpark":
            return connection.sql(sql).to_pandas()
        else:
            from browser_connection import execute_sql
            return execute_sql(connection, sql)
    except Exception:
        return pd.DataFrame()


def _editor(connection, selected_customer_schema: str, connection_type: str) -> pd.DataFrame:
    df = pd.DataFrame(st.session_state["opt_rows"]) if st.session_state["opt_rows"] else pd.DataFrame()
    if not df.empty:
        for col in ["implementation_start"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    st.markdown("#### Default optimization % split")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.opt_default_compute_pct = st.number_input(
            "Compute %", value=float(st.session_state.opt_default_compute_pct), min_value=0.0, max_value=100.0, step=0.01, key="opt_def_c"
        )
    with c2:
        st.session_state.opt_default_storage_pct = st.number_input(
            "Storage %", value=float(st.session_state.opt_default_storage_pct), min_value=0.0, max_value=100.0, step=0.01, key="opt_def_s"
        )
    with c3:
        st.session_state.opt_default_dt_pct = st.number_input(
            "Data Transfer %", value=float(st.session_state.opt_default_dt_pct), min_value=0.0, max_value=100.0, step=0.01, key="opt_def_dt"
        )
    st.caption("Splits should sum to 100% across Compute, Storage, Data Transfer.")

    with st.expander("Add Optimization"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Optimization", value="")
            impl_start = st.date_input("Implementation start date", key="opt_editor_impl_start")
            ramp_months = st.number_input("Total implementation months", min_value=0, max_value=60, step=1, value=3)
            ramp_curve = st.selectbox("Ramp up curve", ["Slowest Ramp", "Slow Ramp", "Linear Ramp", "Fast Ramp", "Fastest Ramp", "Manual"], index=2)
        with c2:
            range_type = st.selectbox("Range", ["LOW", "HIGH"], index=0)
            annual_savings = st.number_input("Annualized savings ($)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
            split_c = st.number_input("Compute %", min_value=0.0, max_value=100.0, value=float(st.session_state.opt_default_compute_pct), step=0.01, key="opt_form_c")
            split_s = st.number_input("Storage %", min_value=0.0, max_value=100.0, value=float(st.session_state.opt_default_storage_pct), step=0.01, key="opt_form_s")
            split_dt = st.number_input("Data Transfer %", min_value=0.0, max_value=100.0, value=float(st.session_state.opt_default_dt_pct), step=0.01, key="opt_form_dt")
        if st.button("Add optimization", type="primary"):
            if name and (split_c + split_s + split_dt) > 0:
                row = {
                    "optimization": name,
                    "implementation_start": pd.to_datetime(impl_start),
                    "implementation_months": int(ramp_months),
                    "ramp_curve": ramp_curve,
                    "range": range_type,
                    "annualized_savings": float(annual_savings),
                    "split_compute_pct": float(split_c),
                    "split_storage_pct": float(split_s),
                    "split_dt_pct": float(split_dt),
                }
                st.session_state.opt_rows = st.session_state.opt_rows + [row]
                st.success("Optimization added.")
                st.rerun()
            else:
                st.error("Provide a name and non-zero splits.")

    # Removed manual fetch expander; optimizations auto-load on first render

    col_config = {
        "optimization": st.column_config.TextColumn("OPTIMIZATION", required=True),
        "implementation_start": st.column_config.DateColumn("IMPLEMENTATION START", format="YYYY-MM-DD", required=True),
        "implementation_months": st.column_config.NumberColumn("TOTAL IMPLEMENTATION MONTHS", min_value=0, max_value=60, step=1),
        "ramp_curve": st.column_config.SelectboxColumn("RAMP UP CURVE", options=["Slowest Ramp","Slow Ramp","Linear Ramp","Fast Ramp","Fastest Ramp","Manual"], required=True),
        "range": st.column_config.SelectboxColumn("RANGE", options=["LOW","HIGH"], required=True),
        "annualized_savings": st.column_config.NumberColumn("ANNUALIZED SAVINGS ($)", min_value=0.0),
        "split_compute_pct": st.column_config.NumberColumn("COMPUTE %", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
        "split_storage_pct": st.column_config.NumberColumn("STORAGE %", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
        "split_dt_pct": st.column_config.NumberColumn("DATA TRANSFER %", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
    }

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        key="opt_editor",
        column_config=col_config,
        use_container_width=True,
    )

    st.session_state.opt_rows = edited_df.to_dict(orient="records")
    return edited_df


def _compute_monthly(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    months = _build_months(horizon)
    values = [0.0 for _ in months]
    for _, r in df.iterrows():
        try:
            start = _first_of_month(pd.to_datetime(r.get("implementation_start")).to_pydatetime())
        except Exception:
            continue
        ramp_months = int(r.get("implementation_months", 0) or 0)
        curve = str(r.get("ramp_curve", "Linear Ramp"))
        annual = float(r.get("annualized_savings", 0.0) or 0.0)
        base = annual / 12.0
        split_c = float(r.get("split_compute_pct", 0.0) or 0.0) / 100.0
        split_s = float(r.get("split_storage_pct", 0.0) or 0.0) / 100.0
        split_dt = float(r.get("split_dt_pct", 0.0) or 0.0) / 100.0
        for i, m in enumerate(months):
            if m < start:
                continue
            months_since = _months_between(start, m)
            ramp = _curve_ramp_factor(curve, months_since, ramp_months)
            savings = base * ramp * (split_c + split_s + split_dt)
            # savings decrease spend -> treat as negative
            values[i] += -savings
    out = pd.DataFrame({"MONTH": months, "SAVINGS": values})
    return out


def render(connection, selected_customer_schema, validated_info, connection_type="snowpark") -> None:
    _ensure_defaults()
    st.markdown("*Plan and stage optimizations. Configure timing, ramp, splits, and range. Savings are applied as reductions.*")

    # Auto-load known optimizations once if none exist
    if not st.session_state.opt_rows:
        with st.spinner("Fetching optimizations, this may take some time..."):
            known = _fetch_known_optimizations(connection, selected_customer_schema, connection_type)
        rows = []
        for _, r in known.iterrows():
            opt_name = f"{str(r.get('ROCK_CATEGORY','')).title()} - {str(r.get('ROCK','')).title()}" if 'ROCK' in r else str(r.get('OPTIMIZATION','')).strip()
            if not opt_name:
                continue
            # create LOW and HIGH when available
            pairs = [("LOW", "LOW_SAVINGS"), ("HIGH", "HIGH_SAVINGS")] if "LOW_SAVINGS" in r and "HIGH_SAVINGS" in r else [("LOW", "LOW_SAVINGS")]
            for rng, col in pairs:
                amount = float(r.get(col, 0.0) or 0.0)
                if amount == 0:
                    continue
                rows.append({
                    "optimization": opt_name,
                    "implementation_start": pd.to_datetime(datetime.today()),
                    "implementation_months": 3,
                    "ramp_curve": "Linear Ramp",
                    "range": rng if 'range' in ['range'] else rng,
                    "annualized_savings": float(amount),
                    "split_compute_pct": 100.0,
                    "split_storage_pct": 0.0,
                    "split_dt_pct": 0.0,
                })
        if rows:
            st.session_state.opt_rows = rows

    # Editor
    df = _editor(connection, selected_customer_schema, connection_type)

    # Forecast
    monthly = _compute_monthly(df, int(st.session_state.opt_planning_horizon_months))

    # Charts
    yearly = monthly.copy()
    yearly["YEAR"] = yearly["MONTH"].dt.year
    yearly_sum = yearly.groupby("YEAR")["SAVINGS"].sum().reset_index()
    fig_year = go.Figure(data=[go.Bar(x=yearly_sum["YEAR"], y=abs(yearly_sum["SAVINGS"]), marker_color="#0B8F55")])
    fig_year.update_layout(title="Estimated Optimizations by Year (abs $)", xaxis_title="Year", yaxis_title="Dollars", height=280)
    st.plotly_chart(fig_year, use_container_width=True)

    # Monthly table
    with st.expander("View Monthly Optimization Table"):
        disp = monthly.copy()
        disp["MONTH"] = disp["MONTH"].dt.strftime("%b %Y")
        disp["SAVINGS"] = disp["SAVINGS"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(disp, use_container_width=True)

    # Expose for Overview and trigger rerun if changed
    st.session_state["opt_monthly_df"] = monthly.copy()
    try:
        token = f"{len(monthly)}_{float(monthly['SAVINGS'].sum()):.2f}"
        if st.session_state.get('opt_sync_token') != token:
            st.session_state['opt_sync_token'] = token
            st.rerun()
    except Exception:
        pass


