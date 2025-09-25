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


def _fetch_known_use_cases(connection, selected_customer_schema: str, connection_type: str) -> pd.DataFrame:
    """Fetch known use cases scoped to the customer's Salesforce IDs.

    Strategy: pull SF_IDs from `{schema}.SCOPED_ACCOUNTS`, then query A360 by those IDs.
    This avoids a large cross-database join and works with limited roles.
    """
    try:
        sf_sql = f"SELECT DISTINCT SF_ID FROM {selected_customer_schema}.SCOPED_ACCOUNTS WHERE SF_ID IS NOT NULL"
        if connection_type == "snowpark":
            sf_rows = connection.sql(sf_sql).to_pandas()
        else:
            from browser_connection import execute_sql
            sf_rows = execute_sql(connection, sf_sql)
        ids = [str(x) for x in sf_rows["SF_ID"].dropna().tolist()]
        if not ids:
            return pd.DataFrame()

        # Build IN clause safely (strings quoted)
        id_list = ",".join([f"'{i}'" for i in ids[:200]])  # cap to 200 IDs for safety
        uc_sql = f"""
        SELECT 
          DELIVERABLE_NAME               AS USE_CASE,
          DELIVERABLE_STAGE_NAME         AS STAGE,
          IMPLEMENTATION_START_DATE_C    AS IMPLEMENTATION_START,
          ACTUAL_GO_LIVE_DATE_C          AS GO_LIVE,
          ESTIMATED_ANNUAL_CREDIT_CONSUMPTION_C AS ANNUALIZED_COST
        FROM SALES.SALES_BI.A360_USE_CASES
        WHERE ACCOUNT_ID IN ({id_list})
          AND IMPLEMENTATION_START_DATE_C IS NOT NULL
          AND ACTUAL_GO_LIVE_DATE_C IS NOT NULL
        ORDER BY ACTUAL_GO_LIVE_DATE_C
        """
        if connection_type == "snowpark":
            return connection.sql(uc_sql).to_pandas()
        else:
            from browser_connection import execute_sql
            return execute_sql(connection, uc_sql)
    except Exception as e:
        try:
            st.warning(f"Use case fetch failed: {str(e)[:120]}")
        except Exception:
            pass
        return pd.DataFrame()


def get_session_defaults() -> Dict[str, object]:
    """Return default session state values for Use Case Forecaster slide."""
    today = datetime.today()
    first_month = _first_of_month(today)

    # Start with no customer-specific rows; users add rows per customer context
    default_use_cases: List[Dict[str, object]] = []

    return {
        "ucf_default_compute_pct": 92.62,
        "ucf_default_storage_pct": 7.38,
        "ucf_default_data_transfer_pct": 0.0,
        "ucf_planning_horizon_months": 60,
        "ucf_use_cases": default_use_cases,
    }


def _ensure_session_defaults() -> None:
    defaults = get_session_defaults()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _editor_dataframe() -> pd.DataFrame:
    """Render the editable use case table and return the DataFrame."""
    stage_options = [
        "1 - Discovery",
        "2 - Scoping",
        "3 - Technical / Business Validation",
        "4 - Use Case Won / Migration Plan",
        "5 - Implementation In Progress",
    ]
    ramp_options = [
        "Slowest Ramp",
        "Slow Ramp",
        "Linear Ramp",
        "Fast Ramp",
        "Fastest Ramp",
        "Manual",
    ]

    df = pd.DataFrame(st.session_state["ucf_use_cases"]) if st.session_state["ucf_use_cases"] else pd.DataFrame()
    # Pre-clean: drop blank/"nan" names to avoid showing many empty rows
    if not df.empty and "use_case" in df.columns:
        name_series = df["use_case"].astype(str).str.strip()
        df = df[(name_series != "") & (name_series.str.lower() != "nan")]
        # Persist clean set so next rerun is also clean
        st.session_state.ucf_use_cases = df.to_dict(orient="records")
    # Ensure date-typed columns for Data Editor compatibility
    if not df.empty:
        for date_col in ["implementation_start", "go_live"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Provide column configuration and types
    col_config = {
        "use_case": st.column_config.TextColumn("USE CASE", required=True),
        "stage": st.column_config.SelectboxColumn("STAGE", options=stage_options, required=True),
        "implementation_start": st.column_config.DateColumn("IMPLEMENTATION START DATE", format="YYYY-MM-DD", required=True),
        "go_live": st.column_config.DateColumn("GO LIVE DATE", format="YYYY-MM-DD", required=True),
        "implementation_months": st.column_config.NumberColumn("TOTAL IMPLEMENTATION MONTHS", min_value=0, max_value=60, step=1),
        "ramp_curve": st.column_config.SelectboxColumn("RAMP UP CURVE", options=ramp_options, required=True),
        # Remove format to avoid literal display like ",.2f" in some Streamlit versions
        "annualized_cost": st.column_config.NumberColumn("ANNUALIZED COST ($)", min_value=0.0),
        "split_compute_pct": st.column_config.NumberColumn("COMPUTE %", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
        "split_storage_pct": st.column_config.NumberColumn("STORAGE %", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
        "split_data_transfer_pct": st.column_config.NumberColumn("DATA TRANSFER %", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
        "compute_grow_y1_pct": st.column_config.NumberColumn("Compute MoM Growth Y1 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "compute_grow_y2_pct": st.column_config.NumberColumn("Compute MoM Growth Y2 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "compute_grow_y3_pct": st.column_config.NumberColumn("Compute MoM Growth Y3 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "compute_grow_y4_pct": st.column_config.NumberColumn("Compute MoM Growth Y4 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "compute_grow_y5_pct": st.column_config.NumberColumn("Compute MoM Growth Y5 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "storage_grow_y1_pct": st.column_config.NumberColumn("Storage MoM Growth Y1 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "storage_grow_y2_pct": st.column_config.NumberColumn("Storage MoM Growth Y2 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "storage_grow_y3_pct": st.column_config.NumberColumn("Storage MoM Growth Y3 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "storage_grow_y4_pct": st.column_config.NumberColumn("Storage MoM Growth Y4 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "storage_grow_y5_pct": st.column_config.NumberColumn("Storage MoM Growth Y5 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "dt_grow_y1_pct": st.column_config.NumberColumn("Data Transfer MoM Growth Y1 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "dt_grow_y2_pct": st.column_config.NumberColumn("Data Transfer MoM Growth Y2 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "dt_grow_y3_pct": st.column_config.NumberColumn("Data Transfer MoM Growth Y3 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "dt_grow_y4_pct": st.column_config.NumberColumn("Data Transfer MoM Growth Y4 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
        "dt_grow_y5_pct": st.column_config.NumberColumn("Data Transfer MoM Growth Y5 %", step=0.01, min_value=-100.0, max_value=100.0, format="%.2f"),
    }

    st.markdown("#### Default new use case % split")
    c1, c2, c3 = st.columns(3)
    with c1:
        # Pull defaults from Organic Growth latest-month if available: Compute = Compute + Other
        default_compute = float(
            st.session_state.get("og_default_split_compute_pct", st.session_state.ucf_default_compute_pct)
        )
        st.session_state.ucf_default_compute_pct = st.number_input(
            "Compute %", value=default_compute, min_value=0.0, max_value=100.0, step=0.01, key="ucf_default_compute_pct_widget"
        )
    with c2:
        default_storage = float(
            st.session_state.get("og_default_split_storage_pct", st.session_state.ucf_default_storage_pct)
        )
        st.session_state.ucf_default_storage_pct = st.number_input(
            "Storage %", value=default_storage, min_value=0.0, max_value=100.0, step=0.01, key="ucf_default_storage_pct_widget"
        )
    with c3:
        default_dt = float(
            st.session_state.get("og_default_split_data_transfer_pct", st.session_state.ucf_default_data_transfer_pct)
        )
        st.session_state.ucf_default_data_transfer_pct = st.number_input(
            "Data Transfer %", value=default_dt, min_value=0.0, max_value=100.0, step=0.01, key="ucf_default_dt_pct_widget"
        )

    st.caption("New rows should sum to 100% across Compute, Storage, Data Transfer.")

    # Friendly input form for adding a row
    with st.expander("Add Use Case"):
        c1, c2 = st.columns(2)
        with c1:
            new_use_case = st.text_input("Use case name", value="", key="ucf_form_use_case")
            new_stage = st.selectbox("Stage", options=stage_options, index=2, key="ucf_form_stage")
            new_impl_start = st.date_input("Implementation start date", key="ucf_form_impl_start")
            new_go_live = st.date_input("Go live date", key="ucf_form_go_live")
            new_impl_months = st.number_input("Total implementation months", min_value=0, max_value=60, step=1, value=3, key="ucf_form_impl_months")
            new_ramp = st.selectbox("Ramp up curve", options=ramp_options, index=2, key="ucf_form_ramp_curve")
        with c2:
            new_annual_cost = st.number_input("Annualized cost ($)", min_value=0.0, value=0.0, step=1000.0, format="%.2f", key="ucf_form_annual_cost")
            # Default splits come from Organic Growth-derived session values, editable here
            split_c = st.number_input("Compute %", min_value=0.0, max_value=100.0, value=float(st.session_state.ucf_default_compute_pct), step=0.01, key="ucf_form_compute_pct")
            split_s = st.number_input("Storage %", min_value=0.0, max_value=100.0, value=float(st.session_state.ucf_default_storage_pct), step=0.01, key="ucf_form_storage_pct")
            split_dt = st.number_input("Data Transfer %", min_value=0.0, max_value=100.0, value=float(st.session_state.ucf_default_data_transfer_pct), step=0.01, key="ucf_form_dt_pct")
            st.caption("Splits should sum to 100%.")
            st.markdown("**Y1–Y5 monthly growth (%):**")
            y1c = st.number_input("Compute Y1", value=float(st.session_state.get("og_y1_compute_total_cmgr", st.session_state.get("og_y1_compute_total_cmgr_fallback", 0.0))), step=0.01, key="ucf_form_compute_y1")
            y2c = st.number_input("Compute Y2", value=0.0, step=0.01, key="ucf_form_compute_y2")
            y3c = st.number_input("Compute Y3", value=0.0, step=0.01, key="ucf_form_compute_y3")
            y4c = st.number_input("Compute Y4", value=0.0, step=0.01, key="ucf_form_compute_y4")
            y5c = st.number_input("Compute Y5", value=0.0, step=0.01, key="ucf_form_compute_y5")
            y1s = st.number_input("Storage Y1", value=float(st.session_state.get("og_y1_storage_cmgr", st.session_state.get("og_y1_storage_cmgr_fallback", 0.0))), step=0.01, key="ucf_form_storage_y1")
            y2s = st.number_input("Storage Y2", value=0.0, step=0.01, key="ucf_form_storage_y2")
            y3s = st.number_input("Storage Y3", value=0.0, step=0.01, key="ucf_form_storage_y3")
            y4s = st.number_input("Storage Y4", value=0.0, step=0.01, key="ucf_form_storage_y4")
            y5s = st.number_input("Storage Y5", value=0.0, step=0.01, key="ucf_form_storage_y5")

        add_ok = st.button("Add use case", type="primary")
        if add_ok:
            if new_use_case and (split_c + split_s + split_dt) > 0:
                new_row = {
                    "use_case": new_use_case,
                    "stage": new_stage,
                    "implementation_start": pd.to_datetime(new_impl_start),
                    "go_live": pd.to_datetime(new_go_live),
                    "implementation_months": int(new_impl_months),
                    "ramp_curve": new_ramp,
                    "annualized_cost": float(new_annual_cost),
                    "split_compute_pct": float(split_c),
                    "split_storage_pct": float(split_s),
                    "split_data_transfer_pct": float(split_dt),
                    "compute_grow_y1_pct": float(y1c),
                    "compute_grow_y2_pct": float(y2c),
                    "compute_grow_y3_pct": float(y3c),
                    "compute_grow_y4_pct": float(y4c),
                    "compute_grow_y5_pct": float(y5c),
                    "storage_grow_y1_pct": float(y1s),
                    "storage_grow_y2_pct": float(y2s),
                    "storage_grow_y3_pct": float(y3s),
                    "storage_grow_y4_pct": float(y4s),
                    "storage_grow_y5_pct": float(y5s),
                }
                st.session_state.ucf_use_cases = st.session_state.ucf_use_cases + [new_row]
                df = pd.DataFrame(st.session_state.ucf_use_cases)
                for date_col in ["implementation_start", "go_live"]:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                st.success("Use case added.")
                st.rerun()
            else:
                st.error("Please provide a name and ensure splits total > 0%.")

    # Quick-fill helpers
    with st.expander("Quick Fill Growth Rates"):
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            fill_col = st.selectbox("Column", [
                "compute_grow_y1_pct","compute_grow_y2_pct","compute_grow_y3_pct","compute_grow_y4_pct","compute_grow_y5_pct",
                "storage_grow_y1_pct","storage_grow_y2_pct","storage_grow_y3_pct","storage_grow_y4_pct","storage_grow_y5_pct",
                "dt_grow_y1_pct","dt_grow_y2_pct","dt_grow_y3_pct","dt_grow_y4_pct","dt_grow_y5_pct",
            ], index=0)
        with qc2:
            fill_value = st.number_input("Value (%)", value=0.0, step=0.01)
        with qc3:
            if st.button("Apply to all rows") and not df.empty and fill_col in df.columns:
                df[fill_col] = float(fill_value)
                st.session_state.ucf_use_cases = df.to_dict(orient="records")
                st.success(f"Applied {fill_value:.2f}% to '{fill_col}' for all rows.")

    # Editor for fine-tuning rows
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        key="ucf_editor",
        column_config=col_config,
        use_container_width=True,
    )

    st.info("Tip: Double-click a cell to edit. Values are monthly % growth. Select a block of cells and paste to fill multiple rows.")

    # Clear all control
    cclear1, cclear2 = st.columns([1, 6])
    with cclear1:
        if st.button("Clear all use cases", type="secondary"):
            st.session_state.ucf_use_cases = []
            edited_df = pd.DataFrame()
            st.info("All rows cleared.")

    # Normalize splits for any new rows that have 0s
    def _apply_default_splits(row: pd.Series) -> pd.Series:
        if (
            float(row.get("split_compute_pct", 0.0)) == 0.0
            and float(row.get("split_storage_pct", 0.0)) == 0.0
            and float(row.get("split_data_transfer_pct", 0.0)) == 0.0
        ):
            row["split_compute_pct"] = st.session_state.ucf_default_compute_pct
            row["split_storage_pct"] = st.session_state.ucf_default_storage_pct
            row["split_data_transfer_pct"] = st.session_state.ucf_default_data_transfer_pct
        return row

    edited_df = edited_df.apply(_apply_default_splits, axis=1)

    # Drop empty/placeholder rows (no use case name)
    try:
        cleaned_df = edited_df.dropna(subset=["use_case"]).copy()
        # Remove blanks and literal "nan" strings produced by text conversion
        name_series = cleaned_df["use_case"].astype(str).str.strip()
        cleaned_df = cleaned_df[(name_series != "") & (name_series.str.lower() != "nan")]
    except Exception:
        cleaned_df = edited_df

    # Persist back to session
    st.session_state.ucf_use_cases = cleaned_df.to_dict(orient="records")
    return cleaned_df


def _compute_growth_factor(month_index: int, growth_per_year_monthly: List[float]) -> float:
    """
    Compute compounded growth factor after a number of months with piecewise
    monthly rates per year. Example: [y1, y2, y3, y4, y5].
    Returns a multiplicative factor (e.g., 1.10 means +10%).
    """
    factor = 1.0
    remaining = month_index
    year = 0
    while remaining > 0 and year < len(growth_per_year_monthly):
        months_in_this_year = min(12, remaining)
        rate = growth_per_year_monthly[year] / 100.0
        # Compound monthly for the months in this segment
        factor *= (1.0 + rate) ** months_in_this_year
        remaining -= months_in_this_year
        year += 1
    return factor


def _linear_ramp_factor(months_since_go_live: int, ramp_months: int) -> float:
    if ramp_months <= 0:
        return 1.0
    if months_since_go_live <= 0:
        return 0.0
    if months_since_go_live >= ramp_months:
        return 1.0
    return months_since_go_live / float(ramp_months)


def _curve_ramp_factor(curve: str, months_since_go_live: int, ramp_months: int) -> float:
    """Different ramp shapes inspired by Excel options; Manual treated as Linear for now."""
    if ramp_months <= 0:
        return 1.0
    m = max(0, min(months_since_go_live, ramp_months))
    x = m / float(ramp_months)  # 0..1
    if curve == "Slowest Ramp":
        # Ease-in cubic
        return x ** 3
    if curve == "Slow Ramp":
        # Ease-in quadratic
        return x ** 2
    if curve == "Fast Ramp":
        # Ease-out quadratic
        return 1 - (1 - x) ** 2
    if curve == "Fastest Ramp":
        # Ease-out cubic
        return 1 - (1 - x) ** 3
    # Linear or Manual default
    return x


def _build_month_range(use_cases_df: pd.DataFrame, horizon_months: int) -> List[datetime]:
    today = datetime.today()
    start = _first_of_month(today)
    # If any implementation start dates are earlier, begin from that
    try:
        impl_starts = pd.to_datetime(use_cases_df["implementation_start"]).dropna()
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


def _compute_forecast(use_cases_df: pd.DataFrame, horizon_months: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    months = _build_month_range(use_cases_df, horizon_months)
    timeline_df = pd.DataFrame({"MONTH": months})

    # Initialize totals per type
    totals = {
        "COMPUTE": [0.0 for _ in months],
        "STORAGE": [0.0 for _ in months],
        "DATA_TRANSFER": [0.0 for _ in months],
    }

    stage_totals: Dict[str, float] = {}

    for _, row in use_cases_df.iterrows():
        try:
            impl_start = _first_of_month(pd.to_datetime(row["implementation_start"]).to_pydatetime())
            go_live = _first_of_month(pd.to_datetime(row["go_live"]).to_pydatetime())
        except Exception:
            # Skip rows with invalid dates
            continue

        ramp_months = int(row.get("implementation_months", 0) or 0)
        annualized_cost = float(row.get("annualized_cost", 0.0) or 0.0)
        # Monthly base comes from annualized cost divided by 12 (matches Excel)
        monthly_run_rate = annualized_cost / 12.0

        # Splits
        compute_split = float(row.get("split_compute_pct", 0.0) or 0.0) / 100.0
        storage_split = float(row.get("split_storage_pct", 0.0) or 0.0) / 100.0
        dt_split = float(row.get("split_data_transfer_pct", 0.0) or 0.0) / 100.0

        # Growth arrays per year (monthly %)
        compute_growth = [
            float(row.get("compute_grow_y1_pct", 0.0) or 0.0),
            float(row.get("compute_grow_y2_pct", 0.0) or 0.0),
            float(row.get("compute_grow_y3_pct", 0.0) or 0.0),
            float(row.get("compute_grow_y4_pct", 0.0) or 0.0),
            float(row.get("compute_grow_y5_pct", 0.0) or 0.0),
        ]
        storage_growth = [
            float(row.get("storage_grow_y1_pct", 0.0) or 0.0),
            float(row.get("storage_grow_y2_pct", 0.0) or 0.0),
            float(row.get("storage_grow_y3_pct", 0.0) or 0.0),
            float(row.get("storage_grow_y4_pct", 0.0) or 0.0),
            float(row.get("storage_grow_y5_pct", 0.0) or 0.0),
        ]
        dt_growth = [
            float(row.get("dt_grow_y1_pct", 0.0) or 0.0),
            float(row.get("dt_grow_y2_pct", 0.0) or 0.0),
            float(row.get("dt_grow_y3_pct", 0.0) or 0.0),
            float(row.get("dt_grow_y4_pct", 0.0) or 0.0),
            float(row.get("dt_grow_y5_pct", 0.0) or 0.0),
        ]

        for idx, m in enumerate(months):
            if m < go_live:
                continue

            months_since_go_live = _months_between(go_live, m)
            ramp_curve = str(row.get("ramp_curve", "Linear Ramp"))
            ramp_factor = _curve_ramp_factor(ramp_curve, months_since_go_live, ramp_months)

            # Compute compounded growth factor since go-live for each type
            growth_factor_compute = _compute_growth_factor(months_since_go_live, compute_growth)
            growth_factor_storage = _compute_growth_factor(months_since_go_live, storage_growth)
            growth_factor_dt = _compute_growth_factor(months_since_go_live, dt_growth)

            base = monthly_run_rate * ramp_factor
            totals["COMPUTE"][idx] += base * compute_split * growth_factor_compute
            totals["STORAGE"][idx] += base * storage_split * growth_factor_storage
            totals["DATA_TRANSFER"][idx] += base * dt_split * growth_factor_dt

        stage = str(row.get("stage", "Unknown"))
        stage_totals.setdefault(stage, 0.0)
        stage_totals[stage] += annualized_cost

    result_df = timeline_df.copy()
    result_df["COMPUTE"] = totals["COMPUTE"]
    result_df["STORAGE"] = totals["STORAGE"]
    result_df["DATA_TRANSFER"] = totals["DATA_TRANSFER"]
    result_df["TOTAL"] = result_df[["COMPUTE", "STORAGE", "DATA_TRANSFER"]].sum(axis=1)

    stage_df = pd.DataFrame({"STAGE": list(stage_totals.keys()), "AMOUNT": list(stage_totals.values())})
    return result_df, stage_df


def _compute_use_case_yearly_totals(use_cases_df: pd.DataFrame, horizon_months: int) -> Dict[str, List[float]]:
    """Return per-use-case totals for each of 5 years based on our forecast assumptions."""
    months = _build_month_range(use_cases_df, horizon_months)
    per_uc_monthly: Dict[str, List[float]] = {}

    for _, row in use_cases_df.iterrows():
        # Skip rows with missing/NaN names
        raw_name = row.get("use_case", "")
        if pd.isna(raw_name):
            continue
        use_case_name = str(raw_name).strip()
        if use_case_name == "" or use_case_name.lower() == "nan":
            continue
        per_uc_monthly.setdefault(use_case_name, [0.0 for _ in range(len(months))])
        # Parse go_live; if invalid/NaN, skip row
        try:
            go_live_val = pd.to_datetime(row["go_live"], errors="coerce")
        except Exception:
            go_live_val = pd.NaT
        if pd.isna(go_live_val):
            continue
        go_live = _first_of_month(go_live_val.to_pydatetime())

        ramp_months = int(row.get("implementation_months", 0) or 0)
        monthly_run_rate = float(row.get("annualized_cost", 0.0) or 0.0) / 12.0

        compute_split = float(row.get("split_compute_pct", 0.0) or 0.0) / 100.0
        storage_split = float(row.get("split_storage_pct", 0.0) or 0.0) / 100.0
        dt_split = float(row.get("split_data_transfer_pct", 0.0) or 0.0) / 100.0

        compute_growth = [
            float(row.get("compute_grow_y1_pct", 0.0) or 0.0),
            float(row.get("compute_grow_y2_pct", 0.0) or 0.0),
            float(row.get("compute_grow_y3_pct", 0.0) or 0.0),
            float(row.get("compute_grow_y4_pct", 0.0) or 0.0),
            float(row.get("compute_grow_y5_pct", 0.0) or 0.0),
        ]
        storage_growth = [
            float(row.get("storage_grow_y1_pct", 0.0) or 0.0),
            float(row.get("storage_grow_y2_pct", 0.0) or 0.0),
            float(row.get("storage_grow_y3_pct", 0.0) or 0.0),
            float(row.get("storage_grow_y4_pct", 0.0) or 0.0),
            float(row.get("storage_grow_y5_pct", 0.0) or 0.0),
        ]
        dt_growth = [
            float(row.get("dt_grow_y1_pct", 0.0) or 0.0),
            float(row.get("dt_grow_y2_pct", 0.0) or 0.0),
            float(row.get("dt_grow_y3_pct", 0.0) or 0.0),
            float(row.get("dt_grow_y4_pct", 0.0) or 0.0),
            float(row.get("dt_grow_y5_pct", 0.0) or 0.0),
        ]

        for idx, m in enumerate(months):
            if m < go_live:
                continue
            months_since_go_live = _months_between(go_live, m)
            ramp_curve = str(row.get("ramp_curve", "Linear Ramp"))
            ramp_factor = _curve_ramp_factor(ramp_curve, months_since_go_live, ramp_months)

            base = monthly_run_rate * ramp_factor
            value = 0.0
            value += base * compute_split * _compute_growth_factor(months_since_go_live, compute_growth)
            value += base * storage_split * _compute_growth_factor(months_since_go_live, storage_growth)
            value += base * dt_split * _compute_growth_factor(months_since_go_live, dt_growth)
            per_uc_monthly[use_case_name][idx] += value

    # Aggregate into 5 yearly totals
    per_uc_yearly: Dict[str, List[float]] = {}
    for uc, arr in per_uc_monthly.items():
        totals = []
        for y in range(5):
            start = y * 12
            end = start + 12
            totals.append(sum(arr[start:end]))
        per_uc_yearly[uc] = totals

    return per_uc_yearly


def _render_charts(monthly_df: pd.DataFrame, stage_df: pd.DataFrame) -> None:
    # Estimated Dollars by Year (bar)
    yearly = monthly_df.copy()
    yearly["YEAR"] = yearly["MONTH"].dt.year
    yearly_sum = yearly.groupby("YEAR")["TOTAL"].sum().reset_index()
    fig_year = go.Figure(
        data=[go.Bar(x=yearly_sum["YEAR"], y=yearly_sum["TOTAL"], marker_color="#29B5E8")]
    )
    fig_year.update_layout(title="Estimated Dollars by Year", xaxis_title="Year", yaxis_title="Dollars", height=280)

    # Placeholder; replaced later by per-use-case Year-1 bar in render()
    fig_impact = go.Figure()

    # Estimated Amount by Type (donut)
    type_totals = [monthly_df["COMPUTE"].sum(), monthly_df["STORAGE"].sum(), monthly_df["DATA_TRANSFER"].sum()]
    fig_type = go.Figure(data=[go.Pie(labels=["Compute", "Storage", "Data Transfer"], values=type_totals, hole=0.6)])
    fig_type.update_layout(title="Estimated Amt by Type", height=280)

    # Estimated Amount by Stage (donut)
    if not stage_df.empty:
        fig_stage = go.Figure(data=[go.Pie(labels=stage_df["STAGE"], values=stage_df["AMOUNT"], hole=0.6)])
        fig_stage.update_layout(title="Estimated Amount by Stage", height=280)
    else:
        fig_stage = go.Figure()

    # Render three charts (year summary, by type, by stage). Impact by use case is shown below.
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(fig_year, use_container_width=True, key="ucf_fig_year")
    with c2:
        st.plotly_chart(fig_type, use_container_width=True, key="ucf_fig_type")
    with c3:
        st.plotly_chart(fig_stage, use_container_width=True, key="ucf_fig_stage")

    # Forecast Trend: last 12 months actuals (gray) + next 60 months baseline-with-growth + new use cases (blue)
    try:
        og_hist = st.session_state.get("og_monthly_df", pd.DataFrame()).copy()
        if not og_hist.empty:
            og_hist = og_hist.sort_values("MONTH")
            hist_tail = og_hist.tail(12).copy()
            hist_tail.rename(columns={"TOTAL": "BILL"}, inplace=True)
            last_hist_month = hist_tail["MONTH"].max()

            # Build forecast months starting from next month after latest history
            def _add_month(d: datetime, n: int) -> datetime:
                y = d.year + (d.month - 1 + n) // 12
                m = (d.month - 1 + n) % 12 + 1
                return datetime(y, m, 1)

            forecast_months = [_add_month(last_hist_month, i) for i in range(1, 61)]

            # Baseline-with-growth per month starting from latest month total
            latest_total = float(hist_tail.loc[hist_tail["MONTH"] == last_hist_month, "BILL"].iloc[0])
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
            baseline_series: List[float] = []
            for i, _m in enumerate(forecast_months):
                year_idx = min(i // 12, 4)
                if i >= 0:  # apply growth for each forward month
                    if i > 0:
                        comp_val *= 1 + og_cmgr_compute[year_idx] / 100.0
                        stor_val *= 1 + og_cmgr_storage[year_idx] / 100.0
                baseline_series.append(comp_val + stor_val + dt_val)

            # Map new use cases monthly totals onto the same forecast months
            nc_map = monthly_df.set_index("MONTH")["TOTAL"].to_dict() if not monthly_df.empty else {}
            new_cases_series = [float(nc_map.get(m, 0.0)) for m in forecast_months]

            forecast_amt = [baseline_series[i] + new_cases_series[i] for i in range(len(forecast_months))]

            trend_df = pd.DataFrame({
                "MONTH": list(hist_tail["MONTH"]) + forecast_months,
                "BILL": list(hist_tail["BILL"]) + [None] * len(forecast_months),
                "FORECAST AMT": [None] * len(hist_tail) + forecast_amt,
            })

            fig_trend = go.Figure()
            fig_trend.add_bar(x=trend_df["MONTH"], y=trend_df["BILL"], name="BILL", marker_color="#888888")
            fig_trend.add_bar(x=trend_df["MONTH"], y=trend_df["FORECAST AMT"], name="FORECAST AMT", marker_color="#29B5E8")
            fig_trend.update_layout(title="Forecast Trend", barmode="group", height=320)
            st.plotly_chart(fig_trend, use_container_width=True, key="ucf_fig_trend")
    except Exception:
        pass


def render(connection, selected_customer_schema, validated_info, connection_type="snowpark") -> None:
    """
    Render the Use Case Forecaster slide, aligned to the Excel structure:
    - Editable table of use cases with splits and growth
    - Computation of monthly totals by type with ramp and growth
    - Summary charts similar to Excel
    """
    _ensure_session_defaults()

    st.markdown("*Plan and scale use cases. Configure timing, ramp, splits, and growth. The monthly forecast updates below.*")

    # Fixed planning horizon (5 years)
    st.session_state.ucf_planning_horizon_months = 60
    st.caption("Planning horizon: 60 months (fixed)")

    # Auto-load known use cases once if none present
    if not st.session_state.ucf_use_cases:
        with st.spinner("Fetching use cases, this may take some time..."):
            known_df = _fetch_known_use_cases(connection, selected_customer_schema, connection_type)
        if not known_df.empty:
            rows = []
            for _, r in known_df.iterrows():
                row = {
                    "use_case": str(r.get("USE_CASE", "")).strip(),
                    "stage": str(r.get("STAGE", "")).strip(),
                    "implementation_start": pd.to_datetime(r.get("IMPLEMENTATION_START", None), errors="coerce"),
                    "go_live": pd.to_datetime(r.get("GO_LIVE", None), errors="coerce"),
                    "implementation_months": 3,
                    "ramp_curve": "Linear Ramp",
                    "annualized_cost": float(r.get("ANNUALIZED_COST", 0.0) or 0.0),
                    "split_compute_pct": float(st.session_state.ucf_default_compute_pct),
                    "split_storage_pct": float(st.session_state.ucf_default_storage_pct),
                    "split_data_transfer_pct": float(st.session_state.ucf_default_data_transfer_pct),
                    "compute_grow_y1_pct": float(st.session_state.get("og_y1_compute_total_cmgr", st.session_state.get("og_y1_compute_total_cmgr_fallback", 0.0))),
                    "compute_grow_y2_pct": 0.0,
                    "compute_grow_y3_pct": 0.0,
                    "compute_grow_y4_pct": 0.0,
                    "compute_grow_y5_pct": 0.0,
                    "storage_grow_y1_pct": float(st.session_state.get("og_y1_storage_cmgr", st.session_state.get("og_y1_storage_cmgr_fallback", 0.0))),
                    "storage_grow_y2_pct": 0.0,
                    "storage_grow_y3_pct": 0.0,
                    "storage_grow_y4_pct": 0.0,
                    "storage_grow_y5_pct": 0.0,
                }
                if row["use_case"]:
                    rows.append(row)
            if rows:
                st.session_state.ucf_use_cases = rows
                st.rerun()

    # Editable table
    use_cases_df = _editor_dataframe()

    # Auto-populate Y1 growth defaults from Organic Growth when present
    y1_compute_default = st.session_state.get(
        "og_y1_compute_total_cmgr",
        st.session_state.get("og_y1_compute_total_cmgr_fallback", 0.0),
    )
    y1_storage_default = st.session_state.get(
        "og_y1_storage_cmgr",
        st.session_state.get("og_y1_storage_cmgr_fallback", 0.0),
    )
    y1_data_transfer_default = st.session_state.get(
        "og_y1_data_transfer_cmgr",
        st.session_state.get("og_y1_data_transfer_cmgr_fallback", 0.0),
    )

    st.caption(
        f"Y1 defaults (from Organic Growth): Compute Total {y1_compute_default:.2f}% • Storage {y1_storage_default:.2f}% • Data Transfer {y1_data_transfer_default:.2f}%"
    )

    # Compute forecast
    monthly_df, stage_df = _compute_forecast(use_cases_df, int(st.session_state.ucf_planning_horizon_months))

    # Store monthly for other pages; render charts on this page
    st.session_state['ucf_monthly_df'] = monthly_df.copy()
    try:
        token = f"{len(monthly_df)}_{float(monthly_df['TOTAL'].sum()):.2f}"
        if st.session_state.get('ucf_sync_token') != token:
            st.session_state['ucf_sync_token'] = token
            st.rerun()
    except Exception:
        pass
    # Charts
    _render_charts(monthly_df, stage_df)

    # Per-use-case Year selector and bar chart
    per_uc_yearly = _compute_use_case_yearly_totals(use_cases_df, int(st.session_state.ucf_planning_horizon_months))
    if per_uc_yearly:
        st.markdown("### Use Case Impact by Year")
        year_choice = st.selectbox("Year", options=["1", "2", "3", "4", "5", "Total (1-5)"], index=0)
        is_total = year_choice.startswith("Total")
        year_idx = int(year_choice) - 1 if not is_total else None
        labels = []
        values = []
        for uc, totals in per_uc_yearly.items():
            val = sum(totals) if is_total else (totals[year_idx] if year_idx < len(totals) else 0.0)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            labels.append(uc)
            values.append(val)
        # Sort descending
        pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
        labels = [p[0] for p in pairs]
        values = [p[1] for p in pairs]

        fig_uc = go.Figure(data=[go.Bar(x=values, y=labels, orientation='h', marker_color="#29B5E8", text=[f"${v:,.0f}" for v in values], textposition='outside')])
        fig_uc.update_layout(height=500, xaxis_title="Dollars", yaxis_title="Use Case")
        st.plotly_chart(fig_uc, use_container_width=True, key="ucf_fig_uc_impact")

        # Table of Years 1–5 totals per use case
        # Build table with the same filtered labels order
        ordered = {uc: per_uc_yearly[uc] for uc in labels}
        table_df = pd.DataFrame({
            "Use Case": list(ordered.keys()),
            "Year 1": [v[0] for v in ordered.values()],
            "Year 2": [v[1] for v in ordered.values()],
            "Year 3": [v[2] for v in ordered.values()],
            "Year 4": [v[3] for v in ordered.values()],
            "Year 5": [v[4] for v in ordered.values()],
            "Total (1-5)": [sum(v) for v in ordered.values()],
        })
        st.dataframe(table_df, use_container_width=True)

    # Monthly detail table
    with st.expander("View Monthly Forecast Table"):
        display_df = monthly_df.copy()
        display_df["MONTH"] = display_df["MONTH"].dt.strftime("%b %Y")
        for col in ["COMPUTE", "STORAGE", "DATA_TRANSFER", "TOTAL"]:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_df, use_container_width=True)


