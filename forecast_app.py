import streamlit as st
import importlib
import os
import pandas as pd
from datetime import datetime
from browser_connection import execute_sql

# Try to get Snowflake connection - cache in session state to avoid re-auth
if 'snowflake_connection' not in st.session_state:
    try:
        # For Streamlit in Snowflake
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        st.session_state.snowflake_connection = session
        st.session_state.use_snowpark = True
    except:
        # For local development - use browser connection
        try:
            from browser_connection import set_connection, close_connection
            cursor, conn = set_connection()
            st.session_state.snowflake_connection = cursor
            st.session_state.browser_conn = conn
            st.session_state.use_snowpark = False
            st.success("Connected to Snowflake (Role: PUBLIC, Warehouse: SNOWADHOC)")
        except Exception as e:
            st.error(f"**Connection Failed:** {str(e)[:200]}...")
            st.error("Please check your Snowflake credentials and network connection.")
            st.stop()

# Use cached connection
if st.session_state.use_snowpark:
    session = st.session_state.snowflake_connection
    use_snowpark = True
else:
    cursor = st.session_state.snowflake_connection  
    use_snowpark = False

# Configure page with Snowflake branding
st.set_page_config(
    page_title="Forecast Planning Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    """Initialize session state variables for forecast app"""
    if 'forecast_initialized' not in st.session_state:
        st.session_state.forecast_initialized = True
        st.session_state.selected_customer = 'SIEMENS_AG'
        # Show only Overview by default for faster initial load
        st.session_state.selected_slides = ['Forecast Overview']
        
        # Forecast growth rates from organic growth
        st.session_state.forecast_year1_compute_growth = 0.0
        st.session_state.forecast_year1_storage_growth = 0.0
        st.session_state.forecast_year1_other_growth = 0.0
        st.session_state.forecast_year1_data_transfer_growth = 0.0
        st.session_state.forecast_year1_priority_support_growth = 0.0

def get_slide_session_defaults():
    """Get default session state values from all slide modules"""
    slide_defaults = {}
    slides_dir = "slides"
    
    if os.path.exists(slides_dir):
        for filename in os.listdir(slides_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]
                module_path = f"slides.{module_name}"
                
                try:
                    module = importlib.import_module(module_path)
                    # Check if module has session state defaults
                    if hasattr(module, 'get_session_defaults') and callable(getattr(module, 'get_session_defaults')):
                        slide_defaults.update(module.get_session_defaults())
                except ImportError:
                    continue
    
    return slide_defaults

def init_slide_session_state():
    """Initialize session state for all slide modules"""
    slide_defaults = get_slide_session_defaults()
    
    for key, default_value in slide_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def discover_slide_modules():
    """
    Automatically discover slide modules in the slides directory.
    Returns a dictionary with title -> module_path mapping.
    """
    slide_modules = {}
    slides_dir = "slides"
    
    # Get all Python files in the slides directory
    if os.path.exists(slides_dir):
        for filename in os.listdir(slides_dir):
            if filename.endswith('.py') and not filename.startswith('__') and not filename.startswith('_'):
                module_name = filename[:-3]  # Remove .py extension
                module_path = f"slides.{module_name}"
                
                try:
                    # Import the module to check if it has a render function
                    module = importlib.import_module(module_path)
                    if hasattr(module, 'render') and callable(getattr(module, 'render')):
                        # Convert filename to title case
                        title = module_name.replace('_', ' ').title()
                        slide_modules[title] = module_path
                except ImportError:
                    # Skip modules that can't be imported
                    continue
    
    return slide_modules

# Custom CSS for Snowflake branding
st.markdown("""
<style>
    /* Snowflake Brand Colors */
    :root {
        --snowflake-blue: #29B5E8;
        --mid-blue: #11567F;
        --midnight: #000000;
        --star-blue: #71D3DC;
        --valencia-orange: #FF9F36;
        --purple-moon: #7D44CF;
        --first-light: #D45B90;
        --windy-city: #8A999E;
    }
    
    /* Main title styling */
    .main .block-container h1 {
        color: var(--mid-blue);
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Subheader styling */
    .main .block-container h2 {
        color: var(--mid-blue);
        font-weight: 600;
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--snowflake-blue);
        padding-bottom: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--snowflake-blue);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background-color: var(--mid-blue);
        color: white;
    }
    
    /* Slide title styling */
    .slide-title {
        background: linear-gradient(90deg, var(--snowflake-blue), var(--star-blue));
        padding: 1rem;
        border-radius: 8px;
        margin: 2rem 0;
    }
    
    .slide-title h2 {
        color: white;
        margin: 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
init_session_state()
init_slide_session_state()

# Sidebar: Schema selection (replaces Salesforce ID flow)
st.sidebar.markdown("""
    <h3 style="color: #11567F;">Schema Selection</h3>
""", unsafe_allow_html=True)

# Load and cache available schemas
if 'available_finops_schemas' not in st.session_state:
    try:
        if use_snowpark:
            session.sql("USE DATABASE FINOPS_OUTPUTS").collect()
            schema_query = """
            SELECT SCHEMA_NAME 
            FROM INFORMATION_SCHEMA.SCHEMATA 
            WHERE CATALOG_NAME = 'FINOPS_OUTPUTS'
            ORDER BY SCHEMA_NAME
            """
            rows = session.sql(schema_query).collect()
            st.session_state.available_finops_schemas = [r[0] for r in rows]
        else:
            cursor.execute("USE DATABASE FINOPS_OUTPUTS")
            schema_query = """
            SELECT SCHEMA_NAME 
            FROM INFORMATION_SCHEMA.SCHEMATA 
            WHERE CATALOG_NAME = 'FINOPS_OUTPUTS'
            ORDER BY SCHEMA_NAME
            """
            df_s = execute_sql(cursor, schema_query)
            st.session_state.available_finops_schemas = df_s['SCHEMA_NAME'].tolist()
    except Exception as e:
        st.sidebar.error(f"Failed to list schemas: {str(e)[:80]}")
        st.session_state.available_finops_schemas = []

selected_schema = st.sidebar.selectbox(
    "FINOPS_OUTPUTS Schema:",
    options=[""] + st.session_state.available_finops_schemas,
    index=0,
    help="Choose the schema to analyze (e.g., CAPITAL_ONE_UE_20250606)",
)

validated_info = None
selected_customer_schema = None
if selected_schema:
    validated_info = {
        'salesforce_id': selected_schema,  # kept for compatibility
        'salesforce_name': selected_schema,
        'potential_schema': f'FINOPS_OUTPUTS.{selected_schema}',
        'status': 'VALID',
    }
    st.session_state.validated_customer_info = validated_info
    selected_customer_schema = validated_info['potential_schema']
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Schema Selected:**")
    st.sidebar.markdown(f"`{selected_customer_schema}`")

    # Pre-fetch defaults for Overview: use cases and optimizations (only after selection)
    try:
        import slides.use_case_forecaster as ucf_mod
        if not st.session_state.get("ucf_use_cases"):
            known_uc = ucf_mod._fetch_known_use_cases(session if use_snowpark else cursor, selected_customer_schema, 'snowpark' if use_snowpark else 'cursor')
            rows = []
            if known_uc is not None and len(known_uc) > 0:
                for _, r in known_uc.iterrows():
                    rows.append({
                        "use_case": str(r.get("USE_CASE", "")).strip(),
                        "stage": str(r.get("STAGE", "")).strip(),
                        "implementation_start": pd.to_datetime(r.get("IMPLEMENTATION_START", None), errors="coerce"),
                        "go_live": pd.to_datetime(r.get("GO_LIVE", None), errors="coerce"),
                        "implementation_months": 3,
                        "ramp_curve": "Linear Ramp",
                        "annualized_cost": float(r.get("ANNUALIZED_COST", 0.0) or 0.0),
                        # default splits from OG session if available
                        "split_compute_pct": float(st.session_state.get('og_default_split_compute_pct', 100.0)),
                        "split_storage_pct": float(st.session_state.get('og_default_split_storage_pct', 0.0)),
                        "split_data_transfer_pct": float(st.session_state.get('og_default_split_data_transfer_pct', 0.0)),
                        # default growth Y1 from OG
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
                    })
                if rows:
                    st.session_state.ucf_use_cases = rows
    except Exception:
        pass

    try:
        import slides.optimization_forecast as opt_mod
        if not st.session_state.get("opt_rows"):
            known_opt = opt_mod._fetch_known_optimizations(session if use_snowpark else cursor, selected_customer_schema, 'snowpark' if use_snowpark else 'cursor')
            rows = []
            if known_opt is not None and len(known_opt) > 0:
                for _, r in known_opt.iterrows():
                    name = f"{str(r.get('ROCK_CATEGORY','')).title()} - {str(r.get('ROCK','')).title()}"
                    for rng, col in [("LOW", "LOW_SAVINGS"), ("HIGH", "HIGH_SAVINGS")]:
                        amount = float(r.get(col, 0.0) or 0.0)
                        if amount == 0:
                            continue
                        rg = str(r.get("REVENUE_GROUP", "COMPUTE")).upper()
                        c, s, dt = 100.0, 0.0, 0.0
                        if rg.startswith("STORAGE"):
                            c, s, dt = 0.0, 100.0, 0.0
                        elif rg.startswith("DATA"):
                            c, s, dt = 0.0, 0.0, 100.0
                        rows.append({
                            "optimization": name,
                            "implementation_start": pd.to_datetime(datetime.today()),
                            "implementation_months": 3,
                            "ramp_curve": "Linear Ramp",
                            "range": rng,
                            "annualized_savings": float(amount),
                            "split_compute_pct": c,
                            "split_storage_pct": s,
                            "split_dt_pct": dt,
                        })
                if rows:
                    st.session_state.opt_rows = rows
    except Exception:
        pass
else:
    st.sidebar.info("Select a FINOPS_OUTPUTS schema to begin")

# Slide selection
st.sidebar.markdown("""
    <h3 style="color: #11567F;">Analysis Modules</h3>
""", unsafe_allow_html=True)

# Auto-discover slide modules
SLIDE_MODULES = discover_slide_modules()

if SLIDE_MODULES:
    selected_slide_names = st.sidebar.multiselect(
        "Select analysis modules:",
        list(SLIDE_MODULES.keys()),
        default=st.session_state.selected_slides if st.session_state.selected_slides else ['Organic Growth'],
        key="slide_multiselect"
    )
    st.session_state.selected_slides = selected_slide_names
else:
    st.sidebar.warning("No slide modules found")
    st.session_state.selected_slides = []

# Main title
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="color: #11567F; font-weight: 700;">Forecast Planning Tool</h1>
    <p style="font-size: 1.2rem; color: #6c757d;">Advanced consumption forecasting and growth analysis</p>
</div>
""", unsafe_allow_html=True)

# Display current selection
if validated_info:
    st.info(f"Schema: `{validated_info['potential_schema']}`")
else:
    st.warning("Please select a schema from the sidebar to proceed with forecast analysis.")
    st.markdown("""
    ### How to get started:
    1. Select a `FINOPS_OUTPUTS` schema in the sidebar
    2. Choose one or more analysis modules
    """)

# Render selected slides - only if customer is validated
if not selected_customer_schema:
    st.stop()
elif not st.session_state.selected_slides:
    st.warning("Please select at least one analysis module from the sidebar.")
else:
    for slide_name in st.session_state.selected_slides:
        if slide_name in SLIDE_MODULES:
            module_path = SLIDE_MODULES[slide_name]
            
            try:
                slide_module = importlib.import_module(module_path)
                
                # Add slide title with Snowflake styling
                st.markdown("---")
                st.markdown(f"""
                <div class="slide-title">
                    <h2>{slide_name}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Render the slide module with Salesforce information
                if use_snowpark:
                    slide_module.render(session, selected_customer_schema, validated_info, 'snowpark')
                else:
                    slide_module.render(cursor, selected_customer_schema, validated_info, 'cursor')
                
            except Exception as e:
                st.error(f"Error loading {slide_name}: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem;">
    <p>Built with Streamlit and Snowflake • Powered by Snowpark</p>
</div>
""", unsafe_allow_html=True)
