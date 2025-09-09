import streamlit as st
import importlib
import os
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
        st.session_state.selected_slides = ['Organic Growth']
        
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

# Sidebar with Salesforce customer input
st.sidebar.markdown("""
    <h3 style="color: #11567F;">Customer Selection</h3>
""", unsafe_allow_html=True)

# Initialize Salesforce input session state
if 'salesforce_account_id' not in st.session_state:
    st.session_state.salesforce_account_id = ''
if 'customer_name' not in st.session_state:
    st.session_state.customer_name = ''
if 'validated_customer_info' not in st.session_state:
    st.session_state.validated_customer_info = None

# Salesforce Account ID input
salesforce_id = st.sidebar.text_input(
    "Salesforce Account ID:",
    value=st.session_state.salesforce_account_id,
    placeholder="e.g., 0018Z00002ABC123",
    help="Enter the 18-character Salesforce Account ID",
    key="salesforce_id_input"
)

# Auto-derive customer name from validation (no manual input needed)
customer_name = st.session_state.get('customer_name', '')

# Update session state
st.session_state.salesforce_account_id = salesforce_id

# Validate customer information
validated_info = None
selected_customer_schema = None

if salesforce_id and len(salesforce_id) >= 15:
    # Only validate if SF ID changed and user finished typing (avoid validation on every keystroke)
    if ('last_validated_sf_id' not in st.session_state or 
        st.session_state.last_validated_sf_id != salesforce_id) and len(salesforce_id) == 18:
        with st.spinner("Loading customer information..."):
            try:
                # Step 1: Query Fivetran Salesforce to get real customer name from SFID
                customer_name_from_sf = None
                try:
                    if use_snowpark:
                        sf_query = f"""
                        SELECT NAME, ID
                        FROM FIVETRAN.SALESFORCE.ACCOUNT 
                        WHERE ID = '{salesforce_id}'
                        LIMIT 1
                        """
                        sf_result = session.sql(sf_query).collect()
                        if sf_result and len(sf_result) > 0:
                            customer_name_from_sf = sf_result[0][0]  # NAME column
                    else:
                        sf_query = f"""
                        SELECT NAME, ID
                        FROM FIVETRAN.SALESFORCE.ACCOUNT 
                        WHERE ID = '{salesforce_id}'
                        LIMIT 1
                        """
                        sf_df = execute_sql(cursor, sf_query)
                        if not sf_df.empty:
                            customer_name_from_sf = sf_df['NAME'].iloc[0]
                            
                except Exception as sf_error:
                    st.warning(f"Salesforce lookup failed: {str(sf_error)}")
                
                if not customer_name_from_sf:
                    st.error("Cannot find this Salesforce ID in Fivetran Salesforce data")
                    st.session_state.validated_customer_info = None
                    st.stop()
                
                # Step 2: Clean customer name for schema matching
                # Convert "Chewy, Inc." -> "CHEWY", "Capital One Bank" -> "CAPITAL"
                clean_name = customer_name_from_sf.upper()
                clean_name = clean_name.replace(',', '').replace('.', '').replace(' INC', '').replace(' LLC', '').replace(' CORP', '').replace(' CORPORATION', '')
                clean_name = clean_name.split()[0]  # Take first word: "CAPITAL ONE" -> "CAPITAL"
                
                # Step 3: Search FINOPS_OUTPUTS for schemas containing this customer name
                discovered_schema = None
                try:
                    if use_snowpark:
                        session.sql("USE DATABASE FINOPS_OUTPUTS").collect()
                        
                        # Use INFORMATION_SCHEMA instead of SHOW SCHEMAS (more reliable)
                        schema_query = """
                        SELECT SCHEMA_NAME 
                        FROM INFORMATION_SCHEMA.SCHEMATA 
                        WHERE CATALOG_NAME = 'FINOPS_OUTPUTS'
                        ORDER BY SCHEMA_NAME
                        """
                        schema_result = session.sql(schema_query).collect()
                        
                        # First pass: Look for schemas that START with customer name (preferred)
                        for row in schema_result:
                            schema_name = row[0]  # SCHEMA_NAME column
                            if schema_name.upper().startswith(clean_name + '_'):
                                discovered_schema = schema_name
                                break
                                
                        # Second pass: If no exact start match, look for customer name anywhere (fallback)
                        if not discovered_schema:
                            for row in schema_result:
                                schema_name = row[0]  # SCHEMA_NAME column
                                # Avoid false positives like ARCH_CAPITAL when looking for CAPITAL
                                if (clean_name in schema_name.upper() and 
                                    not schema_name.upper().startswith('ARCH_') and
                                    not schema_name.upper().startswith('MARCH_')):
                                    discovered_schema = schema_name
                                    break
                    else:
                        cursor.execute("USE DATABASE FINOPS_OUTPUTS")
                        
                        # Use INFORMATION_SCHEMA instead of SHOW SCHEMAS (more reliable)
                        schema_query = """
                        SELECT SCHEMA_NAME 
                        FROM INFORMATION_SCHEMA.SCHEMATA 
                        WHERE CATALOG_NAME = 'FINOPS_OUTPUTS'
                        ORDER BY SCHEMA_NAME
                        """
                        schema_df = execute_sql(cursor, schema_query)
                        
                        # First pass: Look for schemas that START with customer name (preferred)
                        for _, row in schema_df.iterrows():
                            schema_name = row['SCHEMA_NAME']
                            if schema_name.upper().startswith(clean_name + '_'):
                                discovered_schema = schema_name
                                break
                                
                        # Second pass: If no exact start match, look for customer name anywhere (fallback)
                        if not discovered_schema:
                            for _, row in schema_df.iterrows():
                                schema_name = row['SCHEMA_NAME']
                                # Avoid false positives like ARCH_CAPITAL when looking for CAPITAL
                                if (clean_name in schema_name.upper() and 
                                    not schema_name.upper().startswith('ARCH_') and
                                    not schema_name.upper().startswith('MARCH_')):
                                    discovered_schema = schema_name
                                    break
                                
                except Exception as schema_error:
                    st.error(f"FINOPS_OUTPUTS schema search failed: {str(schema_error)}")
                    st.session_state.validated_customer_info = None
                    st.stop()
                
                if not discovered_schema:
                    st.error(f"Cannot find customer '{customer_name_from_sf}' in FINOPS_OUTPUTS database")
                    st.info("This customer may not have L3 analysis data available")
                    st.session_state.validated_customer_info = None
                    st.stop()
                    
                # Step 4: Success - found both customer in SF and schema in FINOPS_OUTPUTS
                customer_name_derived = customer_name_from_sf
                
                # Store discovered customer name
                st.session_state.customer_name = customer_name_derived
                
                # Create validation info
                validated_info = {
                    'salesforce_id': salesforce_id,
                    'salesforce_name': customer_name_derived,
                    'potential_schema': f'FINOPS_OUTPUTS.{discovered_schema}',
                    'data_points': 1,  # Schema discovery successful
                    'status': 'VALID'
                }
                st.session_state.validated_customer_info = validated_info
                st.session_state.last_validated_sf_id = salesforce_id  # Prevent re-validation
                
                if validated_info['status'] == 'VALID':
                    st.sidebar.success(f"{validated_info['salesforce_name']}")
                else:
                    st.sidebar.warning("Account found but no recent data")
                    
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)[:50]}...")

# Use validated information or show validation requirement
if st.session_state.validated_customer_info and st.session_state.validated_customer_info['status'] == 'VALID':
    validated_info = st.session_state.validated_customer_info
    selected_customer_schema = validated_info['potential_schema']
    
    # Display validated customer info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Customer Validated:**")
    st.sidebar.markdown(f"**Name:** {validated_info['salesforce_name']}")
    st.sidebar.markdown(f"**ID:** {validated_info['salesforce_id']}")
    st.sidebar.markdown(f"**Schema:** `{validated_info['potential_schema']}`")
    
elif salesforce_id or customer_name:
    st.sidebar.warning("Please click 'Validate Customer' to proceed")
    selected_customer_schema = None
else:
    st.sidebar.info("Enter Salesforce Account ID and Customer Name to begin")

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

# Display current selections
if validated_info:
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Customer:** {validated_info['salesforce_name']}")
    with col2:
        st.info(f"**Salesforce ID:** {validated_info['salesforce_id']}")
        
else:
    st.warning("Please validate a customer first using the sidebar to proceed with forecast analysis.")
    st.markdown("""
    ### How to get started:
    1. **Enter Salesforce Account ID** (18-character ID from Salesforce)
    2. **Enter Customer Name** (exact or partial name for validation)  
    3. **Click 'Validate Customer'** to verify data availability
    4. **Select Analysis Modules** to begin forecasting
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
