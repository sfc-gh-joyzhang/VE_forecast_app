# Forecast Planning Tool

A Streamlit application for analyzing Snowflake consumption patterns and building forecasts. Automatically discovers customers via Salesforce integration and calculates growth rates from billing data.

## Features

### Organic Growth Analysis
- Calculate CAGR and CMGR for billing categories (Compute, Storage, Data Transfer, etc.)
- Custom date range selection for growth analysis
- Visual charts with 24-month historical data
- Apply calculated growth rates to forecast scenarios

### Customer Discovery
- Enter Salesforce Account ID to automatically discover customer
- Dynamic schema detection in FINOPS_OUTPUTS database
- Real customer name retrieval from Fivetran Salesforce data

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Snowflake Connection
Create `.env` file with your Snowflake credentials:
```bash
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_username
SNOWFLAKE_AUTHENTICATOR=externalbrowser
SNOWFLAKE_WAREHOUSE=SNOWADHOC
SNOWFLAKE_ROLE=PUBLIC
```

### 3. Run Application
```bash
streamlit run forecast_app.py
```

### 4. Usage
1. Enter 18-character Salesforce Account ID
2. App automatically finds customer name and billing schema
3. Analyze organic growth with custom date ranges
4. Apply calculated growth rates to forecast

## Data Requirements

The app uses billing data from `FINOPS_OUTPUTS.{CUSTOMER_SCHEMA}.BILLING` tables with columns:
- `MONTH` - Monthly billing period
- `REVENUE_CATEGORY` - Compute, Storage, etc.
- `REVENUE_GROUP` - Data Transfer, Priority Support, etc.  
- `REVENUE` - Dollar amounts

Customers are discovered via `FIVETRAN.SALESFORCE.ACCOUNT` table using Salesforce Account ID.

## Technical Details

- Built with Streamlit and Snowflake
- Uses browser authentication for local development
- Supports Streamlit in Snowflake deployment
- Modular design with analysis modules in `slides/` directory
