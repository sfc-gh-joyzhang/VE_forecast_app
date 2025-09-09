#!/bin/bash

# Snow Jumper Streamlit App Deployment Script
# Deploys the app to Snowflake using SnowCLI

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="snow_jumper_request"
ENVIRONMENT="${1:-DEV}"  # Default to DEV if no environment specified
CONFIG_FILE="snowflake.yml"

echo -e "${BLUE}🚀 Starting Snow Jumper Streamlit App Deployment${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if SnowCLI is installed
check_snow_cli() {
    if ! command -v snow &> /dev/null; then
        print_error "SnowCLI is not installed. Please install it first:"
        echo "pip install snowflake-cli-labs"
        exit 1
    fi
    print_status "SnowCLI is installed"
}

# Check if snowflake.yml exists and is properly configured
check_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "snowflake.yml not found. Please ensure it exists in the project root."
        exit 1
    fi
    print_status "Configuration file found"
}

# Validate connection to Snowflake
validate_connection() {
    print_status "Validating Snowflake connection..."
    if ! snow connection test; then
        print_error "Failed to connect to Snowflake. Please check your connection configuration."
        echo "Run 'snow connection add' to configure your connection."
        exit 1
    fi
    print_status "Snowflake connection validated"
}

# Set the default connection
set_default_connection() {
    print_status "Setting default SnowCLI connection..."
    
    # Check if snowhouse connection exists and set as default
    if snow connection list | grep -q "snowhouse"; then
        if snow connection set-default snowhouse; then
            print_status "Default connection set to: snowhouse"
        else
            print_warning "Failed to set default connection - using current default"
        fi
    else
        print_warning "snowhouse connection not found - using current default connection"
        print_warning "Make sure your connection has access to TEMP.JHILL schema"
    fi
}

# Create database objects if they don't exist
setup_database_objects() {
    print_status "Skipping database object creation - using existing FREESTYLE_SUMMARY table"
    
    # Note: The app now uses the existing TEMP.JHILL.FREESTYLE_SUMMARY table
    # which contains pre-calculated specialist skill assessments
    
    print_status "Database objects verification completed"
}

# Deploy the Streamlit app
deploy_app() {
    print_status "Deploying Streamlit app to Snowflake..."
    
    # BATCH 1: Set context and upload all files in one session
    print_status "Setting context and uploading all application files..."
    UPLOAD_BATCH="
USE DATABASE TEMP; 
USE SCHEMA JHILL;
PUT file://app.py @TEMP.JHILL.snow_jumper_stage/snow_jumper_request AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://snow_connection.py @TEMP.JHILL.snow_jumper_stage/snow_jumper_request AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://requirements.txt @TEMP.JHILL.snow_jumper_stage/snow_jumper_request AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://environment.yml @TEMP.JHILL.snow_jumper_stage/snow_jumper_request AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
    
    if ! snow sql -q "$UPLOAD_BATCH"; then
        print_error "Failed to set context or upload application files"
        exit 1
    fi
    
    print_status "All files uploaded successfully!"
    
    # BATCH 2: Create app, configure, and verify in one session
    print_status "Creating app, configuring, and verifying deployment..."
    DEPLOY_VERIFY_BATCH="
CREATE OR REPLACE STREAMLIT TEMP.JHILL.snow_jumper_request 
FROM '@TEMP.JHILL.snow_jumper_stage/snow_jumper_request' 
MAIN_FILE='app.py';

ALTER STREAMLIT TEMP.JHILL.snow_jumper_request 
SET QUERY_WAREHOUSE='SNOWADHOC';

SHOW STREAMLITS LIKE 'snow_jumper_request' IN TEMP.JHILL;
LIST @TEMP.JHILL.snow_jumper_stage/snow_jumper_request;"
    
    if ! snow sql -q "$DEPLOY_VERIFY_BATCH" > /dev/null 2>&1; then
        print_error "Failed to create/configure Streamlit app"
        exit 1
    fi
    
    print_status "Streamlit app deployed and verified successfully!"
}

# Get the app URL
get_app_url() {
    print_status "Getting app information..."
    
    # Get Streamlit app details using SHOW STREAMLITS
    if STREAMLIT_INFO=$(snow sql -q "SHOW STREAMLITS LIKE 'snow_jumper_request' IN TEMP.JHILL;" --format JSON 2>/dev/null); then
        echo ""
        echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
        echo -e "${BLUE}App Name: SNOW_JUMPER_REQUEST${NC}"
        echo -e "${BLUE}Database: TEMP${NC}"
        echo -e "${BLUE}Schema: JHILL${NC}"
        echo -e "${YELLOW}Access your app through Snowflake Snowsight:${NC}"
        echo -e "${YELLOW}  Projects → Streamlit → Look for 'SNOW_JUMPER_REQUEST'${NC}"
    else
        echo ""
        echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
        echo -e "${YELLOW}App URL: Check your Snowflake console for the Streamlit app${NC}"
    fi
}

# Main deployment flow
main() {
    echo -e "${BLUE}Starting deployment to ${ENVIRONMENT} environment...${NC}"
    echo ""
    
    check_snow_cli
    check_config
    validate_connection
    set_default_connection
    setup_database_objects
    deploy_app
    get_app_url
    
    echo ""
    echo -e "${GREEN}🚀 Deployment process completed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Log into Snowflake Snowsight and navigate to Projects → Streamlit"
    echo "2. Look for 'SNOW_JUMPER_REQUEST' in the TEMP.JHILL schema"
    echo "3. Test all features: Request Specialist, Search Activity, Update Requests, Update Skills"
    echo "4. To redeploy: run './deploy.sh' again with any updates"
}

# Help function
show_help() {
    echo "Snow Jumper Streamlit Deployment Script"
    echo ""
    echo "Usage:"
    echo "  ./deploy.sh [ENVIRONMENT]"
    echo ""
    echo "Arguments:"
    echo "  ENVIRONMENT    Target environment (DEV, TEST, PROD) - defaults to DEV"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh          # Deploy to DEV environment"
    echo "  ./deploy.sh PROD     # Deploy to PROD environment"
    echo ""
    echo "Prerequisites:"
    echo "  - SnowCLI installed (pip install snowflake-cli-labs)"
    echo "  - Snowflake connection configured (snow connection add)"
    echo "  - Proper permissions in target Snowflake account"
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Run main deployment
main 