# Snow Jumper Streamlit App - Deployment Guide

This guide explains how to develop locally and deploy your Snow Jumper Streamlit app to Snowflake using SnowCLI.

## 🏗️ Project Structure

```
SpecReq/
├── app.py                 # Main Streamlit application
├── browser_connection.py  # Snowflake connection helper
├── snowflake.yml         # SnowCLI configuration
├── requirements.txt      # Python dependencies
├── environment.yml       # Conda environment for Snowflake
├── deploy.sh             # Deployment script
├── setup-local.sh        # Local development setup
└── DEPLOYMENT.md         # This file
```

## 🚀 Quick Start

### 1. Set Up Local Development

Run the setup script to configure your local environment:

```bash
chmod +x setup-local.sh
./setup-local.sh
```

This script will:
- ✅ Check Python and Conda installation
- ✅ Create a conda environment (snow_jumper_local)
- ✅ Install required dependencies
- ✅ Create environment file template
- ✅ Set up .gitignore
- ✅ Install SnowCLI

### 2. Configure Snowflake Connection

#### Option A: Using SnowCLI Connection Manager
```bash
snow connection add
```

Follow the prompts to configure your connection.

#### Option B: Using Environment Variables
Edit the `.env` file created by the setup script:

```env
SNOWFLAKE_ACCOUNT=your-account-name
SNOWFLAKE_USER=your-username
SNOWFLAKE_PASSWORD=your-password
SNOWFLAKE_ROLE=your-role
SNOWFLAKE_WAREHOUSE=SNOWADHOC
SNOWFLAKE_DATABASE=TEMP
SNOWFLAKE_SCHEMA=JHILL
```

### 3. Test Locally

Activate the conda environment and run the app:

```bash
conda activate snow_jumper_local
streamlit run app.py
```

### 4. Deploy to Snowflake

Once you're satisfied with local testing, deploy to Snowflake:

```bash
./deploy.sh
```

Or deploy to a specific environment:

```bash
./deploy.sh PROD
```

## 📋 Prerequisites

### Software Requirements
- **Python 3.8+** - Required for Streamlit and SnowCLI
- **Conda** - Anaconda or Miniconda for environment management
- **SnowCLI** - Automatically installed by setup script
- **Streamlit** - For local development and testing

### Snowflake Requirements
- **Snowflake Account** with appropriate permissions
- **ACCOUNTADMIN** or **SYSADMIN** role (for creating databases/schemas)
- **Compute Warehouse** (app uses `SNOWADHOC` by default)
- **CREATE STREAMLIT** privilege in target database/schema

### Required Snowflake Permissions
Your user/role needs the following permissions:
```sql
-- Database and schema permissions
USE ROLE ACCOUNTADMIN;
GRANT USAGE ON DATABASE TEMP TO ROLE <your_role>;
GRANT ALL ON SCHEMA TEMP.JHILL TO ROLE <your_role>;

-- Streamlit permissions
GRANT CREATE STREAMLIT ON SCHEMA TEMP.JHILL TO ROLE <your_role>;
GRANT CREATE STAGE ON SCHEMA TEMP.JHILL TO ROLE <your_role>;

-- Warehouse permissions
GRANT USAGE ON WAREHOUSE SNOWADHOC TO ROLE <your_role>;
```

## 🔧 Configuration Files

### snowflake.yml
The main configuration file for SnowCLI deployment:

```yaml
definition_version: 2

entities:
  snow_jumper_request:
    type: streamlit
    identifier: snow-jumper-request
    stage: snow_jumper_stage
    query_warehouse: SNOWADHOC
    main_file: app.py
    title: "Snow Jumper Request"
    
    artifacts:
      - browser_connection.py
      - requirements.txt
      - environment.yml
      - app.py
```

### Key Configuration Options
- **identifier**: Name of the Streamlit app in Snowflake
- **stage**: Snowflake stage where app files are stored
- **query_warehouse**: Warehouse used to run the app
- **artifacts**: Files to deploy with the app

## 🔧 Deployment Scripts

### deploy.sh
Main deployment script that:
1. ✅ Validates SnowCLI installation
2. ✅ Tests Snowflake connection
3. ✅ Creates required database objects
4. ✅ Deploys the Streamlit app
5. ✅ Provides app URL

Usage:
```bash
./deploy.sh [ENVIRONMENT]
```

### setup-local.sh
Local development setup script that:
1. ✅ Sets up conda environment (snow_jumper_local)
2. ✅ Installs dependencies via conda and pip
3. ✅ Creates configuration templates
4. ✅ Configures git ignore rules

## 🌐 Environment Management

### Development Workflow
1. **Local Development**: Use `.env` for connection settings
2. **Testing**: Run `streamlit run app.py` locally
3. **Deployment**: Use `./deploy.sh` for Snowflake deployment

### Multiple Environments
The deployment script supports different environments:

```bash
./deploy.sh DEV    # Development environment
./deploy.sh TEST   # Testing environment  
./deploy.sh PROD   # Production environment
```

## 🐛 Troubleshooting

### Common Issues

#### SnowCLI Not Found
```bash
# Install SnowCLI
pip install snowflake-cli-labs

# Or run setup script
./setup-local.sh
```

#### Connection Errors
```bash
# Test your connection
snow connection test

# Reconfigure if needed
snow connection add
```

#### Permission Errors
Ensure your Snowflake user has the required permissions listed above.

#### App Not Loading
1. Check the Snowflake console for error messages
2. Verify all artifacts are properly uploaded
3. Check warehouse permissions

### Debugging Commands

```bash
# List deployed apps
snow streamlit list

# Get app logs
snow streamlit logs --name snow-jumper-request

# Get app URL
snow streamlit get-url --name snow-jumper-request

# Delete app (if needed)
snow streamlit drop --name snow-jumper-request
```

## 🔄 CI/CD Integration

### GitHub Actions Example
Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Snowflake

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install SnowCLI
      run: pip install snowflake-cli-labs
    
    - name: Configure Snowflake
      env:
        SNOWFLAKE_ACCOUNT: ${{ secrets.SNOWFLAKE_ACCOUNT }}
        SNOWFLAKE_USER: ${{ secrets.SNOWFLAKE_USER }}
        SNOWFLAKE_PASSWORD: ${{ secrets.SNOWFLAKE_PASSWORD }}
      run: |
        snow connection add --account $SNOWFLAKE_ACCOUNT \
                           --user $SNOWFLAKE_USER \
                           --password $SNOWFLAKE_PASSWORD
    
    - name: Deploy App
      run: ./deploy.sh PROD
```

## 📚 Additional Resources

- [Streamlit in Snowflake Documentation](https://docs.snowflake.com/en/developer-guide/streamlit/about-streamlit)
- [SnowCLI Documentation](https://docs.snowflake.com/en/developer-guide/snowflake-cli/overview)
- [Snowpark Python Documentation](https://docs.snowflake.com/en/developer-guide/snowpark/python/index)

## 🔒 Security Best Practices

1. **Never commit secrets** - Use `.env` files and add them to `.gitignore`
2. **Use minimal permissions** - Grant only necessary privileges
3. **Rotate credentials regularly** - Update passwords and API keys
4. **Use SSO when available** - Prefer external authenticators
5. **Monitor access logs** - Regular audit of app usage

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review Snowflake query history for errors
3. Check SnowCLI logs for detailed error messages
4. Ensure all prerequisites are met

For additional help, consult the Snowflake documentation or your organization's Snowflake administrator. 