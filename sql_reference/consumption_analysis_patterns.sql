-- SQL Reference Patterns for Consumption Analysis
-- Extracted from original template for future use

-- ==================================================================
-- MONTHLY USAGE WITH GROWTH CALCULATIONS
-- ==================================================================

-- Pattern: Monthly aggregation with rolling growth rates
WITH monthly_usage AS (
    SELECT  
        a.salesforce_account_name,
        a.salesforce_account_id,
        TO_DATE(DATE_TRUNC('MONTH', general_date)) AS year_month_day,
        
        ROUND(SUM(TOTAL_CREDITS), 0) AS total_credits,
        ROUND(SUM(daily_storage_tb), 0) AS total_daily_storage_tbs,
        ROUND(SUM(DATA_TRANSFER_TB), 0) AS data_transfer_tb

    FROM finance.customer.SNOWFLAKE_ACCOUNT_REVENUE a
    WHERE 1=1
        and salesforce_account_id IN ('ACCOUNT_ID_LIST')
        and general_date >= DATEADD(MONTH, -24, DATE_TRUNC('MONTH', CURRENT_DATE))
        AND general_date < DATE_TRUNC('MONTH', CURRENT_DATE) -- Exclude current month
    GROUP BY ALL
),
with_pct_change AS (
    SELECT 
        *,
        
        -- Percent Change from Previous Month
        ROUND(100.0 * (total_credits - LAG(total_credits) OVER (PARTITION BY salesforce_account_id ORDER BY year_month_day)) / NULLIF(LAG(total_credits) OVER (PARTITION BY salesforce_account_id ORDER BY year_month_day), 0), 2) AS PCT_CHANGE_TOTAL_CREDITS,
        
        ROUND(100.0 * (total_daily_storage_tbs - LAG(total_daily_storage_tbs) OVER (PARTITION BY salesforce_account_id ORDER BY year_month_day)) / NULLIF(LAG(total_daily_storage_tbs) OVER (PARTITION BY salesforce_account_id ORDER BY year_month_day), 0), 2) AS PCT_CHANGE_DAILY_STORAGE

    FROM monthly_usage
)

SELECT 
    *,
    
    -- Rolling Averages of % Change (3, 6, 12 months)
    ROUND(AVG(PCT_CHANGE_TOTAL_CREDITS) OVER (
        PARTITION BY salesforce_account_id 
        ORDER BY year_month_day 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS AVG_PCT_CHANGE_TOTAL_CREDITS_3MO,

    ROUND(AVG(PCT_CHANGE_TOTAL_CREDITS) OVER (
        PARTITION BY salesforce_account_id 
        ORDER BY year_month_day 
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ), 2) AS AVG_PCT_CHANGE_TOTAL_CREDITS_6MO,

    ROUND(AVG(PCT_CHANGE_TOTAL_CREDITS) OVER (
        PARTITION BY salesforce_account_id 
        ORDER BY year_month_day 
        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
    ), 2) AS AVG_PCT_CHANGE_TOTAL_CREDITS_12MO

FROM with_pct_change
ORDER BY salesforce_account_id, year_month_day;

-- ==================================================================
-- RUN RATE ANALYSIS
-- ==================================================================

-- Pattern: Daily run rates over different time periods
SELECT  
    a.SALESFORCE_ACCOUNT_NAME,
    a.SALESFORCE_ACCOUNT_ID,
    
    -- Credits
    ROUND(SUM(IFF(a.general_date >= current_date - 30, TOTAL_CREDITS, 0)) / 30, 2) AS AVG_DAILY_TOTAL_CREDITS_30D,
    ROUND(SUM(IFF(a.general_date >= current_date - 60, TOTAL_CREDITS, 0)) / 60, 2) AS AVG_DAILY_TOTAL_CREDITS_60D,
    ROUND(SUM(IFF(a.general_date >= current_date - 90, TOTAL_CREDITS, 0)) / 90, 2) AS AVG_DAILY_TOTAL_CREDITS_90D,
    
    -- Storage
    ROUND(SUM(IFF(a.general_date >= current_date - 30, DAILY_STORAGE_TB, 0)) / 30, 2) AS AVG_DAILY_STORAGE_TB_30D,
    ROUND(SUM(IFF(a.general_date >= current_date - 60, DAILY_STORAGE_TB, 0)) / 60, 2) AS AVG_DAILY_STORAGE_TB_60D,
    ROUND(SUM(IFF(a.general_date >= current_date - 90, DAILY_STORAGE_TB, 0)) / 90, 2) AS AVG_DAILY_STORAGE_TB_90D

FROM finance.customer.SNOWFLAKE_ACCOUNT_REVENUE a
WHERE 1=1
    and salesforce_account_id IN ('ACCOUNT_ID_LIST')
    AND a.general_date >= current_date - 90
    AND a.general_date < current_date
GROUP BY ALL;

-- ==================================================================
-- ACCOUNT HIERARCHY QUERIES  
-- ==================================================================

-- Pattern: Account selection and filtering
SELECT DISTINCT 
    SALESFORCE_ACCOUNT_ID, 
    SALESFORCE_ACCOUNT_NAME
FROM FINANCE.SALES_FINANCE.ACCOUNT_TO_TERRITORY_ASSOCIATION 
WHERE market = 'AMS'
    and GEO_NAME = 'AMSExpansion'
ORDER BY SALESFORCE_ACCOUNT_NAME;

-- Pattern: Snowflake account mapping
SELECT 
    SNOWFLAKE_DEPLOYMENT, 
    SNOWFLAKE_ACCOUNT_NAME, 
    SNOWFLAKE_ACCOUNT_ALIAS, 
    SNOWFLAKE_ACCOUNT_ID, 
    SALESFORCE_ACCOUNT_NAME, 
    SALESFORCE_ACCOUNT_ID, 
    sum(total_credits)
FROM finance.customer.SNOWFLAKE_ACCOUNT_REVENUE
WHERE general_date > current_date() - 365
    and SALESFORCE_ACCOUNT_ID in ('ACCOUNT_ID_LIST')
GROUP BY ALL
HAVING sum(total_credits) > 10;

-- ==================================================================
-- FINOPS SPECIFIC PATTERNS
-- ==================================================================

-- Pattern: Warehouse usage analysis (from UEW procedure)
SELECT 
    TO_CHAR(YEAR(CREATED_ON::DATE)) || '-Q' || TO_CHAR(QUARTER(CREATED_ON::DATE)) AS PERIOD,
    DATE_TRUNC('month', CREATED_ON::DATE) as MONTH,
    J.ACCOUNT_ID,
    WAREHOUSE_NAME,
    COUNT_IF(dur_xp_executing > 0) AS XP_JOBS,
    SUM(dur_xp_executing) AS dur_xp_executing,
    ROUND(SUM(DUR_XP_EXECUTING) / 1000 / 60 / 60, 1) AS CPU_HOURS,
    AVG(XP_CURRENT_CONCURRENCY_LEVEL) AS AVG_CONCURRENCY,
    SUM(TOTAL_DURATION) AS TOTAL_DURATION,
    SUM(stats:stats.producedRows) AS PRODUCED_ROWS,
    SUM(stats:stats.ioLocalTempWriteBytes) + SUM(stats:stats.ioRemoteTempWriteBytes) AS BYTES_SPILLED,
    COUNT(DISTINCT user_id) AS ACTIVE_USERS,
    COUNT(DISTINCT DATABASE_NAME) AS USED_DATABASE
FROM SNOWHOUSE_IMPORT.{deployment}.JOB_ETL_V J
JOIN FINOPS_OUTPUTS.{customer}.SCOPED_ACCOUNTS S 
    ON J.account_id = S.ACCOUNT_ID
WHERE J.CREATED_ON::DATE BETWEEN '{start_date}' AND '{end_date}'
    AND DUR_XP_EXECUTING > 0
    and WAREHOUSE_NAME not like 'COMPUTE_SERVICE_WH_%'
GROUP BY ALL;

-- Pattern: Cost/pricing analysis
SELECT 
    salesforce_account_id,
    salesforce_account_name,
    snowflake_account_id::INT as account_id,
    snowflake_deployment as deployment,
    MIN(storage_pricing) as storage_pricing_min,
    MAX(storage_pricing) as storage_pricing_max,
    ROUND(AVG(storage_pricing),2) as storage_price,
    ROUND(AVG(price_per_credit),2) as price_per_credit
FROM FINANCE.CUSTOMER.PRICING_DAILY pd
JOIN FINOPS_OUTPUTS.{customer}.SCOPED_ACCOUNTS s 
    on pd.snowflake_account_id = s.account_id
    and pd.snowflake_deployment = s.deployment
WHERE date_trunc('month',general_date) = '{month_start}'
GROUP BY ALL;
