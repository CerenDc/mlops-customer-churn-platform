with staging as (
    select count(*) as customer_count
    from {{ ref('stg_telco_customers') }}
),
features as (
    select count(*) as customer_count
    from {{ ref('fct_customer_churn_features') }}
)
select
    staging.customer_count as staging_customer_count,
    features.customer_count as feature_customer_count
from staging
cross join features
where staging.customer_count <> features.customer_count
