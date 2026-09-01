select
    *,
    case when internet_service <> 'No' then 1 else 0 end as has_internet,
    case when phone_service = 'Yes' then 1 else 0 end as has_phone,
    (
        case when phone_service = 'Yes' then 1 else 0 end
        + case when internet_service <> 'No' then 1 else 0 end
        + case when multiple_lines = 'Yes' then 1 else 0 end
        + case when online_security = 'Yes' then 1 else 0 end
        + case when online_backup = 'Yes' then 1 else 0 end
        + case when device_protection = 'Yes' then 1 else 0 end
        + case when tech_support = 'Yes' then 1 else 0 end
        + case when streaming_tv = 'Yes' then 1 else 0 end
        + case when streaming_movies = 'Yes' then 1 else 0 end
    )::integer as service_count,
    case when contract = 'Month-to-month' then 1 else 0 end as is_month_to_month,
    case when contract in ('One year', 'Two year') then 1 else 0 end
        as has_long_term_contract,
    monthly_charges / nullif(total_charges, 0)
        as monthly_to_total_charge_ratio,
    case
        when tenure = 0 then 'No tenure'
        when tenure <= 12 then '0-12 months'
        when tenure <= 24 then '13-24 months'
        when tenure <= 48 then '25-48 months'
        else '49+ months'
    end as tenure_group
from {{ ref('stg_telco_customers') }}
