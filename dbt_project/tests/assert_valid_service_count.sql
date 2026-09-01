select customer_id, service_count
from {{ ref('fct_customer_churn_features') }}
where service_count < 0 or service_count > 9
