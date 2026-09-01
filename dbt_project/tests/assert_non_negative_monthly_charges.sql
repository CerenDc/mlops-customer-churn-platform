select customer_id, monthly_charges
from {{ ref('fct_customer_churn_features') }}
where monthly_charges < 0
