select customer_id, tenure
from {{ ref('fct_customer_churn_features') }}
where tenure < 0
