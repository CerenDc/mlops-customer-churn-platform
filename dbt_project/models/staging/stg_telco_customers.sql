select
    customerID as customer_id,
    gender,
    SeniorCitizen as senior_citizen,
    Partner as partner,
    Dependents as dependents,
    tenure,
    PhoneService as phone_service,
    MultipleLines as multiple_lines,
    InternetService as internet_service,
    OnlineSecurity as online_security,
    OnlineBackup as online_backup,
    DeviceProtection as device_protection,
    TechSupport as tech_support,
    StreamingTV as streaming_tv,
    StreamingMovies as streaming_movies,
    Contract as contract,
    PaperlessBilling as paperless_billing,
    PaymentMethod as payment_method,
    MonthlyCharges as monthly_charges,
    TotalCharges as total_charges,
    Churn as churn,
    case Churn
        when 'Yes' then 1
        when 'No' then 0
    end as churn_flag,
    _processed_at,
    _source_file
from {{ source('processed', 'telco_customers') }}
