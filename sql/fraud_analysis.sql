-- this view aggregates fraud detection metrics by month
create or replace view monthly_fraud_analysis as
select
    date_trunc('month', transaction_date) as month,
    count(case when is_fraudulent then 1 end) as total_frauds,
    sum(case when is_fraudulent then transaction_amount else 0 end) as total_fraud_amount,
    count(*) as total_transactions,
    sum(transaction_amount) as total_transaction_value
from
    transactions
group by
    month
order by
    month desc;

-- this will help in tracking trends over time
-- TODO: consider adding filters for specific transaction types or customer segments