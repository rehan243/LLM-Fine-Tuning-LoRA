-- this sql view will help analyze potential fraudulent transactions
-- it checks for patterns in transaction amounts and frequencies

create or replace view fraud_analysis as
select 
    user_id,
    count(*) as transaction_count,
    sum(amount) as total_spent,
    avg(amount) as average_transaction,
    max(amount) as max_transaction,
    min(amount) as min_transaction,
    case 
        when count(*) > 10 and avg(amount > 500) then 'high risk'
        when sum(amount) > 10000 then 'medium risk'
        else 'low risk'
    end as risk_level
from 
    transactions
where 
    transaction_date >= current_date - interval '30 days'
group by 
    user_id
having 
    count(*) >= 5
order by 
    risk_level desc, transaction_count desc
-- TODO: maybe add more filters based on location or time of day