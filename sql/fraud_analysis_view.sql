create or replace view fraud_analysis as
select
    transaction_id,
    user_id,
    amount,
    transaction_date,
    case
        when amount > 1000 then 'high_value'
        when amount between 500 and 1000 then 'medium_value'
        else 'low_value'
    end as transaction_value,
    case
        when (select count(*) from transactions t2 where t2.user_id = t1.user_id and t2.transaction_date > t1.transaction_date - interval '30 days') > 5 then 'potential_fraud'
        else 'normal'
    end as fraud_status
from transactions t1
where transaction_date >= current_date - interval '90 days'
order by transaction_date desc;

-- TODO: consider adding more fields like payment method or location for deeper insights
-- this should help in identifying trends and patterns in potential fraud cases