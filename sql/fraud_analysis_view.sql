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
        when fraud_detected is true then 'fraud'
        else 'legitimate'
    end as transaction_status
from 
    transactions
where 
    transaction_date >= current_date - interval '30 days'
    and user_id is not null
order by 
    transaction_date desc; 

-- TODO: consider adding more filters for specific user segments

-- this view helps track recent transactions and their statuses for fraud analysis
-- it can be useful for identifying trends over the last month