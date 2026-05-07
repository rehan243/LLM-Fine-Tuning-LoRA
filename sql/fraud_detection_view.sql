create or replace view fraud_detection_view as
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
        when user_id in (select user_id from flagged_users) then true
        else false
    end as is_flagged
from 
    transactions
where 
    transaction_date >= current_date - interval '30 days'
    and transaction_status = 'completed';

-- TODO: consider adding more filters based on user behavior
-- e.g., check for unusual patterns in transaction frequency

-- this view will help in analyzing potentially fraudulent transactions
-- and assist in generating reports for the fraud detection team