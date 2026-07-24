create view fraud_detection as
select 
    transaction_id, 
    user_id, 
    transaction_amount, 
    transaction_date,
    case 
        when transaction_amount > (select avg(transaction_amount) from transactions) * 2 
            then 'high_value'
        when transaction_amount < 10 
            then 'low_value'
        else 'normal_value'
    end as transaction_category,
    case 
        when exists (
            select 1 
            from blacklisted_users 
            where blacklisted_users.user_id = transactions.user_id
        ) 
            then 'blacklisted'
        else 'not_blacklisted'
    end as user_status
from 
    transactions
where 
    transaction_date >= current_date - interval '30 days'
    and transaction_status = 'completed'
    and (user_status = 'blacklisted' or transaction_category = 'high_value')

-- TODO: consider adding more filters or grouping options based on feedback