-- this view aggregates transactional data to identify potential fraud patterns
create or replace view fraud_analysis as
select
    t.user_id,
    count(t.transaction_id) as total_transactions,
    sum(case when t.amount > 1000 then 1 else 0 end) as high_value_transactions,
    avg(t.amount) as average_transaction_value,
    max(t.transaction_date) as last_transaction_date
from
    transactions t
where
    t.transaction_date >= current_date - interval '30 days' -- last 30 days
group by
    t.user_id
having
    count(t.transaction_id) > 5 -- only consider users with more than 5 transactions
    and sum(case when t.amount > 1000 then 1 else 0 end) > 2 -- flagging users with multiple high-value transactions
order by
    total_transactions desc; -- sort by total transactions

-- TODO: consider adding a threshold for average transaction value to refine this view further