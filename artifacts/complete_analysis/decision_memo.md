# Decision memo — Causal Cafe coupon campaign

## 1. Estimated effect
The naive treated-minus-control gap is 13.8%, but coupons were
sent preferentially to likely buyers, so this is not an ATE. After adjusting for
the seven pre-treatment features with a T-Learner, the estimated average effect
is +9.8% on 7-day purchase probability, ranging from
+0.2% to +21.9%
(5th–95th percentile) across customers.

## 2. Identification assumptions
- Conditional exchangeability: the DAG assumes loyalty, orders, app sessions,
  recency and order value capture all common causes of coupon and purchase.
- Overlap: estimated propensities range 0.12–0.52;
  0.0% of customers fall outside [0.05, 0.95].
- Consistency: one coupon type (20%), one outcome window (7 days).
- No post-treatment variables (redemption, post-coupon browsing) are adjusted for.

## 3. Budget impact
1,691 customers recommended; expected coupon spend
$2,499.84 of the $2,500 Growth Budget; estimated
incremental profit $927.19.
2,267 customers have positive expected value, so the budget binds.

## 4. Recommended action
Send coupons only to the recommended list: positive expected incremental profit,
ranked from highest, until the budget is spent. Stop ranking by purchase
probability; it spends the budget on customers who would buy anyway.

## 5. What still needs validation
- The ranking was checked on last month's randomized pilot: the logistic T-Learner
  scored a Qini of 29.2,
  against 6.6 for gradient boosting and
  17.6 for purchase probability.
- Run a randomized holdout inside the recommended list before scaling.
- Check calibration of the profit estimate: in the synthetic check the realised
  value was $844.73 against $927.19
  estimated, so treat the estimate as optimistic.
- Monitor redemption cost and re-fit when customer mix changes.
- Synthetic check only (not production evidence): the predictive ranking would
  have produced -$945.64 under the same budget.
