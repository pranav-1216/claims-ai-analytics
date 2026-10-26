-- Analytical SQL used by src/sql_analysis.py (SQLite dialect).

-- name: rejection_by_reason
SELECT rejection_reason,
       COUNT(*)                                     AS n_claims,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_rejections,
       ROUND(SUM(billed_amount), 2)                 AS billed_at_risk
FROM claims
WHERE status = 'REJECTED'
GROUP BY rejection_reason
ORDER BY n_claims DESC;

-- name: provider_billing_vs_peers
-- Compare each provider's average billed amount to peers of the same specialty.
WITH prov AS (
    SELECT p.provider_id, p.specialty,
           COUNT(*) AS n_claims,
           AVG(c.billed_amount) AS avg_billed
    FROM claims c JOIN providers p USING (provider_id)
    GROUP BY p.provider_id, p.specialty
    HAVING COUNT(*) >= 20
)
SELECT provider_id, specialty, n_claims,
       ROUND(avg_billed, 2) AS avg_billed,
       ROUND(avg_billed / AVG(avg_billed) OVER (PARTITION BY specialty), 2) AS ratio_to_specialty_avg
FROM prov
ORDER BY ratio_to_specialty_avg DESC
LIMIT 15;

-- name: monthly_trend
SELECT strftime('%Y-%m', service_date) AS month,
       COUNT(*) AS n_claims,
       ROUND(100.0 * SUM(status = 'REJECTED') / COUNT(*), 2) AS rejection_rate_pct,
       ROUND(SUM(allowed_amount), 2) AS paid_amount
FROM claims
GROUP BY month
ORDER BY month;

-- name: claims_outside_coverage
-- Claims whose service date falls outside the member's enrollment window.
SELECT c.claim_id, c.member_id, c.service_date, e.enroll_date, e.term_date, c.status
FROM claims c JOIN enrollment e USING (member_id)
WHERE c.service_date < e.enroll_date OR c.service_date > e.term_date
LIMIT 20;

-- name: high_cost_without_auth
SELECT procedure_code, COUNT(*) AS n_claims,
       ROUND(100.0 * SUM(status = 'REJECTED') / COUNT(*), 2) AS rejection_rate_pct
FROM claims
WHERE has_auth = 0
GROUP BY procedure_code
ORDER BY rejection_rate_pct DESC;
