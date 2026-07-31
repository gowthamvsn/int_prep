# LeetCode SQL — 40 Real Interview Problems, Answers Included

SQL is the single most common LeetCode-tagged category data scientists actually get asked in screens — these 40 are real, well-known problems (by their actual LeetCode names, so you can find and run them there too), picked to cover every pattern that keeps reappearing: joins, self-joins, window functions, subqueries, and `GROUP BY`/`HAVING` logic. Terse by design — problem, schema assumption, solution, one-line technique note — built to review fast, not to re-derive from scratch. Builds directly on `sql-practice.md`.

**Visual + memory hook — the order SQL actually RUNS in, which explains at least a third of the "why doesn't this work" moments below:**
```
FROM  ──▶  WHERE  ──▶  GROUP BY  ──▶  HAVING  ──▶  SELECT  ──▶  ORDER BY  ──▶  LIMIT
 get the    filter       collapse       filter      pick        sort          cut
 table(s)   raw rows     into groups    GROUPS      columns/    the result    down
                                                     aliases
```
**Remember it as "the table exists before you can filter it, and groups exist before you can filter THEM"** — that single ordering fact is why `WHERE COUNT(*) > 5` is a syntax error (`COUNT(*)` isn't computed until `GROUP BY` runs, which is AFTER `WHERE`) and why problem #2's subquery-wrapping trick exists (`ORDER BY`/`LIMIT` run dead last, after `SELECT`, so an empty result at that point has nothing left to fall back to without a wrapping trick). Every "why can't I just write it the obvious way" moment in this doc traces back to this one diagram.

## Easy

**1. Combine Two Tables** — list every person's first/last name plus their city/state, even people with no address on file.
```sql
SELECT p.firstName, p.lastName, a.city, a.state
FROM Person p LEFT JOIN Address a ON p.personId = a.personId;
```
*Technique: LEFT JOIN to keep unmatched rows — an INNER JOIN would silently drop people with no address.*

**2. Second Highest Salary** — the second-highest distinct salary, or `NULL` if it doesn't exist.
```sql
SELECT (SELECT DISTINCT salary FROM Employee ORDER BY salary DESC LIMIT 1 OFFSET 1) AS SecondHighestSalary;
```
*Technique: wrapping in a subquery makes `NULL` the natural result when the OFFSET row doesn't exist, instead of an error.*

**3. Duplicate Emails** — find every email that appears more than once.
```sql
SELECT email FROM Person GROUP BY email HAVING COUNT(*) > 1;
```

**4. Customers Who Never Order** — customer names who have placed zero orders.
```sql
SELECT c.name AS Customers
FROM Customers c LEFT JOIN Orders o ON c.id = o.customerId
WHERE o.id IS NULL;
```
*Technique: LEFT JOIN + `WHERE ... IS NULL` is the standard "find the missing side" pattern.*

**5. Employees Earning More Than Their Managers**
```sql
SELECT e.name AS Employee
FROM Employee e JOIN Employee m ON e.managerId = m.id
WHERE e.salary > m.salary;
```
*Technique: self-join — the same table played twice, once as employee, once as manager.*

**6. Rising Temperature** — dates where the temperature was higher than the previous calendar day.
```sql
SELECT w2.id
FROM Weather w1 JOIN Weather w2 ON DATEDIFF(w2.recordDate, w1.recordDate) = 1
WHERE w2.temperature > w1.temperature;
```

**7. Employee Bonus** — employee names and bonus, including employees with no bonus row or a bonus under 1000.
```sql
SELECT e.name, b.bonus
FROM Employee e LEFT JOIN Bonus b ON e.empId = b.empId
WHERE b.bonus < 1000 OR b.bonus IS NULL;
```

**8. Not Boring Movies** — odd-numbered movie IDs, non-boring description, sorted by rating descending.
```sql
SELECT * FROM Cinema
WHERE id % 2 = 1 AND description <> 'boring'
ORDER BY rating DESC;
```

**9. Big Countries** — countries with area over 3,000,000 or population over 25,000,000.
```sql
SELECT name, population, area FROM World
WHERE area >= 3000000 OR population >= 25000000;
```

**10. Classes More Than 5 Students**
```sql
SELECT class FROM Courses GROUP BY class HAVING COUNT(DISTINCT student) >= 5;
```

**11. Find Customer Referee** — customers not referred by customer id 2 (including no referee at all).
```sql
SELECT name FROM Customer WHERE referee_id <> 2 OR referee_id IS NULL;
```
*Technique: the classic NULL trap — `referee_id <> 2` alone silently drops NULL rows since any comparison to NULL is unknown, not true.*

**12. Article Views I** — authors who viewed their own article, distinct, sorted ascending.
```sql
SELECT DISTINCT author_id AS id FROM Views
WHERE author_id = viewer_id
ORDER BY id;
```

**13. Invalid Tweets** — tweet ids where content is longer than 15 characters.
```sql
SELECT tweet_id FROM Tweets WHERE CHAR_LENGTH(content) > 15;
```

**14. Find Users With Valid E-Mails** — emails matching `name@leetcode.com` with a proper leading-letter local part.
```sql
SELECT * FROM Users
WHERE mail REGEXP '^[A-Za-z][A-Za-z0-9_.-]*@leetcode\\.com$';
```
*Technique: `REGEXP` for pattern validation SQL's plain `LIKE` can't express (anchored alternation of characters).*

**15. Patients With a Condition** — patients whose `conditions` field contains a code starting with "DIAB1" as a whole word.
```sql
SELECT * FROM Patients
WHERE conditions LIKE 'DIAB1%' OR conditions LIKE '% DIAB1%';
```
*Technique: two LIKE clauses because the code could be the first word or a later word — matching mid-string without the leading space would also match "PREDIAB1", a false positive.*

**16. Sales Person** — names of salespeople who never sold to "RED" company.
```sql
SELECT s.name
FROM SalesPerson s
WHERE s.sales_id NOT IN (
  SELECT o.sales_id FROM Orders o
  JOIN Company c ON o.com_id = c.com_id
  WHERE c.name = 'RED'
);
```

## Medium

**17. Nth Highest Salary** — generalize #2 to the Nth highest, as a reusable function.
```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  SET N = N - 1;
  RETURN (SELECT DISTINCT salary FROM Employee ORDER BY salary DESC LIMIT 1 OFFSET N);
END
```

**18. Rank Scores** — rank scores with ties sharing a rank and no gaps skipped after a tie.
```sql
SELECT score, DENSE_RANK() OVER (ORDER BY score DESC) AS `rank`
FROM Scores;
```
*Technique: `DENSE_RANK` specifically — `RANK` would leave a gap after a tie, which this problem explicitly disallows.*

**19. Consecutive Numbers** — any number that appears at least 3 times in a row (by consecutive `id`).
```sql
SELECT DISTINCT l1.num AS ConsecutiveNums
FROM Logs l1
JOIN Logs l2 ON l1.id = l2.id - 1
JOIN Logs l3 ON l1.id = l3.id - 2
WHERE l1.num = l2.num AND l2.num = l3.num;
```
*Technique: self-join three copies of the same table offset by id, so a single row lines up with "the next one" and "the one after that" simultaneously.*

**20. Department Highest Salary** — the highest-paid employee(s) per department.
```sql
SELECT d.name AS Department, e.name AS Employee, e.salary AS Salary
FROM Employee e
JOIN Department d ON e.departmentId = d.id
WHERE e.salary = (
  SELECT MAX(salary) FROM Employee e2 WHERE e2.departmentId = e.departmentId
);
```

**21. Department Top Three Salaries** — the top 3 *distinct* salaries per department.
```sql
SELECT d.name AS Department, e.name AS Employee, e.salary AS Salary
FROM Employee e JOIN Department d ON e.departmentId = d.id
WHERE (
  SELECT COUNT(DISTINCT e2.salary) FROM Employee e2
  WHERE e2.departmentId = e.departmentId AND e2.salary > e.salary
) < 3;
```
*Technique: "count how many distinct salaries beat mine, in my department" is a clean way to express "top 3" without a window function.*

**22. Exchange Seats** — swap each pair of adjacent student ids (1↔2, 3↔4, ...), leaving a final odd one in place.
```sql
SELECT
  CASE
    WHEN id % 2 = 1 AND id = (SELECT MAX(id) FROM Seat) THEN id
    WHEN id % 2 = 1 THEN id + 1
    ELSE id - 1
  END AS id,
  student
FROM Seat
ORDER BY id;
```

**23. Trips and Users** — the cancellation rate of non-banned client trips, per day, over a date range.
```sql
SELECT t.request_at AS Day,
  ROUND(SUM(CASE WHEN t.status <> 'completed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS 'Cancellation Rate'
FROM Trips t
JOIN Users u1 ON t.client_id = u1.users_id AND u1.banned = 'No'
JOIN Users u2 ON t.driver_id = u2.users_id AND u2.banned = 'No'
WHERE t.request_at BETWEEN '2013-10-01' AND '2013-10-03'
GROUP BY t.request_at;
```
*Technique: `SUM(CASE WHEN ... THEN 1 ELSE 0 END) / COUNT(*)` is the standard "rate as a fraction of rows" pattern.*

**24. Game Play Analysis IV** — the fraction of players who logged in again exactly one day after their very first login.
```sql
WITH first_login AS (
  SELECT player_id, MIN(event_date) AS first_date FROM Activity GROUP BY player_id
)
SELECT ROUND(
  (SELECT COUNT(*) FROM first_login f
   JOIN Activity a ON a.player_id = f.player_id AND a.event_date = DATE_ADD(f.first_date, INTERVAL 1 DAY))
  / (SELECT COUNT(*) FROM first_login), 2
) AS fraction;
```

**25. Product Sales Analysis III** — for each product, the quantity/price in the *first* year it sold.
```sql
SELECT s.product_id, s.year AS first_year, s.quantity, s.price
FROM Sales s
JOIN (SELECT product_id, MIN(year) AS min_year FROM Sales GROUP BY product_id) f
  ON s.product_id = f.product_id AND s.year = f.min_year;
```

**26. Students and Examinations** — every student × every subject combination, with 0 if they never took that exam.
```sql
SELECT st.student_id, st.student_name, su.subject_name, COUNT(e.subject_name) AS attended_exams
FROM Students st
CROSS JOIN Subjects su
LEFT JOIN Examinations e
  ON st.student_id = e.student_id AND su.subject_name = e.subject_name
GROUP BY st.student_id, st.student_name, su.subject_name
ORDER BY st.student_id, su.subject_name;
```
*Technique: `CROSS JOIN` to generate every possible pairing first, then LEFT JOIN the real attendance onto it — the only way to make combinations that never happened still show up as 0.*

**27. Monthly Transactions I** — count and total amount of transactions per month per country.
```sql
SELECT DATE_FORMAT(trans_date, '%Y-%m') AS month, country,
  COUNT(*) AS trans_count,
  SUM(CASE WHEN state = 'approved' THEN 1 ELSE 0 END) AS approved_count,
  SUM(amount) AS trans_total_amount,
  SUM(CASE WHEN state = 'approved' THEN amount ELSE 0 END) AS approved_total_amount
FROM Transactions
GROUP BY month, country;
```

**28. Last Person to Fit in a Bus** — the last person (by turn order) whose cumulative weight still fits under the limit.
```sql
WITH running AS (
  SELECT person_name, SUM(weight) OVER (ORDER BY turn) AS running_weight
  FROM Queue
)
SELECT person_name FROM running WHERE running_weight <= 1000
ORDER BY running_weight DESC LIMIT 1;
```
*Technique: `SUM(...) OVER (ORDER BY ...)` for a running total — the same window-function idea as `sql-practice.md`'s LAG/LEAD example, applied to cumulative sums instead.*

**29. Number of Calls Between Two Persons** — count and total duration of calls between each unordered pair of people.
```sql
SELECT LEAST(from_id, to_id) AS person1, GREATEST(from_id, to_id) AS person2,
  COUNT(*) AS call_count, SUM(duration) AS total_duration
FROM Calls
GROUP BY LEAST(from_id, to_id), GREATEST(from_id, to_id);
```
*Technique: `LEAST`/`GREATEST` collapse "A called B" and "B called A" into the same unordered pair before grouping.*

**30. Human Traffic of Stadium** — find stretches of 3+ *consecutive* stadium records that each had at least 100 visitors.
```sql
WITH qualifying AS (
  SELECT *, id - ROW_NUMBER() OVER (ORDER BY id) AS grp
  FROM Stadium WHERE people >= 100
)
SELECT id, visit_date, people FROM qualifying
WHERE grp IN (SELECT grp FROM qualifying GROUP BY grp HAVING COUNT(*) >= 3)
ORDER BY id;
```
*Technique: `id − ROW_NUMBER()` is a classic trick — for a run of truly consecutive ids, that difference is constant, so it becomes a free grouping key for "consecutive streaks."*

**31. Queries Quality and Percentage**
```sql
SELECT query_name,
  ROUND(AVG(rating / position), 2) AS quality,
  ROUND(100 * SUM(CASE WHEN rating < 3 THEN 1 ELSE 0 END) / COUNT(*), 2) AS poor_query_percentage
FROM Queries
GROUP BY query_name;
```

**32. Market Analysis I** — for each user, their join date and how many orders they placed in 2019.
```sql
SELECT u.user_id AS buyer_id, u.join_date,
  COUNT(o.order_id) AS orders_in_2019
FROM Users u
LEFT JOIN Orders o ON u.user_id = o.buyer_id AND YEAR(o.order_date) = 2019
GROUP BY u.user_id, u.join_date;
```
*Technique: the date filter goes in the `ON` clause, not `WHERE` — filtering in `WHERE` would silently turn the LEFT JOIN back into an INNER JOIN for any buyer with zero 2019 orders.*

**33. Project Employees I** — average experience-years of employees on each project.
```sql
SELECT p.project_id, ROUND(AVG(e.experience_years), 2) AS average_years
FROM Project p JOIN Employee e ON p.employee_id = e.employee_id
GROUP BY p.project_id;
```

**34. Percentage of Users Attended a Contest**
```sql
SELECT contest_id,
  ROUND(100 * COUNT(DISTINCT user_id) / (SELECT COUNT(*) FROM Users), 2) AS percentage
FROM Register
GROUP BY contest_id
ORDER BY percentage DESC, contest_id ASC;
```

## Hard

**35. Median Employee Salary** — the median salary within each company (odd or even count).
```sql
WITH ranked AS (
  SELECT company, salary,
    ROW_NUMBER() OVER (PARTITION BY company ORDER BY salary) AS rn,
    COUNT(*) OVER (PARTITION BY company) AS cnt
  FROM Employee
)
SELECT company, salary AS Salary
FROM ranked
WHERE rn IN (FLOOR((cnt + 1) / 2), FLOOR((cnt + 2) / 2));
```
*Technique: the two-`FLOOR` trick picks one middle row for an odd count and both middle rows for an even count with a single condition, avoiding separate odd/even branches.*

**36. Tree Node** — classify each node in a tree table as "Root", "Leaf", or "Inner".
```sql
SELECT id,
  CASE
    WHEN p_id IS NULL THEN 'Root'
    WHEN id NOT IN (SELECT DISTINCT p_id FROM Tree WHERE p_id IS NOT NULL) THEN 'Leaf'
    ELSE 'Inner'
  END AS type
FROM Tree;
```

**37. Delete Duplicate Emails** — keep only the lowest-id row per duplicate email (an actual `DELETE`, not a `SELECT`).
```sql
DELETE p1 FROM Person p1
JOIN Person p2 ON p1.email = p2.email AND p1.id > p2.id;
```
*Technique: self-join on "same email, higher id," then delete exactly those rows — the lowest-id row for each email never matches the `p1.id > p2.id` condition, so it survives.*

**38. Biggest Single Number** — the largest value that appears exactly once across a table.
```sql
SELECT MAX(num) AS num FROM (
  SELECT num FROM MyNumbers GROUP BY num HAVING COUNT(*) = 1
) t;
```

**39. Average Selling Price** — weighted average price per product, accounting for prices that change over date ranges.
```sql
SELECT p.product_id, ROUND(COALESCE(SUM(p.price * u.units) / NULLIF(SUM(u.units), 0), 0), 2) AS average_price
FROM Prices p
LEFT JOIN UnitsSold u
  ON p.product_id = u.product_id AND u.purchase_date BETWEEN p.start_date AND p.end_date
GROUP BY p.product_id;
```
*Technique: `NULLIF(SUM(units), 0)` avoids a divide-by-zero for a product with prices but zero recorded sales; `COALESCE(...,0)` then turns that resulting NULL into a clean 0.*

**40. Immediate Food Delivery II** — the percentage of customers whose *first* order was also their *immediate* (same-day) delivery order.
```sql
WITH first_order AS (
  SELECT customer_id, MIN(order_date) AS first_date FROM Delivery GROUP BY customer_id
)
SELECT ROUND(100 * SUM(CASE WHEN d.order_date = d.customer_pref_delivery_date THEN 1 ELSE 0 END) / COUNT(*), 2)
  AS immediate_percentage
FROM Delivery d
JOIN first_order f ON d.customer_id = f.customer_id AND d.order_date = f.first_date;
```

## Practice Q&A (Self-Test)

### Problem #2 (Second Highest Salary) uses `LIMIT 1 OFFSET 1` wrapped in a subquery instead of just `SELECT DISTINCT salary FROM Employee ORDER BY salary DESC LIMIT 1 OFFSET 1` directly. Why wrap it?
Without the wrapper, if there's no second-highest salary (e.g. only one employee exists), the query returns zero rows — but the problem requires returning one row containing `NULL`. Wrapping the whole thing as a scalar subquery inside a `SELECT` guarantees exactly one row is returned either way; if the inner query finds nothing, the outer `SELECT` naturally returns `NULL` instead of an empty result set.

### Why does problem #26 (Students and Examinations) need a `CROSS JOIN` before the `LEFT JOIN`, instead of just a `LEFT JOIN` from Students to Examinations directly?
A direct LEFT JOIN would only ever produce rows for subject/student combinations that already exist in the Examinations table (with NULLs filled in for missing *columns*, not missing *combinations*) — student/subject pairs with literally zero exam rows wouldn't appear at all. The `CROSS JOIN` first generates every possible student × subject combination as a starting scaffold, so the subsequent LEFT JOIN (and the `COUNT` that follows) can correctly show 0 for combinations that never happened.

### In problem #32, why does the `YEAR(o.order_date) = 2019` filter go in the `ON` clause instead of `WHERE`?
Putting it in `WHERE` would filter the joined result *after* the LEFT JOIN happens, which removes any row where that condition isn't met — including the placeholder NULL row LEFT JOIN creates for a buyer with zero 2019 orders. That silently converts the LEFT JOIN back into behaving like an INNER JOIN, dropping exactly the buyers (zero orders) the LEFT JOIN was meant to keep. Putting the condition in `ON` filters which rows get matched during the join itself, so buyers with no matching 2019 orders still get one row with `orders_in_2019 = 0` (via `COUNT`, which ignores NULLs) rather than disappearing.

### What's the shared idea behind the `id - ROW_NUMBER()` trick in #30 and the `LEAST/GREATEST` trick in #29?
Both convert something that's naturally hard to `GROUP BY` directly into something that is: `id - ROW_NUMBER()` turns "is this part of a consecutive run" into a constant value you can group on, and `LEAST/GREATEST` turns "these two rows represent the same unordered pair, regardless of column order" into a consistently-ordered pair you can group on. The general pattern — compute a derived value specifically so `GROUP BY` can express something SQL has no direct syntax for — is worth recognizing as a technique, not just memorizing per-problem.
