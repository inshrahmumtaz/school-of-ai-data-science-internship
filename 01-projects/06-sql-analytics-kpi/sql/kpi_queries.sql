-- Total Revenue
SELECT SUM(Sales) AS Total_Revenue
FROM orders;

-- Revenue by Category
SELECT
    Category,
    SUM(Sales) AS Revenue
FROM orders
GROUP BY Category;

-- Revenue by Region
SELECT
    Region,
    SUM(Sales) AS Revenue
FROM orders
GROUP BY Region;

-- Top 10 Customers
SELECT
    [Customer Name],
    SUM(Sales) AS Revenue
FROM orders
GROUP BY [Customer Name]
ORDER BY Revenue DESC
LIMIT 10;

-- Average Order Value
SELECT
    AVG(Sales) AS Average_Order_Value
FROM orders;

-- Orders per Customer
SELECT
    COUNT(DISTINCT [Order ID]) * 1.0 /
    COUNT(DISTINCT [Customer ID]) AS Orders_Per_Customer
FROM orders;

-- Revenue by Month (CTE)
WITH monthly_sales AS (
    SELECT
        substr([Order Date],7,4) || '-' || substr([Order Date],1,2) AS Month,
        SUM(Sales) AS Revenue
    FROM orders
    GROUP BY Month
)
SELECT *
FROM monthly_sales
ORDER BY Month;

-- Month-over-Month Growth (CTE)
WITH monthly_sales AS (
    SELECT
        substr([Order Date],7,4) || '-' || substr([Order Date],1,2) AS Month,
        SUM(Sales) AS Revenue
    FROM orders
    GROUP BY Month
)
SELECT
    Month,
    Revenue,
    100.0 * (Revenue - LAG(Revenue) OVER (ORDER BY Month))
    / LAG(Revenue) OVER (ORDER BY Month) AS Growth_Percent
FROM monthly_sales;