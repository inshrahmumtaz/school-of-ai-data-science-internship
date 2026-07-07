# SQL Analytics & KPI Report

## Business Context

This project analyzes the Sample Superstore retail dataset using SQL and Python to generate key business performance indicators (KPIs). The goal is to identify revenue trends, customer behavior, and category performance to support better business decisions.

---

## Dataset

**Sample Superstore Dataset**

The dataset contains retail order information including:

- Order ID
- Customer ID
- Customer Name
- Region
- Category
- Sales
- Quantity
- Discount
- Profit
- Order Date

---

## KPIs

The following KPIs were calculated using SQL:

- Total Revenue
- Revenue by Region
- Revenue by Category
- Revenue by Month
- Month-over-Month Growth
- Top 10 Customers by Revenue
- Average Order Value
- Orders per Customer

---

## Business Insights

### Observation
Technology generated the highest revenue among all product categories.

### Insight
A relatively small number of customers contributed a large share of total sales.

### Recommendation
- Focus marketing efforts on high-value customers.
- Increase inventory for top-performing product categories.
- Expand sales strategies in high-performing regions.
- Monitor monthly revenue trends to identify seasonal demand.

---

## SQL Highlights

Examples of SQL analysis performed:

- Aggregate functions (SUM, AVG, COUNT)
- GROUP BY
- ORDER BY
- Common Table Expressions (CTEs)
- Window Functions (LAG)
- Revenue trend analysis
- Customer segmentation

---

## Charts

The project includes the following visualizations:

- Revenue Trend
- Revenue by Category
- Top 10 Customers

These charts are available in:

```
reports/figures/
```

---

## Project Structure

```
06-sql-analytics-kpi/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_load_to_sql.ipynb
│   └── 02_kpi_analysis.ipynb
│
├── reports/
│   ├── figures/
│   └── kpi_summary.csv
│
├── sql/
│   └── kpi_queries.sql
│
└── README.md
```

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
notebooks/01_load_to_sql.ipynb
```

and

```
notebooks/02_kpi_analysis.ipynb
```

---

## Technologies Used

- Python
- Pandas
- SQLite
- SQL
- Matplotlib
- Seaborn
- Jupyter Notebook