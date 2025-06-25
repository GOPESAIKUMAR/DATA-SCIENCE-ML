# 🚖 Maximizing Revenue for Taxi Cab Drivers through Payment Type Analysis

## 📌 Project Overview

* In the fast-paced taxi booking sector, maximizing revenue is crucial for driver profitability and long-term sustainability. This project aims to explore the impact of different payment methods on total fare amount to identify the most revenue-generating options for drivers. Using a real-world NYC taxi dataset, we conduct hypothesis testing to uncover whether payment method significantly influences fare values.

---

## 🎯 Objective

* The objective of this project is to run an A/B test to evaluate the relationship between total fare and method of payment. Using Python, we apply hypothesis testing and descriptive statistics to determine whether there's a meaningful difference in earnings when customers use credit cards compared to cash.

---

## ❓ Research Question

> Is there a relationship between total fare amount and payment type, and can we nudge customers toward methods that generate higher revenue for drivers — without negatively impacting customer experience?

---

## 🧰 Tools & Technologies

- **Python**: Data analysis and hypothesis testing
- **Pandas, NumPy**: Data wrangling and transformation
- **Matplotlib, Seaborn**: Data visualization
- **Scipy.stats**: Statistical testing (t-test, etc.)
- **Jupyter Notebook**: Analysis and storytelling

---

## 🗃️ Dataset Overview

The dataset is based on NYC Yellow Taxi trip records. It includes variables like:

| Field Name         | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `tpep_pickup_datetime` | Trip start time                                                          |
| `tpep_dropoff_datetime`| Trip end time                                                           |
| `Passenger_count`  | Number of passengers                                                        |
| `Trip_distance`    | Distance covered in miles                                                   |
| `Payment_type`     | Payment method (1=Credit card, 2=Cash, etc.)                                |
| `Fare_amount`      | Base fare calculated                                                        |
| `Tip_amount`       | Tip provided (auto for card, manual for cash)                               |
| `Total_amount`     | Total fare including tolls, tips, and surcharges                            |
| `RateCodeID`       | Pricing rate category                                                       |
| `Extra`, `MTA_tax`, `Tolls_amount`, `Improvement_surcharge`, `Airport_fee` | Additional fees     |

---

## 📊 Methodology

- Performed data cleaning and feature selection
- Grouped fare data by payment method
- Conducted statistical testing (two-sample t-test) to evaluate mean differences
- Visualized distributions and revenue trends per payment type

---

## 🔍 Key Insights

- Explored whether certain payment types (e.g., card vs. cash) correlate with higher fare amounts.
- Identified potential policy or interface nudges to encourage high-revenue behaviors.
- Evaluated findings while maintaining a balance between driver earnings and customer experience.

---

## 👨‍💻 Author

**Gope Sai Kumar**  
Data Analyst | Python • SQL • Power BI  
📧 saikumargope7@gmail.com  
🔗 [GitHub](https://github.com/GOPESAIKUMAR) • [LinkedIn](https://www.linkedin.com/in/sai-kumar-gope-618373289)

---


