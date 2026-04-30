# 🛒 Best Buy Electronics Product Analysis

A comprehensive data analysis project exploring Best Buy product data — covering pricing, ratings, discounts, reviews, and consumer sentiment.

---

## 📌 Project Overview

This notebook performs end-to-end analysis on a Best Buy product dataset (`best_buy_products_filtered.csv`), progressing from raw data cleaning through exploratory analysis, predictive modeling, and NLP-based sentiment classification.

---

## 📂 Dataset

| Field | Description |
|---|---|
| `final_price` / `initial_price` | Product pricing (cleaned from string format) |
| `rating` | Average customer rating (0–5) |
| `reviews_count` | Number of customer reviews |
| `recommend_percentage` | % of customers who recommend the product |
| `discount` | Raw discount label (e.g. "Save $50") |
| `root_category` | Top-level product category |
| `release_date` | Product release date |
| `product_description` | Text description used for NLP |
| `esrb_rating` | ESRB classification (for gaming products) |
| `availability` | JSON-like field of availability types |

---

## 🔧 Pipeline

1. Data Cleaning

2. Univariate Analysis (EDA)

3. Bivariate & Multivariate Analysis

4. Machine Learning — Regression

| Model | Type |
|---|---|
| OLS Linear Regression | Baseline |
| Ridge Regression | L2 regularization |
| Lasso Regression | L1 regularization |

Evaluation: **R² score** and **Mean Squared Error (MSE)**

Multicollinearity check via **Variance Inflation Factor (VIF)**.

### 5. Time Series Forecasting
- **SARIMA(1,1,1)(1,1,1,12)** model on monthly product release counts
- Forecast next 12 months with confidence intervals

### 6. NLP — Sentiment Analysis
- Model: `facebook/bart-large-mnli` (zero-shot classification via HuggingFace)
- Labels: `Positive`, `Neutral`, `Negative`
- Applied to each product's `product_description`

Sentiment visualized via:
- Treemap by category
- Sankey diagram (category → sentiment flow)
- Bubble chart (rating vs. reviews, colored by sentiment)
- Polar area chart

### 7. Advanced Visualizations
- Word cloud of product descriptions
- Sunburst chart (category → year → ESRB rating)
- Radar plot of normalized metrics by category

---

## 📦 Libraries Used

```
pandas, numpy
plotly (express + graph_objects)
statsmodels (OLS, SARIMAX, VIF)
scikit-learn (LinearRegression, Ridge, Lasso, train_test_split, metrics)
transformers (HuggingFace pipeline)
wordcloud, matplotlib
sklearn.preprocessing (MinMaxScaler)
```

---

## 📤 Output

Final cleaned and sentiment-labeled dataset saved as:
```
final_best_buy_data.csv
```
