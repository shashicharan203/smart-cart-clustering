# 🛒 SmartCart — Customer Segmentation for Retail Marketing

Unsupervised machine learning pipeline that segments retail customers into distinct behavioral personas using **PCA + K-Means / Agglomerative Clustering**, enabling data-driven, targeted marketing strategies.

---

## 📌 Overview

Retailers waste marketing spend when every customer receives the same campaign. **SmartCart** solves this by clustering customers based on income, spending habits, purchase channels, and household profile — turning a flat customer list into **actionable segments** a marketing team can act on immediately.

The pipeline takes **2,240 raw customer records (22 features)** through a full data science workflow — cleaning, feature engineering, outlier handling, dimensionality reduction, and clustering — and ends with four clearly interpretable customer personas backed by real statistics.

---

## 🎯 Business Problem

> Given demographic and purchase-history data, can we group customers into segments that are meaningfully different in **value** and **behavior**, so marketing campaigns can be targeted rather than broadcast?

---

## 🗂️ Dataset

| | |
|---|---|
| **Records** | 2,240 customers (2,236 after outlier removal) |
| **Raw features** | 22 (demographics, spend by category, purchase channel, campaign response) |
| **Key fields** | `Income`, `Year_Birth`, `Education`, `Marital_Status`, `Kidhome`, `Teenhome`, `Dt_Customer`, `Mnt*` (spend per category), `Num*Purchases` (channel usage), `Response`, `Complain` |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| ML / Preprocessing | scikit-learn (`OneHotEncoder`, `StandardScaler`, `PCA`, `KMeans`, `AgglomerativeClustering`, `silhouette_score`) |
| Cluster Tuning | kneed (`KneeLocator` for elbow detection) |
| Environment | Jupyter Notebook |

---

## 🔄 Project Workflow

**1. Data Cleaning**
- Handled missing `Income` values via median imputation

**2. Feature Engineering**
- `Age` derived from `Year_Birth`
- `Customer_Tenure_Days` derived from `Dt_Customer` relative to the most recent signup
- `Total_Spending` aggregated across 6 product categories (wine, fruit, meat, fish, sweets, gold)
- `Total_Children` combined from `Kidhome` + `Teenhome`
- `Education` simplified into 3 tiers (Undergraduate / Graduate / Postgraduate)
- `Living_With` derived from marital status (Partner / Alone), with noisy categories (`Absurd`, `YOLO`) cleaned up

**3. Outlier Removal**
- Dropped implausible ages (> 90 years) and extreme incomes (> ₹600,000), reducing bias in cluster formation

**4. Exploratory Data Analysis**
- Pairplots and a correlation heatmap to understand feature relationships before modeling

**5. Encoding & Scaling**
- One-hot encoded categorical variables (`Education`, `Living_With`)
- Standardized all features with `StandardScaler` so no single feature dominates distance-based clustering

**6. Dimensionality Reduction**
- Applied **PCA (3 components)** for both noise reduction and 3D visualization of customer spread

**7. Optimal Cluster Selection**
- Combined **Elbow Method (WCSS)** and **Silhouette Score** analysis to validate the ideal number of clusters → **k = 4**

**8. Clustering**
- Compared **K-Means** and **Agglomerative Clustering (Ward linkage)**, both converging on 4 well-separated segments

**9. Cluster Characterization**
- Profiled each segment by income, spend, purchase channel, tenure, and household structure to translate clusters into business personas

---

## 📊 Key Results — Four Customer Personas

| Segment | Income | Total Spend | Household | Campaign Response | Profile |
|---|---|---|---|---|---|
| **Cluster 3 — Affluent Singles** | ~₹70.7K | ~₹1,190 (high) | Single, few kids | **32.0%** (highest) | Best campaign ROI target — high spend *and* high responsiveness |
| **Cluster 1 — Affluent Partners** | ~₹72.8K | ~₹1,237 (highest) | Partnered, few kids | 16.7% | High-value loyal spenders via store & catalog channels |
| **Cluster 2 — Budget Singles** | ~₹37.0K | ~₹166 (lowest) | Single, has kids | 14.2% | High web browsing, low conversion — price-sensitive |
| **Cluster 0 — Budget Partners** | ~₹39.7K | ~₹222 | Partnered, has kids | 7.6% (lowest) | Deal-driven, needs incentive-based campaigns |

**Business takeaway:** Clusters 1 & 3 (affluent, ~35% of spend concentration) justify premium/loyalty campaigns, while Clusters 0 & 2 respond better to discount-led, deal-based offers rather than broad promotions.

---

## 📁 Repository Structure

```
smartcart/
├── smartcart.ipynb          # Full analysis notebook (EDA → clustering → insights)
├── smartcart_customers.csv  # Raw dataset (not included — add locally)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd smartcart

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook smartcart.ipynb
```

---

---



---

## 👤 Author
shashi charan
