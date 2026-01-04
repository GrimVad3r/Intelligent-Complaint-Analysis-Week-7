# Exploratory Data Analysis (EDA) of Consumer Complaints

This Jupyter notebook contains the Exploratory Data Analysis phase of the Intelligent Complaint Analysis project. It focuses on understanding the distribution, characteristics, and quality of consumer complaint narratives before they are processed for semantic indexing.

---

## 📊 Analysis Objectives

The primary goal of this notebook is to perform a deep dive into the cleaned dataset to justify subsequent architectural decisions, such as chunking size and sampling strategies.

### 1. Data Loading & Initial Inspection

* **Source**: Loads the filtered and cleaned complaint dataset from `filtered_complaints.csv`.
* **Scope**: Inspects core features including `product_category`, `cleaned_narrative`, `Company`, and `State`.

### 2. Category Distribution

* **Proportional Analysis**: Evaluates the volume of complaints across the four primary categories: **Credit Card**, **Savings Account**, **Money Transfers**, and **Personal Loan**.
* **Sampling Foundation**: This distribution serves as the basis for the stratified sampling logic implemented in later stages of the pipeline.

### 3. Narrative Length Analysis

* **Tokenization & Counting**: Calculates the length of complaint narratives to determine the distribution of text sizes.
* **Strategic Justification**: Analyzing text length provides the empirical evidence needed to set the `chunk_size` (500) and `chunk_overlap` (50) in the `ComplaintChunker`.
* **Outlier Detection**: Identifies extremely short or excessively long complaints that may impact embedding quality.

### 4. Categorical Insights

* **Company Trends**: Identifies which companies receive the highest volume of complaints within the sampled data.
* **Geographic Trends**: Maps complaints by state to identify regional reporting patterns.

---

## 🛠️ Tools Used

| Library | Purpose |
| --- | --- |
| **Pandas** | Data manipulation and category frequency analysis. |
| **NumPy** | Numerical operations for length calculations. |
| **Matplotlib/Seaborn** | Visualization of distributions and correlations. |
| **Scikit-learn** | Initial testing of stratified split logic. |

---

## 📈 Key Findings for the Pipeline

* **Imbalance Handling**: Confirmed the need for stratified sampling to ensure smaller categories like "Personal Loan" are adequately represented.
* **Chunking Logic**: Determined that a recursive splitter is necessary because complaint narratives vary significantly in length and structure.
