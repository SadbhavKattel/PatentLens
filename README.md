# PatentLens

PatentLens is a patent similarity search engine that uses **Natural Language Processing (NLP)** to identify existing patents that are most similar to a user's invention description. The project focuses on **AI and Machine Learning patents (CPC G06N3)** and demonstrates how text retrieval techniques can be used as a lightweight **prior-art search tool**.

Given a patent—or a brand-new invention idea—PatentLens retrieves and ranks the most relevant existing patents using **TF-IDF**, **Cosine Similarity**, and **Nearest Neighbor Search**.

---

## Features

- Search similar AI/ML patents from a USPTO patent corpus
- TF-IDF vectorization for keyword-based retrieval
- Cosine Similarity ranking
- Retrieval evaluation using patent citation data
- Exploratory Data Analysis (EDA)
- Bias and fairness evaluation
- Visualization of retrieval performance metrics

---

## Dataset

**patents_g06n3_wide.csv**

A curated dataset containing **3,000 U.S. patents** in CPC class **G06N3** (Artificial Intelligence / Neural Networks).

Each patent contains:

- publication_number
- filing_date
- publication_date
- title
- abstract
- cpc_codes
- cited_patents

---

## Repository Structure

```text
PatentLens/
│
├── .gitignore
├── README.md
├── RESULTS.md
├── app.py
├── patents_g06n3_wide.csv
├── requirements.txt
├── requirements-app.txt
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_model_training.ipynb       # Main training notebook
│   └── patent_similarity_engine.ipynb  # Prototype engine + week8 audit walkthrough
│
├── outputs/                          # committed charts and metric snapshots
│   ├── eda_overview.png
│   ├── mrr_comparison.png
│   ├── ndcg_comparison.png
│   ├── recall_comparison.png
│   ├── citation_signal_*.png/.csv
│   ├── product_*.png/.csv
│   ├── metrics_summary.csv
│   ├── metrics_pivot.csv
│   ├── significance_tests.csv
│   │
│   └── week8/
│       ├── metrics.md
│       └── figures/
│           ├── 1_retrieval_precision.png
│           ├── 2_roc_citation_links.png
│           ├── 3_threshold_audit.png
│           └── 4_hubness_bias.png
│
├── scripts/
│   ├── train.py                      # fit + evaluate all models, write models/
│   ├── citation_signal_test.py       # cited-vs-random pair separation (AUC)
│   ├── product_metrics.py            # latency, thresholds, diversity, footprint
│   └── product_metrics_charts.py     # charts for product_metrics.py
│
└── src/
    └── patentlens/
        ├── __init__.py
        ├── cleaning.py
        ├── data_fetch.py
        ├── evaluation.py
        └── retrieval.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/PatentLens.git
cd PatentLens
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For the web application:

```bash
pip install -r requirements-app.txt
```

---

## Running the Project

### Exploratory Data Analysis

Run:

```
notebooks/01_eda.ipynb
```

### Train and Evaluate the Model

Run:

```
notebooks/02_model_training.ipynb
```

or execute:

```bash
python scripts/train.py
```

### Launch the Application

The app is a Streamlit UI and reads the artifacts `scripts/train.py` writes to `models/`,
so run the training step first.

```bash
streamlit run app.py
```

---

## Evaluation

PatentLens evaluates retrieval quality using real patent citation relationships as ground truth.

Metrics include:

- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Precision
- ROC Analysis

Additional bias audits include:

- Threshold sensitivity
- Hubness bias analysis

Evaluation figures are available in:

```
outputs/week8/figures/
```

and summary metrics can be found in:

```
RESULTS.md
```

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Future Improvements

- Sentence Transformer embeddings
- FAISS semantic search
- Multi-CPC patent retrieval
- International patent support
- Explainable similarity highlighting

---

## Project

Developed as part of the **AI4ALL Ignite Responsible AI Project**, demonstrating how NLP-based information retrieval can support transparent and accessible patent prior-art search.
