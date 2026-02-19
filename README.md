# RetailMiner
<img width="572" height="259" alt="image" src="https://github.com/user-attachments/assets/1560da4b-589c-49cf-b87e-40fbf70b3028" />

Intelligent Knowledge Discovery from Supermarket Transaction Data

📌 Overview

RetailMiner is a data mining–based analytics platform that extracts hidden knowledge from supermarket transaction data. The system discovers abnormal billing behavior, customer segments, buying patterns, and seasonal trends using machine learning and data mining techniques, and presents insights through an interactive Streamlit dashboard.

🚀 Features

- Anomaly detection using Isolation Forest
- Customer segmentation using K-Means clustering
- Market basket analysis using Apriori
- Temporal and seasonal trend analysis
- Interactive Streamlit dashboard

🧠 Techniques Used

- Data preprocessing & feature engineering
- RFM analysis
- Isolation Forest
- K-Means clustering
- Association Rule Mining (Apriori)
- Temporal mining

🏗️ Project Structure
```
RetailMiner/
│
├── app/                # Streamlit dashboard
├── data/               # Dataset (ignored in git)
├── models/             # Trained models (.pkl)
├── notebook/           # Jupyter notebooks
├── outputs/            # Generated CSV insights
├── src/                # ML pipeline scripts
├── README.md
├── requirements.txt
└── .gitignore
```

▶️ How to Run

pip install -r requirements.txt
streamlit run app/app.py


📊 Dashboard Modules

- Overview metrics
- Anomaly analytics
- Customer segmentation
- Market basket analysis
- Seasonal trends
- Model information

