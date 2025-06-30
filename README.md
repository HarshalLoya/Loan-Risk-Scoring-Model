# LoanDefault Navigator 🏦🔍  
_A Streamlit app for end-to-end loan default risk prediction_

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg) ![Streamlit](https://img.shields.io/badge/streamlit-1.0%2B-orange.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Welcome!

LoanDefault Navigator is an interactive dashboard that lets you **load real loan data** (or generate realistic synthetic data), **engineer features**, **train multiple machine-learning models**, and **visualize** predictions and risk tiers—all in one place. Built with **Python**, **scikit-learn**, **Plotly**, and **Streamlit**, it’s perfect for analysts, data scientists, and decision-makers who need to understand credit risk quickly.

---

## 📁 Project Structure

```plaintext
loan-default-prediction/
├── data_generator.py        # Load real CSV or generate synthetic data + target mapping
├── feature_engineering.py   # Create credit-specific features (e.g. DTI bins, credit history length)
├── preprocessing.py         # Encode categorical, impute missing, preserve training feature set
├── modeling.py              # Train & evaluate Logistic Regression, Random Forest, GBM
├── visualization.py         # Plot model metrics: ROC, confusion matrix, importances
├── risk_assessment.py       # Map probabilities to Low/Medium/High risk tiers & dashboards
├── app.py                   # Streamlit dashboard wiring all components together
├── requirements.txt         # Python dependencies
└── README.md                # This file
````

---

## ✨ Key Features

* 🔄 **Flexible Data Input:** Automatically reads your loan CSV (if present) or generates a realistic synthetic dataset.
* 🛠️ **Custom Feature Engineering:** Calculates credit history length, DTI categories, utilization ratios, delinquency scores, and more.
* 🤖 **Multi-Model Training:** Compares Logistic Regression, Random Forest, and Gradient Boosting with accuracy & AUC metrics.
* 📊 **Interactive Visualizations:** Explore data distributions, model ROC curves, confusion matrices, and feature importances using Plotly charts.
* ⚠️ **Risk Tier Dashboard:** Classify loans into Low/Medium/High risk and view probability distributions.
* 🎯 **Individual Prediction:** Enter custom loan details in the UI to get a realtime default probability and actionable recommendation.

---

## 🚀 Installation & Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/loan-default-prediction.git
   cd loan-default-prediction
   ```

2. **Create & activate a Python virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate          # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Provide your own loan data**

   * Place your CSV file in the project folder (e.g. `data/loans.csv`).
   * Set the path in `app.py` when instantiating `DataLoader`:

     ```python
     loader = DataLoader(csv_path="data/loans.csv", n_samples=5000)
     ```
   * If no CSV is found, synthetic data will be generated automatically.

---

## ▶️ How to Run

Launch the Streamlit app with:

```bash
streamlit run app.py
```

* After startup, your default browser will open the dashboard.
* Use the sidebar to navigate between **Overview**, **Data Analysis**, **Model Performance**, **Risk Assessment**, and **Individual Prediction** pages.

---

## 📝 Additional Notes

* The synthetic data generator mimics LendingClub-style loans and injects realistic default probabilities.
* The preprocessing step **locks in** the full set of training features so that your custom inputs are always aligned.
* You can extend this pipeline by adding new features, models, or alternative sampling strategies (e.g., SMOTE).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m "Add awesome feature"`)
4. Push to your branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

Questions or suggestions? Open an issue or drop me a line!

```
