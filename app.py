import os
import streamlit as st
import pandas as pd
import plotly.express as px

from data_generator import DataLoader
from feature_engineering import FeatureEngineer
from preprocessing import Preprocessor
from modeling import ModelTrainer
from visualization import Visualizer
from risk_assessment import RiskAssessor


def create_streamlit_dashboard():
    st.set_page_config(page_title="Loan Default Prediction Dashboard", layout="wide")
    st.title("🏦 Loan Default Prediction System")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Go to",
        [
            "Overview",
            "Data Analysis",
            "Model Performance",
            "Risk Assessment",
            "Individual Prediction",
        ],
    )

    if "df" not in st.session_state:
        with st.spinner("Loading data & training models..."):
            loader = DataLoader(csv_path="path/to/your_loans.csv", n_samples=15000)
            df_raw = loader.load_data()

            fe = FeatureEngineer()
            df_feat = fe.transform(df_raw)

            pre = Preprocessor()
            X, y = pre.encode_and_impute(df_feat)

            mt = ModelTrainer()
            results, X_test, X_test_scaled = mt.train_models(X, y)

            vis = Visualizer()
            ra = RiskAssessor()

            st.session_state.update(
                {
                    "df": df_raw,
                    "fe": fe,
                    "pre": pre,
                    "mt": mt,
                    "results": results,
                    "X_test": X_test,
                    "vis": vis,
                    "ra": ra,
                }
            )

    df = st.session_state["df"]
    fe = st.session_state["fe"]
    pre = st.session_state["pre"]
    mt = st.session_state["mt"]
    results = st.session_state["results"]
    X_test = st.session_state["X_test"]
    vis = st.session_state["vis"]
    ra = st.session_state["ra"]

    if page == "Overview":
        st.header("📊 Dataset Overview")
        total_loans = len(df)
        default_rate = df["target"].mean()
        avg_loan_amt = df["loan_amnt"].mean()
        avg_int_rate = df["int_rate"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Loans", total_loans)
        col2.metric("Default Rate", f"{default_rate:.2%}")
        col3.metric("Avg Loan Amount", f"${avg_loan_amt:,.0f}")
        col4.metric("Avg Interest Rate", f"{avg_int_rate:.1f}%")

        st.subheader("Sample Data")
        st.dataframe(df.head())

        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

    elif page == "Data Analysis":
        st.header("📈 Exploratory Data Analysis")

        default_by_grade = df.groupby("grade")["target"].mean().reset_index()
        fig1 = px.bar(
            default_by_grade,
            x="grade",
            y="target",
            title="Default Rate by Loan Grade",
            labels={"target": "Default Rate"},
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(
            df,
            x="int_rate",
            color=df["target"].map({0: "Good", 1: "Default"}),
            title="Interest Rate Distribution by Loan Outcome",
            labels={"color": "Outcome", "int_rate": "Interest Rate (%)"},
        )
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(
            df,
            x=df["target"].map({0: "Good", 1: "Default"}),
            y="dti",
            title="Debt-to-Income Ratio by Loan Outcome",
            labels={"x": "Outcome", "dti": "DTI Ratio"},
        )
        st.plotly_chart(fig3, use_container_width=True)

    elif page == "Model Performance":
        st.header("🤖 Model Performance Analysis")

        comp_fig = vis.plot_model_comparison(results)
        st.plotly_chart(comp_fig, use_container_width=True)

        roc_fig = vis.plot_roc_curves(results)
        st.plotly_chart(roc_fig, use_container_width=True)

        imp_fig, imp_df = vis.plot_feature_importance(
            results["Random Forest"]["model"], pre.feature_names
        )
        if imp_fig:
            st.subheader("Top Feature Importances")
            st.plotly_chart(imp_fig, use_container_width=True)
            st.dataframe(imp_df.head(10))

        st.subheader("Confusion Matrices")
        for name, res in results.items():
            cm_fig = vis.plot_confusion_matrix(res["y_test"], res["preds"], name)
            st.plotly_chart(cm_fig, use_container_width=True)

    elif page == "Risk Assessment":
        st.header("⚠️ Risk Tier Analysis")

        pie_fig, hist_fig, risk_dist = ra.create_risk_dashboard(results, X_test)
        c1, c2 = st.columns(2)
        c1.plotly_chart(pie_fig, use_container_width=True)
        c2.plotly_chart(hist_fig, use_container_width=True)

        st.subheader("Risk Tier Breakdown")
        st.dataframe(risk_dist.to_frame("Count"))

        st.subheader("Risk Tier Definitions")
        tier_defs = pd.DataFrame(
            [
                ["Low Risk", "0%–30%", "Very low default probability"],
                ["Medium Risk", "30%–70%", "Moderate risk — monitor closely"],
                ["High Risk", "70%–100%", "High default probability"],
            ],
            columns=["Tier", "Probability Range", "Description"],
        )
        st.dataframe(tier_defs)

    elif page == "Individual Prediction":
        st.header("🎯 Individual Loan Risk Prediction")
        st.write("Enter loan details to predict default probability:")

        col1, col2, col3 = st.columns(3)
        with col1:
            loan_amnt = st.number_input(
                "Loan Amount ($)", min_value=1000, max_value=40000, value=15000
            )
            int_rate = st.number_input(
                "Interest Rate (%)", min_value=5.0, max_value=25.0, value=12.0
            )
            annual_inc = st.number_input(
                "Annual Income ($)", min_value=20000, max_value=300000, value=60000
            )
            dti = st.number_input(
                "Debt-to-Income Ratio", min_value=0.0, max_value=40.0, value=15.0
            )

        with col2:
            grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
            home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
            emp_length = st.number_input(
                "Employment Length (yrs)", min_value=0, max_value=10, value=5
            )
            purpose = st.selectbox(
                "Loan Purpose",
                [
                    "debt_consolidation",
                    "credit_card",
                    "home_improvement",
                    "other",
                    "major_purchase",
                    "small_business",
                ],
            )

        with col3:
            verification_status = st.selectbox(
                "Income Verification", ["Verified", "Source Verified", "Not Verified"]
            )
            revol_util = st.number_input(
                "Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=50.0
            )
            open_acc = st.number_input(
                "Open Accounts", min_value=1, max_value=30, value=10
            )
            total_acc = st.number_input(
                "Total Accounts", min_value=5, max_value=80, value=20
            )

        if st.button("Predict Default Risk"):
            installment = loan_amnt * (int_rate / 100) / 12
            input_df = pd.DataFrame(
                [
                    {
                        "loan_amnt": loan_amnt,
                        "int_rate": int_rate,
                        "installment": installment,
                        "grade": grade,
                        "home_ownership": home_ownership,
                        "emp_length": emp_length,
                        "annual_inc": annual_inc,
                        "verification_status": verification_status,
                        "purpose": purpose,
                        "dti": dti,
                        "delinq_2yrs": 0,
                        "open_acc": open_acc,
                        "pub_rec": 0,
                        "revol_bal": 0,
                        "revol_util": revol_util,
                        "total_acc": total_acc,
                        "collections_12_mths_ex_med": 0,
                        "acc_now_delinq": 0,
                        "tot_coll_amt": 0,
                        "tot_cur_bal": 0,
                        "earliest_cr_line": pd.to_datetime("2015-01-01"),
                        "issue_d": pd.to_datetime("2020-01-01"),
                    }
                ]
            )

            df_feat = fe.transform(input_df)
            X_input, _ = pre.encode_and_impute(df_feat)
            X_input = X_input.reindex(columns=pre.feature_names, fill_value=0)

            best_name = max(results, key=lambda n: results[n]["auc_score"])
            model = mt.models[best_name]

            if best_name == "Logistic Regression":
                X_for_pred = mt.scaler.transform(X_input)
            else:
                X_for_pred = X_input

            prob = model.predict_proba(X_for_pred)[0, 1]
            tier = ra.predict_risk_tier(prob)

            c1, c2, c3 = st.columns(3)
            c1.metric("Default Probability", f"{prob:.1%}")
            c2.metric("Risk Tier", tier)
            color = {"Low Risk": "green", "Medium Risk": "orange"}.get(tier, "red")
            c3.markdown(
                f"<h3 style='color:{color};'>{tier}</h3>", unsafe_allow_html=True
            )

            st.subheader("Recommendations")
            if tier == "Low Risk":
                st.success("✅ Low risk. Approve with standard terms.")
            elif tier == "Medium Risk":
                st.warning("⚠️ Medium risk. Consider extra verification or higher rate.")
            else:
                st.error("❌ High risk. Recommend rejection or require collateral.")


if __name__ == "__main__":
    create_streamlit_dashboard()
