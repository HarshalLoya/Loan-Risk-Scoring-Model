import pandas as pd
import numpy as np


class FeatureEngineer:

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df["earliest_cr_line"]):
            df["earliest_cr_line"] = pd.to_datetime(
                df["earliest_cr_line"], errors="coerce"
            )

        current_year = 2025
        df["credit_history_length"] = current_year - df["earliest_cr_line"].dt.year

        df["dti_category"] = pd.cut(
            df["dti"],
            bins=[0, 10, 20, 30, float("inf")],
            labels=["Low", "Medium", "High", "Very High"],
        )

        df["income_to_loan_ratio"] = df["annual_inc"] / df["loan_amnt"]

        df["revol_util_category"] = pd.cut(
            df["revol_util"],
            bins=[0, 30, 60, 80, 100],
            labels=["Low", "Medium", "High", "Very High"],
        )

        df["employment_stability"] = np.where(
            df["emp_length"] >= 5, "Stable", "Unstable"
        )

        df["credit_efficiency"] = df["revol_bal"] / (df["total_acc"] + 1)

        df["delinq_risk_score"] = (
            df["delinq_2yrs"] * 2
            + df["pub_rec"] * 3
            + df["collections_12_mths_ex_med"] * 2
            + df["acc_now_delinq"] * 4
        )

        df["loan_burden"] = df["installment"] / (df["annual_inc"] / 12)
        return df
