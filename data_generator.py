import os
import pandas as pd
import numpy as np
import sqlite3


class DataLoader:
    REQUIRED_COLS = [
        "loan_amnt",
        "funded_amnt",
        "funded_amnt_inv",
        "term",
        "int_rate",
        "installment",
        "grade",
        "sub_grade",
        "emp_length",
        "home_ownership",
        "annual_inc",
        "verification_status",
        "issue_d",
        "loan_status",
        "purpose",
        "dti",
        "delinq_2yrs",
        "earliest_cr_line",
        "inq_last_6mths",
        "mths_since_last_delinq",
        "mths_since_last_record",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "collections_12_mths_ex_med",
        "acc_now_delinq",
        "tot_coll_amt",
        "tot_cur_bal",
    ]

    STATUS_MAP = {
        "Fully Paid": 0,
        "Current": 0,
        "Charged Off": 1,
        "Late (31-120 days)": 0,
        "In Grace Period": 0,
        "Late (16-30 days)": 0,
        "Does not meet the credit policy. Status:Fully Paid": 0,
        "Does not meet the credit policy. Status:Charged Off": 1,
        "Default": 1,
    }

    def __init__(self, csv_path: str = None, n_samples: int = 10000):
        self.csv_path = csv_path
        self.n_samples = n_samples

    def load_data(self) -> pd.DataFrame:
        if self.csv_path and os.path.exists(self.csv_path):
            df = pd.read_csv(
                self.csv_path,
                parse_dates=["issue_d", "earliest_cr_line"],
                low_memory=False,
            )
            missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            df = df[self.REQUIRED_COLS].copy()

            df["target"] = df["loan_status"].map(self.STATUS_MAP)
            df = df[df["target"].notna()]
            df["target"] = df["target"].astype(int)

            df.drop(columns=["loan_status"], inplace=True)

        else:
            df = self.create_sample_data(self.n_samples)
            df.rename(columns={"loan_status": "target"}, inplace=True)

        return df

    def create_sample_data(self, n_samples=10000) -> pd.DataFrame:
        np.random.seed(42)
        data = {
            "loan_amnt": np.random.normal(15000, 8000, n_samples).clip(1000, 40000),
            "term": np.random.choice([36, 60], n_samples, p=[0.7, 0.3]),
            "int_rate": np.random.normal(12, 4, n_samples).clip(5, 25),
            "installment": np.random.normal(400, 200, n_samples).clip(50, 1200),
            "grade": np.random.choice(
                ["A", "B", "C", "D", "E", "F", "G"],
                n_samples,
                p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.08, 0.02],
            ),
            "emp_length": np.random.choice(
                list(range(0, 11)),
                n_samples,
                p=[0.1] + [0.08] * 9 + [0.18],
            ),
            "home_ownership": np.random.choice(
                ["RENT", "OWN", "MORTGAGE"], n_samples, p=[0.4, 0.2, 0.4]
            ),
            "annual_inc": np.random.lognormal(10.5, 0.8, n_samples).clip(20000, 300000),
            "verification_status": np.random.choice(
                ["Verified", "Source Verified", "Not Verified"],
                n_samples,
                p=[0.3, 0.3, 0.4],
            ),
            "purpose": np.random.choice(
                [
                    "debt_consolidation",
                    "credit_card",
                    "home_improvement",
                    "other",
                    "major_purchase",
                    "small_business",
                ],
                n_samples,
                p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05],
            ),
            "dti": np.random.normal(15, 8, n_samples).clip(0, 40),
            "delinq_2yrs": np.random.poisson(0.5, n_samples).clip(0, 10),
            "earliest_cr_line": np.random.choice(range(1980, 2020), n_samples),
            "open_acc": np.random.poisson(10, n_samples).clip(1, 30),
            "pub_rec": np.random.poisson(0.2, n_samples).clip(0, 5),
            "revol_bal": np.random.lognormal(8, 1.5, n_samples).clip(0, 100000),
            "revol_util": np.random.normal(50, 25, n_samples).clip(0, 100),
            "total_acc": np.random.poisson(20, n_samples).clip(5, 80),
            "collections_12_mths_ex_med": np.random.poisson(0.1, n_samples).clip(0, 5),
            "acc_now_delinq": np.random.poisson(0.05, n_samples).clip(0, 3),
            "tot_coll_amt": np.random.exponential(1000, n_samples).clip(0, 50000),
            "tot_cur_bal": np.random.lognormal(10, 1, n_samples).clip(0, 500000),
        }
        df = pd.DataFrame(data)
        default_prob = (
            df["grade"].map(
                {
                    "A": 0.02,
                    "B": 0.05,
                    "C": 0.08,
                    "D": 0.12,
                    "E": 0.18,
                    "F": 0.25,
                    "G": 0.35,
                }
            )
            + (df["dti"] / 100)
            + (df["delinq_2yrs"] * 0.1)
            + (df["pub_rec"] * 0.05)
            + (df["int_rate"] / 100)
            + np.where(df["annual_inc"] < 40000, 0.1, 0)
            + np.random.normal(0, 0.05, n_samples)
        ).clip(0, 1)

        df["loan_status"] = np.random.binomial(1, default_prob, n_samples)
        return df

    def setup_database(self, df: pd.DataFrame) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        df.to_sql("loans", conn, if_exists="replace", index=False)
        return conn
