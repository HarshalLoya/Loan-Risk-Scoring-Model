import pandas as pd
from sklearn.impute import SimpleImputer


class Preprocessor:

    def __init__(self):
        self.feature_names = []

    def encode_and_impute(self, df: pd.DataFrame):
        categorical_cols = [
            "grade",
            "home_ownership",
            "verification_status",
            "purpose",
            "dti_category",
            "revol_util_category",
            "employment_stability",
        ]
        df_enc = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        drop_cols = ["earliest_cr_line", "issue_d"]
        if "target" in df_enc.columns:
            drop_cols.append("target")

        to_drop = [c for c in drop_cols if c in df_enc.columns]

        X = df_enc.drop(columns=to_drop)
        y = df_enc["target"] if "target" in df_enc.columns else None

        imputer = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

        if y is not None:
            self.feature_names = X_imp.columns.tolist()
        return X_imp, y
