from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()

    def train_models(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        candidates = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }

        results = {}
        for name, model in candidates.items():
            if name == "Logistic Regression":
                model.fit(X_train_s, y_train)
                preds = model.predict(X_test_s)
                probs = model.predict_proba(X_test_s)[:, 1]
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                probs = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)

            results[name] = {
                "model": model,
                "accuracy": acc,
                "auc_score": auc,
                "y_test": y_test,
                "preds": preds,
                "probs": probs,
            }
            self.models[name] = model

        return results, X_test, X_test_s
