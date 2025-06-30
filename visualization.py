import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve


class Visualizer:

    def plot_model_comparison(self, results: dict) -> go.Figure:
        names = list(results.keys())
        accs = [results[n]["accuracy"] for n in names]
        aucs = [results[n]["auc_score"] for n in names]

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Accuracy", "AUC Score"])
        fig.add_trace(go.Bar(x=names, y=accs, name="Accuracy"), row=1, col=1)
        fig.add_trace(go.Bar(x=names, y=aucs, name="AUC Score"), row=1, col=2)
        fig.update_layout(title="Model Performance Comparison", showlegend=False)
        return fig

    def plot_roc_curves(self, results: dict) -> go.Figure:
        fig = go.Figure()
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(res["y_test"], res["probs"])
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} (AUC={res['auc_score']:.3f})",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="red"),
                name="Random",
            )
        )
        fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        return fig

    def plot_confusion_matrix(self, y_test, y_pred, model_name: str) -> go.Figure:
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            labels=dict(x="Predicted", y="Actual"),
            title=f"Confusion Matrix — {model_name}",
        )
        return fig

    def plot_feature_importance(self, model, feature_names: list):
        if not hasattr(model, "feature_importances_"):
            return None, None
        imp = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        fig = px.bar(
            imp.head(15),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Feature Importances",
        )
        return fig, imp
