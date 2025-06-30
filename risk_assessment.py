import pandas as pd
import plotly.express as px


class RiskAssessor:

    def __init__(self):
        self.risk_tiers = {
            "Low Risk": (0.0, 0.3),
            "Medium Risk": (0.3, 0.7),
            "High Risk": (0.7, 1.0),
        }

    def predict_risk_tier(self, p: float) -> str:
        for tier, (lo, hi) in self.risk_tiers.items():
            if lo <= p < hi:
                return tier
        return "High Risk"

    def create_risk_dashboard(self, results: dict, X_test) -> tuple:
        best = max(results, key=lambda name: results[name]["auc_score"])
        probs = results[best]["probs"]
        tiers = [self.predict_risk_tier(p) for p in probs]
        dist = pd.Series(tiers).value_counts()

        fig_pie = px.pie(
            names=dist.index, values=dist.values, title="Risk Tier Distribution"
        )
        fig_hist = px.histogram(
            x=probs, nbins=50, title="Default Probability Distribution"
        )
        return fig_pie, fig_hist, dist
