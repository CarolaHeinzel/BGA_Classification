from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    import numpy as np


class StaticExpertFeatureSelector(BaseEstimator, TransformerMixin):
    """Keep the feature selected by expert knowledge."""

    method_name = "StaticExpertFeatures"

    def __init__(self):
        self.selected_feature_names: list[str] | None = None

    @property
    def expert_feature_names(self) -> list[str]:
        """List of feature names selected by expert knowledge."""
        return [str(i) for i in range(104)]

    def fit(self, X, y):
        self.selected_feature_names = [
            c for c in X.columns if c in self.expert_feature_names
        ]
        return self

    def transform(self, X):
        return X[self.selected_feature_names]


class RandomFeatureSelector(BaseEstimator, TransformerMixin):
    """Randomly select N features."""

    method_name = "RandomFeatures"

    def __init__(self, random_state: np.random.RandomState, n_features: int):
        self.rng = random_state
        self.n_features = n_features
        self.selected_feature_names: list[str] | None = None

    def fit(self, X, y):
        self.selected_feature_names = self.rng.choice(
            list(X.columns), size=self.n_features, replace=False
        ).tolist()
        return self

    def transform(self, X):
        return X[self.selected_feature_names]
