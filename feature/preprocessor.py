from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from .feature_generator import FeatureGenerator


class PreProcessor(BaseEstimator, TransformerMixin):
    """
    Full preprocessing transformer.

    Responsibilities:
    - Run FeatureGenerator
    - Impute any remaining missing cell values
    - Scale numeric features (optional)
    - Encode categorical features
    - Enforce final feature schema

    Assumptions:
    - Input schema is stable (columns always exist)
    """

    def __init__(self, scaling=True):
        self.scaling = scaling

    def fit(self, X, y=None):

        # -------------------------------
        # Feature groups
        # -------------------------------

        num_cols = [
            'Age',
            'Fare',
            'FamilySize',
            'GroupSize'
        ]

        ordinal_cols = ['Pclass']

        binary_cols = [
            'IsAloneFamily',
            'IsAloneGroup'
        ]

        cat_cols = [
            'Sex',
            'Embarked',
            'Title'
        ]

        # -------------------------------
        # Numeric pipeline
        # -------------------------------
        num_steps = [('impute', SimpleImputer(strategy='median'))]
        if self.scaling:
            num_steps.append(('scale', StandardScaler()))

        num_pipe = Pipeline(num_steps)

        # -------------------------------
        # Categorical pipeline
        # -------------------------------
        cat_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # -------------------------------
        # Ordinal pipeline (safe fallback)
        # -------------------------------
        ord_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent'))
        ])

        # -------------------------------
        # Binary pipeline (safe fallback)
        # -------------------------------
        bin_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent'))
        ])

        # -------------------------------
        # Full preprocessing pipeline
        # -------------------------------
        self.pipeline_ = Pipeline([
            ('features', FeatureGenerator()),
            ('preprocess', ColumnTransformer(
                transformers=[
                    ('num', num_pipe, num_cols),
                    ('cat', cat_pipe, cat_cols),
                    ('ord', ord_pipe, ordinal_cols),
                    ('bin', bin_pipe, binary_cols)
                ],
                remainder='drop'
            ))
        ])

        self.pipeline_.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline_.transform(X)
