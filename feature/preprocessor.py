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
    - Impute missing values
    - Scale numeric features (optional)
    - Encode categorical features
    - Enforce final feature set
    
    Designed to be:
    - Pipeline-safe
    - GridSearch-compatible
    - Leakage-proof
    """

    def __init__(self, scaling=True):
        self.scaling = scaling

    def fit(self, X, y=None):

        # -------------------------------
        # Feature groups
        # -------------------------------

        # Continuous / count features
        num_cols = [
            'Age',
            'Fare',
            'FamilySize',
            'GroupSize'
        ]

        # Ordinal feature (hierarchy matters)
        ordinal_cols = ['Pclass']

        # Binary indicators
        binary_cols = [
            'IsAloneFamily',
            'IsAloneGroup'
        ]

        # Nominal categorical features
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
        # Full preprocessing pipeline
        # -------------------------------
        self.pipeline_ = Pipeline([
            ('features', FeatureGenerator()),
            ('preprocess', ColumnTransformer(
                transformers=[
                    ('num', num_pipe, num_cols),
                    ('cat', cat_pipe, cat_cols),
                    ('ord', 'passthrough', ordinal_cols),
                    ('bin', 'passthrough', binary_cols)
                ]
            ))
        ])

        self.pipeline_.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline_.transform(X)
