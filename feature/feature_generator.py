import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Stateless feature generator.
    
    Responsibilities:
    - Create new semantic features
    - Perform deterministic transformations
    - Drop raw columns that should never reach the model
    
    NOTE:
    - No scaling
    - No encoding
    - No target access
    """

    def fit(self, X, y=None):
        # Stateless: nothing to learn
        return self

    def transform(self, X):
        X = X.copy()

        # -------------------------------
        # Ticket-based group size
        # -------------------------------
        # People sharing the same ticket often traveled together
        X['GroupSize'] = X.groupby('Ticket')['Ticket'].transform('count')

        # -------------------------------
        # Title extraction from Name
        # -------------------------------
        X['Title'] = X['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # Normalize equivalent titles
        X['Title'] = X['Title'].replace({
            'Mlle': 'Miss',
            'Ms': 'Miss',
            'Mme': 'Mrs'
        })

        # Group rare titles
        X['Title'] = X['Title'].replace(
            [
                'Don', 'Rev', 'Dr', 'Major', 'Lady',
                'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'
            ],
            'Rare'
        )

        # -------------------------------
        # Age imputation (group-wise)
        # -------------------------------
        # Uses demographic similarity instead of global median
        X['Age'] = X.groupby(
            ['Title', 'Pclass', 'Sex']
        )['Age'].transform(lambda s: s.fillna(s.median()))

        # Safety fallback
        X['Age'] = X['Age'].fillna(X['Age'].median())

        # -------------------------------
        # Family-based features
        # -------------------------------
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAloneFamily'] = (X['FamilySize'] == 1).astype(int)
        X['IsAloneGroup'] = (X['GroupSize'] == 1).astype(int)

        # -------------------------------
        # Fare cleanup
        # -------------------------------
        X['Fare'] = X['Fare'].round().astype(int)

        # -------------------------------
        # Drop raw / leakage-prone columns
        # -------------------------------
        X = X.drop(
            columns=['Name', 'Ticket', 'Cabin'],
            errors='ignore'
        )

        return X
