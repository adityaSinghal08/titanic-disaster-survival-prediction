import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Stateless feature generator.

    Responsibilities:
    - Create semantic features
    - Handle missing cell values deterministically
    - Drop raw columns that should never reach the model

    Assumptions:
    - Input schema is stable (columns always exist)
    - Only cell-level missing values may occur

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
        # Missing Ticket ⇒ assume solo traveler
        # Create a unique placeholder for missing tickets
        missing_mask = X['Ticket'].isna()
        X.loc[missing_mask, 'Ticket'] = (
            'UNKNOWN_' + X.loc[missing_mask].index.astype(str)
        )

        # Now compute group size
        X['GroupSize'] = X.groupby('Ticket')['Ticket'].transform('count')

        # -------------------------------
        # Title extraction from Name
        # -------------------------------
        # Missing Name ⇒ Unknown title
        X['Title'] = X['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        X['Title'] = X['Title'].fillna('Unknown')

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
        # Age imputation (hierarchical)
        # -------------------------------
        # Level 1: Title + Pclass + Sex
        X['Age'] = X.groupby(
            ['Title', 'Pclass', 'Sex']
        )['Age'].transform(lambda s: s.fillna(s.median()))

        # Level 2: Title-only fallback
        X['Age'] = X.groupby('Title')['Age'].transform(
            lambda s: s.fillna(s.median())
        )

        # Level 3: Global median (guarantees no NaNs)
        X['Age'] = X['Age'].fillna(X['Age'].median())

        # -------------------------------
        # Family-based features
        # -------------------------------
        # Missing SibSp / Parch ⇒ assume zero
        X['SibSp'] = X['SibSp'].fillna(0)
        X['Parch'] = X['Parch'].fillna(0)

        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAloneFamily'] = (X['FamilySize'] == 1).astype(int)
        X['IsAloneGroup'] = (X['GroupSize'] == 1).astype(int)

        # -------------------------------
        # Fare cleanup
        # -------------------------------
        # Missing Fare ⇒ global median
        X['Fare'] = X['Fare'].fillna(X['Fare'].median())
        X['Fare'] = X['Fare'].round().astype(int)

        # -------------------------------
        # Drop raw / leakage-prone columns
        # -------------------------------
        X = X.drop(
            columns=['Name', 'Ticket', 'Cabin'],
            errors='ignore'
        )

        return X
