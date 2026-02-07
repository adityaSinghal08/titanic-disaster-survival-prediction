# 🚢 Titanic Disaster Survival Prediction using Machine Learning

A comprehensive machine learning project predicting passenger survival on the Titanic using various ML algorithms, advanced feature engineering, and ensemble methods. This project is part of the **Kaggle Getting Started Competition - TITANIC: MACHINE LEARNING FROM DISASTER**, designed as a beginner-friendly introduction to machine learning classification problems.

---

## 🎯 Project Overview

This project analyzes the famous Titanic dataset to predict passenger survival using machine learning techniques. The implementation demonstrates a complete end-to-end ML workflow including exploratory data analysis, feature engineering, individual model training, and advanced ensemble methods such as soft voting and stacking.

**What makes this project unique:**
- Modular Python codebase with custom feature engineering modules
- Comprehensive comparison of 6 different classification algorithms
- Advanced ensemble techniques including soft voting and stacking
- Detailed error analysis and model selection strategies
- Production-ready code structure with reusable components

---

## 📊 Dataset Description

The project uses the Titanic dataset containing passenger information:

| Feature | Description | Type |
|---------|-------------|------|
| `PassengerId` | Unique identifier for each passenger | Integer |
| `Survived` | **Target Variable** - Survival (0 = No, 1 = Yes) | Binary |
| `Pclass` | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) | Categorical |
| `Name` | Passenger name | Text |
| `Sex` | Gender | Categorical |
| `Age` | Age in years | Numeric |
| `SibSp` | Number of siblings/spouses aboard | Integer |
| `Parch` | Number of parents/children aboard | Integer |
| `Ticket` | Ticket number | Text |
| `Fare` | Passenger fare | Numeric |
| `Cabin` | Cabin number | Text |
| `Embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) | Categorical |

**Dataset characteristics:**
- Binary classification problem (Survived: Yes/No)
- Contains missing values requiring imputation
- Mix of numerical and categorical features
- Requires extensive feature engineering for optimal performance

---

## 🏗️ Project Structure

```
titanic-disaster-survival-prediction/
├── data/                                    # Dataset files
│   ├── train.csv                           # Original training data
│   ├── train_partial_fe.csv                # Training data with feature engineering
│   └── test.csv                            # Test data for predictions
│
├── data exploration/                        # Exploratory Data Analysis
│   ├── eda.ipynb                           # Comprehensive EDA notebook
│   ├── partial_feature_engineering.ipynb   # Feature creation experiments
│   └── eda_observations.txt                # Key insights from EDA
│
├── feature/                                 # Feature engineering modules
│   ├── __init__.py                         # Package initialization
│   ├── feature_generator.py                # Custom feature creation
│   └── preprocessor.py                     # Data preprocessing utilities
│
├── models/                                  # Individual ML models
│   ├── decision_tree.ipynb                 # Decision Tree Classifier
│   ├── knn.ipynb                           # K-Nearest Neighbors
│   ├── lightgbm.ipynb                      # LightGBM Classifier
│   ├── logistic_regression.ipynb           # Logistic Regression
│   ├── random_forest.ipynb                 # Random Forest Classifier
│   ├── xgboost.ipynb                       # XGBoost Classifier
│   └── observations.txt                    # Model performance summary
│
├── ensemble models/                         # Advanced ensemble methods
│   ├── soft_voting_xgb_lgbm_lr.ipynb       # Soft voting ensemble
│   ├── soft_voting_xgb_rf_knn.ipynb        # Alternative soft voting
│   ├── stacking_xgb_knn.ipynb              # Stacking with 2 base models
│   ├── stacking_xgb_rf_knn.ipynb           # Stacking with 3 base models
│   └── oof_error_analysis_and_model_selection.ipynb  # Error analysis
│
├── output/                                  # Model predictions
│   ├── decision_tree_predictions.csv
│   ├── knn_predictions.csv
│   ├── lightgbm_predictions.csv
│   ├── logistic_regression_predictions.csv
│   ├── random_forest_predictions.csv
│   ├── xgboost_predictions.csv
│   ├── soft_voting_xgb_lgbm_lr.csv
│   ├── soft_voting_xgb_rf_knn.csv
│   ├── stacking_xgb_rf_knn.csv
│   └── stack_xgb_knn.csv
│
├── .gitignore
└── README.md
```

---

## 🔍 Key Insights from EDA

Based on comprehensive exploratory data analysis, several important survival patterns were discovered:

### Demographic Factors
1. **Gender (Sex)**: Female passengers had a ~74% survival rate vs ~19% for males
2. **Passenger Class (Pclass)**: 
   - 1st class: ~63% survival rate
   - 2nd class: ~47% survival rate
   - 3rd class: ~24% survival rate
3. **Age**: Children (under 16) had significantly better survival rates

### Socioeconomic Indicators
4. **Fare**: Higher fares strongly correlated with better survival rates
5. **Port of Embarkation**: 
   - Cherbourg (C): ~55% survival
   - Queenstown (Q): ~39% survival
   - Southampton (S): ~34% survival
6. **Title Extraction**: Titles (Mr., Mrs., Miss, Master) revealed social status patterns

### Family Relationships
7. **Family Size**: Passengers with 1-3 family members had optimal survival rates
8. **Solo Travelers**: Passengers traveling alone had lower survival rates
9. **Large Families**: Families with 4+ members had reduced survival rates

These insights directly informed the feature engineering strategy.

---

## 🧠 Feature Engineering

The project implements sophisticated feature engineering through modular Python classes:

### Custom Features Created
- **Title Extraction**: Extracted from passenger names (Mr, Mrs, Miss, Master, etc.)
- **Family Size**: Combined SibSp and Parch into total family members
- **Is Alone**: Binary indicator for solo travelers
- **Age Groups**: Binned ages into meaningful categories (Child, Teen, Adult, Senior)
- **Fare Per Person**: Fare divided by family size
- **Cabin Deck**: Extracted deck letter from cabin numbers
- **Name Length**: Character count as proxy for social status

### Preprocessing Steps
- Missing value imputation using median/mode strategies
- Categorical encoding (One-Hot, Label Encoding)
- Feature scaling using StandardScaler
- Outlier detection and handling

---

## 🤖 Models Implemented

### Individual Classification Models

| Model | Algorithm Type | Key Characteristics |
|-------|---------------|---------------------|
| **Logistic Regression** | Linear classifier | Baseline model, interpretable coefficients |
| **Decision Tree** | Tree-based | Interpretable, handles non-linear relationships |
| **K-Nearest Neighbors** | Instance-based | Distance-based classification |
| **Random Forest** | Ensemble (Bagging) | Robust to overfitting, feature importance |
| **XGBoost** | Ensemble (Boosting) | High performance, handles imbalance |
| **LightGBM** | Ensemble (Boosting) | Fast training, efficient memory usage |

### Advanced Ensemble Methods

#### 1. Soft Voting Ensembles
Combines predictions using weighted probability averaging:
- **XGBoost + LightGBM + Logistic Regression**: Balances boosting with linear baseline
- **XGBoost + Random Forest + KNN**: Diverse algorithm combination

#### 2. Stacking Ensembles
Meta-learning approach with two-layer architecture:
- **Base Models**: Generate predictions as meta-features
- **Meta-Learner**: Logistic Regression combines base model outputs
- Configurations tested:
  - XGBoost + KNN (2-model stack)
  - XGBoost + Random Forest + KNN (3-model stack)

#### 3. Out-of-Fold (OOF) Predictions
- Used for unbiased meta-feature generation
- Prevents overfitting in stacking
- Enables robust error analysis

---

## ⚙️ Tech Stack

- **Python 3.x** - Programming language
- **Pandas, NumPy** - Data manipulation and numerical computing
- **Scikit-learn** - ML models, preprocessing, and evaluation
- **XGBoost** - Gradient boosting framework
- **LightGBM** - Microsoft's gradient boosting
- **Matplotlib, Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development environment

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter
```

### Installation

1️⃣ **Clone the Repository**
```bash
git clone https://github.com/adityaSinghal08/titanic-disaster-survival-prediction.git
cd titanic-disaster-survival-prediction
```

2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### Usage Workflow

#### Step 1: Exploratory Data Analysis
```bash
jupyter notebook "data exploration/eda.ipynb"
```
Explore the dataset, visualize patterns, and understand feature relationships.

#### Step 2: Feature Engineering
```bash
jupyter notebook "data exploration/partial_feature_engineering.ipynb"
```
Experiment with custom feature creation and preprocessing strategies.

#### Step 3: Train Individual Models
Navigate to the `models/` directory and run any model notebook:
- `logistic_regression.ipynb` - Linear baseline
- `decision_tree.ipynb` - Tree-based classifier
- `knn.ipynb` - Instance-based learning
- `random_forest.ipynb` - Bagging ensemble
- `xgboost.ipynb` - Gradient boosting
- `lightgbm.ipynb` - Efficient boosting

#### Step 4: Ensemble Methods
Navigate to `ensemble models/` for advanced techniques:
- Soft voting for probability averaging
- Stacking for meta-learning
- Error analysis for model selection

### Custom Feature Engineering Modules
The project includes modular feature engineering that is used throughout all models:
```python
from feature.feature_generator import FeatureGenerator
from feature.preprocessor import Preprocessor

# Initialize feature generator
generator = FeatureGenerator()
preprocessor = Preprocessor()

# Apply transformations (used in all models)
df_features = generator.create_features(train_data)
df_processed = preprocessor.preprocess(df_features)
```

**Note**: Custom features from the `feature/` module are applied during data preprocessing and used by all individual models and ensemble methods from the start.

---

## 📈 Model Performance & Evaluation

### Evaluation Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: Positive prediction reliability
- **Recall**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed error breakdown

### Optimization Techniques
- **Grid Search Cross-Validation**: Hyperparameter tuning
- **Stratified K-Fold**: Maintains class distribution in folds
- **Class Balancing**: Handles imbalanced survival rates
- **Feature Selection**: Identifies most predictive features

Performance metrics and best hyperparameters are documented in `models/observations.txt`.

---

## 🎯 Results & Key Findings

### Best Performing Models
1. **Stacking ensembles** achieved the highest accuracy by combining diverse base models
2. **XGBoost and LightGBM** showed strong individual performance
3. **Soft voting** improved robustness through probability averaging
4. **Feature engineering** increased all model accuracies by 3-5%

### Critical Features (by importance)
- Gender (Sex) - Most predictive single feature
- Passenger Class (Pclass) - Strong socioeconomic indicator
- Fare - Proxy for wealth and cabin location
- Age - Especially impactful for children
- Title - Captures social status and gender interaction
- Family Size - Non-linear relationship with survival

---

## 🔄 Project Workflow

```
1. Data Exploration → Understanding patterns and distributions
2. Feature Engineering → Creating predictive features (integrated into preprocessing)
3. Individual Models → Train all models using engineered features
4. Hyperparameter Tuning → Optimizing each model
5. Ensemble Methods → Combining models (using same features) for improved accuracy
6. Error Analysis → Understanding limitations and failure cases
7. Final Predictions → Generating submission files
```

**Note**: Custom features from the `feature/` module are used consistently across all models and ensembles.

---

## 📊 Visualizations

The project includes comprehensive visualizations:
- Survival rate distributions by demographic factors
- Correlation heatmaps for feature relationships
- Feature importance plots from tree-based models
- ROC curves for model comparison
- Confusion matrices for error analysis

---

## 📝 Key Learnings

- **Feature engineering** is more impactful than model selection for this dataset
- **Gender and passenger class** are the strongest predictors
- **Ensemble methods** consistently outperform individual models
- **Missing value imputation** strategy significantly affects results
- **Cross-validation** is essential for reliable performance estimates
- **Class imbalance** handling improves minority class detection
- **Stacking** requires careful OOF prediction to prevent overfitting

---

## 🚧 Future Enhancements

- **Neural Networks**: Deep learning approaches with TensorFlow/Keras
- **Feature Interactions**: Polynomial and interaction terms
- **Advanced Imputation**: KNN or iterative imputation for missing values
- **Text Features**: NLP on passenger names and tickets
- **Bayesian Optimization**: More sophisticated hyperparameter tuning
- **SHAP Values**: Model interpretability and feature attribution
- **Automated Feature Engineering**: Tools like Featuretools
- **Model Deployment**: Flask/FastAPI REST API for predictions

---

## 🏆 Kaggle Competition Context

This project was developed for the **Kaggle Getting Started Competition - TITANIC: MACHINE LEARNING FROM DISASTER**. It serves as an educational introduction to:
- Binary classification problems
- Feature engineering techniques
- Model evaluation and selection
- Ensemble learning methods
- Kaggle submission workflows

**Competition Link**: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)

**Repository Description**: *A beginner-friendly machine learning project using the Titanic dataset to explore data preprocessing, feature engineering, and multiple classification models in Python.*

---

## 📦 Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 👤 Author

**Aditya Singhal**  
ML/AI Enthusiast | Data Science Student

📧 Email: adityasinghal07805@gmail.com  
🔗 LinkedIn: [linkedin.com/in/aditya-singhal-0b27322ab](https://www.linkedin.com/in/aditya-singhal-0b27322ab)  
💻 GitHub: [github.com/adityaSinghal08](https://github.com/adityaSinghal08)

Feel free to connect! I'm always interested in discussing machine learning projects, data science techniques, and potential collaborations.

---

## 🌟 Acknowledgments

- **Kaggle** for hosting the Titanic competition and providing the dataset
- **Scikit-learn community** for excellent ML libraries and documentation
- **XGBoost and LightGBM teams** for powerful gradient boosting implementations
- **The Titanic dataset maintainers** for this classic ML learning resource
- **Open-source ML community** for tutorials and best practices

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Contribution Ideas:**
- Add new ensemble techniques
- Improve feature engineering modules
- Enhance visualizations
- Optimize hyperparameters
- Add model interpretability (SHAP, LIME)
- Implement deep learning models

---

## 📄 Project Status

✅ **Completed** - Successfully implemented multiple classification models with ensemble methods and comprehensive error analysis.

**Latest Update**: Enhanced ensemble techniques with stacking and soft voting (February 2026)

---

**If you find this project helpful for learning machine learning, please consider ⭐ starring the repository!**

---

*This project demonstrates fundamental and advanced machine learning techniques and is designed for educational purposes and Kaggle competition participation.*