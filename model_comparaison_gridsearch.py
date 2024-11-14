import numpy as np
import pandas as pd
import gc
import time
import mlflow
import mlflow.lightgbm
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt
import seaborn as sns

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def application_train_test(num_rows=None, nan_as_category=False):
    df = pd.read_csv('application_train.csv', nrows=num_rows)
    test_df = pd.read_csv('application_test.csv', nrows=num_rows)
    df = df._append(test_df).reset_index()

    # Nettoyage des colonnes pour éviter les caractères spéciaux et majuscules
    df.columns = df.columns.str.replace('[^A-Za-z0-9]+', '', regex=True)
    df.columns = df.columns.str.lower()

    df = df[df['codegender'] != 'XNA']
    for bin_feature in ['codegender', 'flagowncar', 'flagownrealty']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # Nettoyage après one-hot encoding
    df.columns = df.columns.str.replace('[^A-Za-z0-9]+', '', regex=True)

    df['daysemployed'].replace(365243, np.nan)
    df['daysemployedperc'] = df['daysemployed'] / df['daysbirth']
    df['incomecreditperc'] = df['amtincometotal'] / df['amtcredit']
    df['incomeperperson'] = df['amtincometotal'] / df['cntfammembers']
    df['annuityincomeperc'] = df['amtannuity'] / df['amtincometotal']
    df['paymentrate'] = df['amtannuity'] / df['amtcredit']

    del test_df
    gc.collect()

    return df

# Fonction pour calculer le coût métier (FN coûte 10 fois plus cher que FP)
def cost_function(y_true, y_pred, threshold, fn_cost=10, fp_cost=1):
    y_pred_bin = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    total_cost = fn * fn_cost + fp * fp_cost
    return total_cost

# Optimisation du seuil pour minimiser le coût métier
def optimize_threshold(y_true, y_pred, fn_cost=10, fp_cost=1):
    best_threshold = 0.5
    best_cost = float('inf')

    for threshold in np.arange(0.1, 1.0, 0.01):
        cost = cost_function(y_true, y_pred, threshold, fn_cost=fn_cost, fp_cost=fp_cost)
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return best_threshold, best_cost

# GridSearchCV pour comparer plusieurs modèles avec imputation des NaN
def compare_models_with_gridsearch(X, y):
    # Imputation des valeurs manquantes
    imputer = SimpleImputer(strategy='median')

    # Pipelines pour chaque modèle
    pipeline_log_reg = Pipeline(steps=[
        ('imputer', imputer),
        ('logistic', LogisticRegression(class_weight='balanced'))
    ])

    pipeline_rf = Pipeline(steps=[
        ('imputer', imputer),
        ('rf', RandomForestClassifier(class_weight='balanced'))
    ])

    pipeline_lgbm = Pipeline(steps=[
        ('imputer', imputer),
        ('lgbm', LGBMClassifier())
    ])

    # Modèles à comparer
    models = {
        'Logistic Regression': pipeline_log_reg,
        'Random Forest': pipeline_rf,
        'LightGBM': pipeline_lgbm
    }

    # Paramètres à tester dans GridSearchCV
    param_grid = {
        'Logistic Regression': {'logistic__C': [0.1, 1, 10]},
        'Random Forest': {'rf__n_estimators': [100, 200], 'rf__max_depth': [3, 5, 10]},
        'LightGBM': {'lgbm__num_leaves': [31, 50], 'lgbm__learning_rate': [0.01, 0.1], 'lgbm__n_estimators': [100, 200]}
    }

    best_models = {}
    mlflow.set_experiment('credit_scoring_comparison')

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid[model_name],
            scoring='roc_auc',
            cv=5,
            verbose=1
        )

        grid_search.fit(X, y)
        best_models[model_name] = grid_search.best_estimator_

        # Enregistrer le modèle et les résultats dans MLFlow
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param('best_params', grid_search.best_params_)
            mlflow.log_metric('best_auc', grid_search.best_score_)
            mlflow.sklearn.log_model(best_models[model_name], f'{model_name}_model')

        print(f"Best AUC for {model_name}: {grid_search.best_score_}")
        print(f"Best parameters: {grid_search.best_params_}")

    return best_models

# Appel des fonctions avec GridSearchCV et MLFlow
df = application_train_test()

# Séparation des données en X (features) et y (target)
train_df = df[df['target'].notnull()]
X = train_df.drop(columns=['target'])
y = train_df['target']

# Comparer plusieurs modèles et choisir le meilleur avec GridSearchCV
best_models = compare_models_with_gridsearch(X, y)

# Importance globale des features pour LightGBM
if 'LightGBM' in best_models:
    best_lgbm_model = best_models['LightGBM']
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': best_lgbm_model.named_steps['lgbm'].feature_importances_
    })

    # Visualiser l'importance des features
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=importance_df.sort_values(by="importance", ascending=False).head(20))
    plt.title("Top 20 feature importance")
    plt.show()

    # SHAP pour importance locale des features
    explainer = shap.TreeExplainer(best_lgbm_model.named_steps['lgbm'])
    shap_values = explainer.shap_values(X)
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
