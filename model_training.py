import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
import joblib
import os
def get_heart_disease_data():
    import warnings
    warnings.filterwarnings('ignore')

def load_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol',
-37
+240
    ]
    try:
        df = pd.read_csv(url, names=column_names, na_values='?')
        print("Downloaded UCI Heart Disease dataset from web.")
    except Exception:
        print("Could not download from web, using built-in sample data.")
        df = generate_sample_data()
        print(f"[INFO] Dataset loaded from UCI repository. Shape: {df.shape}")
    except Exception as e:
        print(f"[WARN] Could not download dataset: {e}. Using fallback synthetic data.")
        df = _generate_synthetic_data()
    return df
def generate_sample_data():
    pass
def _generate_synthetic_data(n=303):
    np.random.seed(42)
    n = 303
    data = {
        'age': np.random.randint(29, 77, n),
        'sex': np.random.randint(0, 2, n),
        'cp': np.random.randint(0, 4, n),
        'trestbps': np.random.randint(94, 200, n),
        'chol': np.random.randint(126, 564, n),
        'fbs': np.random.randint(0, 2, n),
        'restecg': np.random.randint(0, 3, n),
        'thalach': np.random.randint(71, 202, n),
        'exang': np.random.randint(0, 2, n),
        'oldpeak': np.round(np.random.uniform(0, 6.2, n), 1),
        'slope': np.random.randint(0, 3, n),
        'ca': np.random.randint(0, 4, n),
        'thal': np.random.choice([3, 6, 7], n),
        'target': np.random.randint(0, 2, n),
        'sex':      np.random.randint(0, 2, n).astype(float),
        'cp':       np.random.randint(0, 4, n).astype(float),
        'trestbps': np.random.randint(94, 200, n).astype(float),
        'chol':     np.random.randint(126, 564, n).astype(float),
        'fbs':      np.random.randint(0, 2, n).astype(float),
        'restecg':  np.random.randint(0, 3, n).astype(float),
        'thalach':  np.random.randint(71, 202, n).astype(float),
        'exang':    np.random.randint(0, 2, n).astype(float),
        'oldpeak':  np.round(np.random.uniform(0, 6.2, n), 1),
        'slope':    np.random.randint(0, 3, n).astype(float),
        'ca':       np.random.randint(0, 4, n).astype(float),
        'thal':     np.random.choice([3.0, 6.0, 7.0], n),
        'target':   np.random.randint(0, 2, n)
    }
    return pd.DataFrame(data)
def preprocess_data(df):
    df = df.dropna()

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
def preprocess(df):
    print("\n[STEP] Preprocessing...")
    
    missing = df.isnull().sum()
    if missing.any():
        print(f"  Missing values detected:\n{missing[missing > 0]}")
        
        for col in df.columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  Filled '{col}' missing values with median={median_val:.2f}")
    else:
        print("  No missing values found.")
    
    df['target'] = (df['target'] > 0).astype(int)
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
    X = df[features]
    y = df['target']
    X = df[FEATURES].copy()
    y = df['target'].copy()
    print(f"  Samples: {len(y)} | No Disease: {(y==0).sum()} | Heart Disease: {(y==1).sum()}")
    return X, y
def train_models():
    print("Loading heart disease dataset...")
    df = get_heart_disease_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    X, y = preprocess_data(df)
    print(f"\nFeatures: {list(X.columns)}")
    print(f"Target distribution:\n{y.value_counts()}")

def get_model_configs():
    return {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['lbfgs', 'liblinear'],
                'penalty': ['l2']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt', 'log2']
            }
        },
        'Support Vector Machine': {
            'model': SVC(random_state=42, class_weight='balanced', probability=True),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
    }

def train_and_evaluate(X_train, X_test, y_train, y_test, scaler):
    configs = get_model_configs()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    results = {}
    for name, cfg in configs.items():
        print(f"\n{'='*50}")
        print(f"[MODEL] {name}")
        print(f"{'='*50}")
        
        grid_search = GridSearchCV(
            estimator=cfg['model'],
            param_grid=cfg['params'],
            scoring='f1',
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        print(f"  Best params : {grid_search.best_params_}")
        print(f"  Best CV F1  : {grid_search.best_score_:.4f}")
        
        cv_acc = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        cv_f1  = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring='f1')
        print(f"  CV Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
        print(f"  CV F1 Score : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob)
        cm   = confusion_matrix(y_test, y_pred).tolist()
        print(f"\n  ── Test Set Metrics ──")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}  (of predicted disease, how many truly have it)")
        print(f"  Recall    : {rec:.4f}  (of actual disease cases, how many detected)")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Disease', 'Heart Disease']))
        print(f"  Confusion Matrix:\n  {np.array(cm)}")
        results[name] = {
            'model': best_model,
            'best_params': str(grid_search.best_params_),
            'cv_accuracy_mean': float(cv_acc.mean()),
            'cv_accuracy_std':  float(cv_acc.std()),
            'cv_f1_mean':       float(cv_f1.mean()),
            'cv_f1_std':        float(cv_f1.std()),
            'test_accuracy':    float(acc),
            'test_precision':   float(prec),
            'test_recall':      float(rec),
            'test_f1':          float(f1),
            'test_auc':         float(auc),
            'confusion_matrix': cm
        }
    return results, X_train_scaled, X_test_scaled

def select_best_model(results):
    """Select best model using a composite score (F1 weighted more, AUC secondary)."""
    scored = {}
    for name, r in results.items():
       
        composite = 0.60 * r['test_f1'] + 0.25 * r['test_auc'] + 0.15 * r['test_recall']
        scored[name] = composite
    best_name = max(scored, key=scored.get)
    return best_name, scored

def save_artefacts(best_name, results, scaler):
    os.makedirs('python_backend/model', exist_ok=True)
    best_model = results[best_name]['model']
    joblib.dump(best_model, 'python_backend/model/heart_disease_model.pkl')
    joblib.dump(scaler,     'python_backend/model/scaler.pkl')
    joblib.dump(FEATURES,   'python_backend/model/feature_names.pkl')
    
    comparison = []
    for name, r in results.items():
        comparison.append({
            'name':         name,
            'best_params':  r['best_params'],
            'cv_accuracy':  f"{r['cv_accuracy_mean']:.4f} ± {r['cv_accuracy_std']:.4f}",
            'test_accuracy': round(r['test_accuracy'], 4),
            'test_precision': round(r['test_precision'], 4),
            'test_recall':    round(r['test_recall'], 4),
            'test_f1':        round(r['test_f1'], 4),
            'test_auc':       round(r['test_auc'], 4),
            'confusion_matrix': r['confusion_matrix']
        })
    model_info = {
        'best_model':     best_name,
        'best_accuracy':  results[best_name]['test_accuracy'],
        'best_f1':        results[best_name]['test_f1'],
        'best_auc':       results[best_name]['test_auc'],
        'best_precision': results[best_name]['test_precision'],
        'best_recall':    results[best_name]['test_recall'],
        'comparison':     comparison
    }
    joblib.dump(model_info, 'python_backend/model/model_info.pkl')
    
    with open('python_backend/model/model_report.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"\n[SAVED] All artefacts written to python_backend/model/")
    return model_info

def train_pipeline():
    print("\n" + "="*60)
    print("  Heart Disease Prediction — ML Training Pipeline")
    print("="*60)
    
    df = load_dataset()
    X, y = preprocess(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
