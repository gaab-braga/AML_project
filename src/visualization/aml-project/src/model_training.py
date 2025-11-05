def standardize_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def create_pipeline(model_name, model_params):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    import xgboost as xgb
    import lightgbm as lgb

    if model_name == 'xgb':
        classifier = xgb.XGBClassifier(**model_params)
    elif model_name == 'lgbm':
        classifier = lgb.LGBMClassifier(**model_params)
    else:
        raise ValueError(f"Modelo '{model_name}' não suportado.")

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler()),
        ('classifier', classifier)
    ])
    return pipeline

def train_model(X_train, y_train, model_name, model_params):
    pipeline = create_pipeline(model_name, model_params)
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
        'f1_score': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }

    return metrics