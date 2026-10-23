from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, preprocessor, config):
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=config["n_estimators"],
            random_state=config["random_state"]
        ))
    ])

    model.fit(X_train, y_train)
    print("Model training completed")
    return model
