import joblib

def predict_new_data(model_path, new_data):
    model = joblib.load(model_path)
    return model.predict(new_data)
