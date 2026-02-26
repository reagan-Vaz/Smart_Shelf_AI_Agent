import joblib

MODEL_PATH = "models/demand_model.pkl"


def recommend_discount(predicted_demand):
    if predicted_demand < 50:
        return 0.25
    elif predicted_demand < 100:
        return 0.15
    else:
        return 0.05


def run_agent(sample_input):
    model = joblib.load(MODEL_PATH)

    predicted_demand = model.predict(sample_input)[0]

    discount = recommend_discount(predicted_demand)

    return predicted_demand, discount