import json
import os

HISTORY_FILE = "../history.json"

def save_history(X, y, coefficients, intercept):
    history = {
        "data": [{"x": list(x), "y": yi} for x, yi in zip(X, y)],
        "coefficients": list(coefficients),
        "intercept": intercept
    }
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)
    print("✅ История обучения сохранена.")

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return None
    with open(HISTORY_FILE, "r") as f:
        data = json.load(f)
    X = [item["x"] for item in data["data"]]
    y = [item["y"] for item in data["data"]]
    return np.array(X), np.array(y), data["coefficients"], data["intercept"]