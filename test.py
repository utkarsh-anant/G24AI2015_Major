"""
test.py
Loads savedmodel.pth and evaluates on the holdout set, printing accuracy.
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import sys

def main():
    # Load dataset
    data = fetch_olivetti_faces()
    X, y = data.data, data.target

    # Same split as training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Load model
    try:
        model = joblib.load("savedmodel.pth")
    except FileNotFoundError:
        print("ERROR: savedmodel.pth not found. Run train.py first.", file=sys.stderr)
        sys.exit(2)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Test Accuracy:", acc)

if __name__ == "__main__":
    main()
