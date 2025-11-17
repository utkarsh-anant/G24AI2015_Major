"""
train.py
Trains a DecisionTreeClassifier on the Olivetti faces dataset and writes a model file savedmodel.pth
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

def main():
    # Load dataset
    data = fetch_olivetti_faces()
    X, y = data.data, data.target

    # Train-test split (70-30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "savedmodel.pth")
    print("Model trained and saved as savedmodel.pth")

if __name__ == "__main__":
    main()
