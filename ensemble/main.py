import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Load the training data
train_data = pd.read_csv("data\\mnist_train.csv")

# Split the data into features (X) and labels (y)
X_train = train_data.drop("label", axis=1)
y_train = train_data["label"]

# Load the test data
test_data = pd.read_csv("data\\mnist_test.csv")

# Split the test data into features (X_test) and labels (y_test)
X_test = test_data.drop("label", axis=1)
y_test = test_data["label"]

# Initialize individual classifiers
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
et_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)

# Create an ensemble of classifiers using majority voting
ensemble_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('gb', gb_clf), ('et', et_clf)],
    voting='hard'
)

classifiers = {'Random Forest': rf_clf, 'Gradient Boosting': gb_clf, 'Extra Trees': et_clf, 'Ensemble': ensemble_clf}

# Compare classifiers
for name, clf in classifiers.items():
    print(f"Training {name} model...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"Predicting using {name} model...")
    start_time = time.time()
    y_pred = clf.predict(X_test)
    test_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} - Accuracy: {accuracy * 100:.2f}%, Training Time: {train_time:.2f}s, Test Time: {test_time:.2f}s")
    print("----------------------------------------------------")
