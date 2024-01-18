import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

# Load training data
train_data = pd.read_csv('data\\train_motion_data.csv')

# Save class categories in an array
classes = {'SLOW': 0, 'NORMAL': 1, 'AGGRESSIVE': 2}

# Map class names to numerical values using the classes dictionary
train_data['ClassCode'] = train_data['Class'].map(classes)

print('Classes: ', classes)
print(train_data[['Class', 'ClassCode']])

# Feature engineering
# For simplicity, only Accelerometer and Gyroscope data are used in the model.
features = train_data[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']]
labels = train_data['ClassCode']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train model (Random Forest)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on test set (split from train data)
train_pred = clf.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, train_pred)
print(f'Model Accuracy (Random Forest): {accuracy:.2f}')

# Load test data
test_data = pd.read_csv('data\\test.csv')

# Feature engineering (for test data)
test_features = test_data[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']]

# Make predictions
predictions = clf.predict(test_features)

# Convert category codes back to labels (classes)
predicted_labels = pd.Series(predictions).apply(
    lambda x: next(class_name for class_name, class_code in classes.items() if class_code == x))

# Export dataframe
result_df = pd.DataFrame({'ID': test_data['ID'], 'Class': predicted_labels})
result_df.to_csv('data\\rf_sample_results.csv', index=False)

# Train model (Logistic Regression)
logreg_clf = LogisticRegression(max_iter=1000, random_state=42)
logreg_clf.fit(X_train, y_train)

# Make predictions (on train data)
logreg_train_pred = logreg_clf.predict(X_test)

# Model performance
accuracy_logreg = accuracy_score(y_test, logreg_train_pred)
print(f'Model Accuracy (Logistic Regression): {accuracy_logreg:.2f}')

r_squared = r2_score(y_test, logreg_train_pred)
print(f'R-squared (Logistic Regression): {r_squared:.2f}'
      ' - [ -∞ (worse performance) - 0.0 (no fit) - 1.0 (perfect fit) ]')

# Make predictions
logreg_predictions = logreg_clf.predict(test_features)

# Convert category codes back to labels (classes)
logreg_predicted_labels = pd.Series(logreg_predictions).apply(
    lambda x: next(class_name for class_name, class_code in classes.items() if class_code == x))

# Export dataframe
result_df = pd.DataFrame({'ID': test_data['ID'], 'Class': logreg_predicted_labels})
result_df.to_csv('data\\logreg_sample_results.csv', index=False)
