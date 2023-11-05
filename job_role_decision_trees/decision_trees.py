import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# Load data
dataset = pd.read_csv('data\\HR-Em.csv', index_col='EmployeeNumber')

# Pre-process data
dataset['JobRole'] = dataset['JobRole'].astype('category')
dataset['JobRoleCode'] = dataset['JobRole'].cat.codes

# Save job_roles
job_roles = np.array(dataset['JobRole'].cat.categories)

print('Job Roles: ', job_roles)
print('Job Roles Count: ', len(job_roles))
print('')

pp_data: pd.DataFrame = dataset.drop([
    # 'Age',
    'Attrition',
    'BusinessTravel',
    'DailyRate',
    # 'Department',
    'DistanceFromHome',
    # 'Education',
    # 'EducationField',
    'EmployeeCount',
    # 'EnvironmentSatisfaction',
    # 'Gender',
    'HourlyRate',
    # 'JobInvolvement',
    # 'JobLevel',
    'JobSatisfaction',
    'MaritalStatus',
    # 'MonthlyIncome',
    'MonthlyRate',
    # 'NumCompaniesWorked',
    'Over18',
    'OverTime',
    # 'PercentSalaryHike',
    'PerformanceRating',
    'RelationshipSatisfaction',
    'StandardHours',
    'StockOptionLevel',
    # 'TotalWorkingYears',
    # 'TrainingTimesLastYear',
    'WorkLifeBalance',
    # 'YearsAtCompany',
    'YearsInCurrentRole',
    # 'YearsSinceLastPromotion',
    # 'YearsWithCurrManager',
], axis=1)

# Drop last row since is NaN
# pp_data.drop(pp_data.tail(1).index, inplace=True)

X = pp_data.drop(['JobRole', 'JobRoleCode'], axis=1)
y = pp_data['JobRoleCode']

# One hot encoding for objects
# X = pd.get_dummies(X, dtype=float)

X['Department'] = X['Department'].astype('category')
X['Department'] = X['Department'].cat.codes

X['EducationField'] = X['EducationField'].astype('category')
X['EducationField'] = X['EducationField'].cat.codes

X['Gender'] = X['Gender'].astype('category')
X['Gender'] = X['Gender'].cat.codes

# Split data into train and test -> 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Make decision tree
decision_tree_model = DecisionTreeClassifier(
    # max_depth=6,
    criterion='entropy',
    # min_samples_leaf=4,
    # ccp_alpha=0.001,
    random_state=42)
decision_tree_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = decision_tree_model.predict(X_test)

# Evaluate predictions
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared
accuracy = accuracy_score(y_test, y_pred) * 100

print(f'Mean Absolute Error: {mae}'
      " - On average the model's predictions are off by about "
      f'{"{:.2f}".format(mae)} '
      'units from the actual values - (LOWER is better)')
print(f'Mean Squared Error: {mse}'
      ' - Average squared differences between predictions and actual values - (LOWER is better)')
print(f'Root Mean Squared Error: {rmse}'
      ' - square root of MSE - (LOWER is better)')
print(f'R-squared: {r2}'
      ' - [ -∞ (worse performance) - 0.0 (no fit) - 1.0 (perfect fit) ]')
print(f'Accuracy: {"{:.2f}".format(accuracy)}%'
      ' - How accurate are the predictions - (HIGHER is better)')

# Make plot for Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

# Make plot for Feature Importance
feature_importance = decision_tree_model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(25, 20))
plt.barh(feature_names, feature_importance)
plt.yticks(fontsize=10)
plt.xlabel('Importance Factor', fontsize=24)
plt.title('Feature Importance', fontsize=48, fontweight='bold')
plt.show()

# Make plot for Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
seaborn.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Make plot for tree
print('\nGenerating decision tree image...')
plt.figure(figsize=(260, 120))
plot_tree(decision_tree_model,
          feature_names=X.columns,
          class_names=job_roles,
          filled=True)
plt.savefig("decision_tree.png")

# Display tree
print('\n' + export_text(decision_tree_model, feature_names=X.columns, class_names=job_roles))

# Make predictions for input.csv
dataset = pd.read_csv('data\\input.csv', index_col='EmployeeNumber')

dataset = dataset.drop([
    # 'Age',
    'Attrition',
    'BusinessTravel',
    'DailyRate',
    # 'Department',
    'DistanceFromHome',
    # 'Education',
    # 'EducationField',
    'EmployeeCount',
    # 'EnvironmentSatisfaction',
    # 'Gender',
    'HourlyRate',
    # 'JobInvolvement',
    # 'JobLevel',
    'JobSatisfaction',
    'MaritalStatus',
    # 'MonthlyIncome',
    'MonthlyRate',
    # 'NumCompaniesWorked',
    'Over18',
    'OverTime',
    # 'PercentSalaryHike',
    'PerformanceRating',
    'RelationshipSatisfaction',
    'StandardHours',
    'StockOptionLevel',
    # 'TotalWorkingYears',
    # 'TrainingTimesLastYear',
    'WorkLifeBalance',
    # 'YearsAtCompany',
    'YearsInCurrentRole',
    # 'YearsSinceLastPromotion',
    # 'YearsWithCurrManager',
], axis=1)

X = dataset

# One hot encoding for objects
# X = pd.get_dummies(X, dtype=float)

X['Department'] = X['Department'].astype('category')
X['Department'] = X['Department'].cat.codes

X['EducationField'] = X['EducationField'].astype('category')
X['EducationField'] = X['EducationField'].cat.codes

X['Gender'] = X['Gender'].astype('category')
X['Gender'] = X['Gender'].cat.codes

pred = decision_tree_model.predict(X)

print(('Prediction for input: ' + job_roles[pred])[0])
