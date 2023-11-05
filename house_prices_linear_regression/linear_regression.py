import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from pre_process import pre_process

# Load train.csv
data_train = pd.read_csv('data\\train.csv', index_col='Id')
data_test = pd.read_csv('data\\test.csv', index_col='Id')

# Pre-process train data
pp_data_train = pre_process(data_train)

X = pp_data_train.drop('SalePrice', axis=1)
y = data_train['SalePrice']

# Train model
model = LinearRegression()
model.fit(X, y)

# Make predictions on train data
pred = model.predict(X)

# Evaluate predictions
mae = mean_absolute_error(y, pred)  # Mean Absolute Error
mse = mean_squared_error(y, pred)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
mape = mean_absolute_percentage_error(y, pred)  # Mean Absolute Percentage Error
r2 = r2_score(y, pred)  # R-squared

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAPE: {mape}')
print(f'R-squared: {r2}')

# Make plot
plt.scatter(y, pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Train Data - Actual vs Predicted")
plt.show()

# Pre-process test data
pp_data_test = pre_process(data_test)

# Make predictions on test data
pred = model.predict(pp_data_test)

pred_df = pd.DataFrame(pred, index=pp_data_test.index)
pred_df.columns = ['SalePrice']

print('\n\n')
print(pred_df)
