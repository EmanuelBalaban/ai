import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

actual = pd.read_csv('data/actual.csv')
data_train = pd.read_csv('data/data_set_ALL_AML_train.csv')
data_independent = pd.read_csv('data/data_set_ALL_AML_independent.csv')

# Drop alphanumeric columns
pp_data_train = data_train.drop(['Gene Description', 'Gene Accession Number'], axis=1)

# Drop call columns
call_cols = [col for col in pp_data_train.columns if "call" not in col]
pp_data_train = pp_data_train[call_cols]

# Transpose data_train
pp_data_train = pp_data_train.T

print(pp_data_train.head())

# Add cancer classification column
patients_count = len(pp_data_train.index)
pp_data_train['cancer'] = list(actual[:patients_count]['cancer'])

print(pp_data_train.head())

pp_data_train.replace({'ALL': 0, 'AML': 1}, inplace=True)

print(pp_data_train)

# Preprocess independent data
pp_data_independent = data_independent.drop(['Gene Description', 'Gene Accession Number'], axis=1)
call_cols_independent = [col for col in pp_data_independent.columns if "call" not in col]
pp_data_independent = pp_data_independent[call_cols_independent].T

# Standardize data
X_train = StandardScaler().fit_transform(pp_data_train.drop('cancer', axis=1))
X_independent = StandardScaler().fit_transform(pp_data_independent)

# Apply PCA on training data
pca_train = PCA()
Y_train = pca_train.fit_transform(X_train)

print(Y_train)

# Apply PCA on independent data
Y_independent = pca_train.transform(X_independent)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Y_train, pp_data_train['cancer'], test_size=0.2, random_state=42)

# Create a logistic regression classifier
classifier = LogisticRegression(random_state=42)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy with all components: {accuracy}')

# Now, let's iterate over a range of components
for n_components in range(1, min(X_train.shape[0], X_train.shape[1])):
    # Apply PCA with a reduced number of components
    pca = PCA(n_components=n_components)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)

    # Train the classifier on the reduced data
    classifier.fit(X_train_reduced, y_train)

    # Make predictions on the test set
    y_pred_reduced = classifier.predict(X_test_reduced)

    # Evaluate the performance
    accuracy = accuracy_score(y_test, y_pred_reduced)
    print(f'Accuracy with {n_components} components: {accuracy}')

    # Make predictions on the independent dataset
    # predictions_independent = classifier.predict(Y_independent)

    # Print the results
    # print(f'Accuracy on the independent dataset: {accuracy_score(actual["cancer"].map({"ALL": 0, "AML": 1}), predictions_independent)}')

