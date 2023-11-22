import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

actual = pd.read_csv('data/actual.csv')
data_train = pd.read_csv('data/data_set_ALL_AML_train.csv')

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

X = StandardScaler().fit_transform(pp_data_train.drop('cancer', axis=1))

pca = PCA()
Y = pca.fit_transform(X)

print(Y)


