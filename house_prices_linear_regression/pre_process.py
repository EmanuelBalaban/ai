# Pre-process data by dropping unneeded columns and by getting rid of NA values
def pre_process(data):
    data = data.drop(['MiscFeature', 'PoolQC', 'Fence', 'Alley'], axis=1)

    # Fill all NA values with median value of column
    num_df = data.select_dtypes(include='number')
    num_cols = num_df.columns
    for col in num_cols:
        data[col] = data[col].fillna(data[col].median())

    # Set all strings to 0
    str_df = data.select_dtypes(include='object')
    str_cols = str_df.columns
    for col in str_cols:
        data[col] = 0

    return data
