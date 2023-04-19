import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor  # Import XGBRegressor
from sklearn.metrics import mean_squared_error

np.set_printoptions(precision=3)
np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)

df = pd.read_csv('diamonds.csv', index_col=[0])

cols = df.columns
col_tolist = cols.tolist()

# print("Head of csv: ", df.head())
# print("Df describe: ", df.describe())

# counting unique values for all rows
c = df.columns
var_dist_cnts = {}
for i in c:
    uniq_cnt = df[i].nunique()
    var_dist_cnts[i] = uniq_cnt


# print("Printing distinct counts for variables : ", var_dist_cnts)

# columns = df.columns
var_dist_cnts = {}

# Better looking print for distinct counts

# for col_name in cols:
#     unique_count = df[col_name].value_counts().count()
#     var_dist_cnts[col_name] = unique_count
#     print(col_name, ' : ', unique_count)

# creating lists for target variable, category variables, and count variables

target = ['total_sales_price']

list_of_categories = ['cut', 'color', 'cut_quality', 'lab', 'symmetry', 'polish', 'eye_clean',
             'culet_size', 'culet_condition']

# for a in list_of_categories:
#     plt.hist(df[a])
#     plt.title("Histogram of " + str(a))
#     plt.xlabel(a)
#     plt.ylabel('Frequency')
#     plt.show()


list_of_continuous = ['carat_weight', 'depth_percent', 'table_percent']

for b in list_of_continuous:
    plt.hist(df[b])
    plt.title("Histogram of " + str(b))
    plt.xlabel(b)
    plt.ylabel('Frequency')
    plt.show()

pd.set_option('display.float_format', lambda x: '%.5f' % x)

# print("Shape of df: ", df.shape)

# print("Describe of target variable --sales: ", df[target].describe())
#
# print("Printing uniques for categoricals to verify:\n ", df[list_of_categories].nunique())
# print("Printing uniques for continuous to verify:\n ", df[list_of_continuous].nunique())
#
# print("Printing dummy variables for all categories -----------")
# for j in list_of_categories:
#     display(pd.get_dummies(df[j]))

# Create new dataframe with all dummy variables for all categories


df2 = df[target]

for a in list_of_categories:
    t = pd.get_dummies(df[a])
    t = t.add_prefix(a + "_")
    df2 = pd.concat([df2, t], axis=1)
    # print(df2.shape)

# print("df2 columns:\n", df2.columns)
# print("df2 describe:\n", df2.describe().T)


corrDf = df2.corr()

print("This is the correlation dataframe: \n", corrDf['total_sales_price'].sort_values())

# df2.to_csv('dummyVars.csv', sep="\t")

# Prepare the dataset
X = df2.drop('total_sales_price', axis=1)
y = df2['total_sales_price']

# Initialize a 10-fold cross-validator
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize the XGBRegressor
xgb_regressor = XGBRegressor()

# Initialize a list to store validation scores
validation_scores = []

# Loop through each fold
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train the XGBRegressor on the training data
    xgb_regressor.fit(X_train, y_train)

    # Make predictions on the validation data
    y_val_pred = xgb_regressor.predict(X_val)

    # Calculate the mean squared error for the current fold
    mse = mean_squared_error(y_val, y_val_pred)

    # Append the validation score to the list
    validation_scores.append(mse)

# Calculate the average validation score
average_validation_score = np.mean(validation_scores)
print("Average Mean Squared Error: ", average_validation_score)


