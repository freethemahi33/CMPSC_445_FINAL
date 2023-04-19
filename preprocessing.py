import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor  # or GradientBoostingClassifier, based on your problem
from sklearn.metrics import r2_score

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

df['unknown_count'] = df.apply(lambda row: (row == 'unknown').sum(), axis=1)
df = df[df['unknown_count'] <= 6]


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

# for b in list_of_continuous:
#     plt.hist(df[b])
#     plt.title("Histogram of " + str(b))
#     plt.xlabel(b)
#     plt.ylabel('Frequency')
#     plt.show()

pd.set_option('display.float_format', lambda x: '%.5f' % x)

df2 = df[target]

for a in list_of_categories:
    t = pd.get_dummies(df[a])
    t = t.add_prefix(a + "_")
    df2 = pd.concat([df2, t], axis=1)
    # print(df2.shape)

corrDf = df2.corr()

# print("This is the correlation dataframe: \n", corrDf['total_sales_price'].sort_values())



import matplotlib.pyplot as plt

# Prepare the dataset
X = df2.drop('total_sales_price', axis=1)
y = df2['total_sales_price']

# Split the dataset into training and testing sets
# training is 20% and test size is 20% with a random number seed of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# display("Describe of x train: ", X_train.describe())

# Create a GradientBoostingRegressor model
model = GradientBoostingRegressor(
    n_estimators=200,   # Number of boosting stages to perform
    learning_rate=0.1,  # Shrinks the contribution of each tree by the learning_rate
    max_depth=3,        # Maximum depth of the individual regression estimators
    min_samples_split=2,  # The minimum number of samples required to split an internal node
    random_state=42     # Seed of the pseudo-random number generator used when shuffling the data
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

y_pred_train = model.predict(X_train)

display(y_pred_train, df2['total_sales_price'])

# Create a new DataFrame to store the predictions
predictions_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})

predictions_df["deciles"] = pd.qcut(predictions_df["y_pred"], q=20, labels=False, duplicates='drop')

df2["y_pred"] = predictions_df["y_pred"]

df2 = df2.sort_values(by="y_pred")

df2["ranking"] = np.arange(len(df2))

print(df2)
# print(predictions_df["deciles"].value_counts())



# print("This is the prediction correlation: \n", predictions_df.corr())

# If you want to see the result, print the predictions_df
# print(predictions_df)



r2 = r2_score(y_test, y_pred)

print("R-squared Score: ", r2)

predictions_df['abs_error'] = abs(predictions_df['y_true'] - predictions_df['y_pred'])

print("Mean of y true: ", predictions_df['y_true'].mean())

print("Mean of y pred: ", predictions_df['y_pred'].mean())

print("Describe of y true and y pred:\n")

print(predictions_df[['y_true', 'y_pred']].describe())

print("head and tail: ")

print(predictions_df[['y_true', 'y_pred']].sort_values(by=['y_pred']).head())
print(predictions_df[['y_true', 'y_pred']].sort_values(by=['y_pred']).tail())

print("head and tail of abs error: ")

print(predictions_df.sort_values(by=['abs_error']).head())
print(predictions_df.sort_values(by=['abs_error']).tail())

print(predictions_df['deciles'].value_counts())

predictions_df['total_sales_price'] = df2['total_sales_price']

decile_averages = predictions_df.groupby('deciles')[['y_pred', 'total_sales_price']].agg(['mean', 'size']).reset_index()
print("Decile averages: \n", decile_averages)



# print(decile_errors)

# plt.figure(figsize=(12, 6))
# plt.bar(decile_errors.index, decile_errors['abs_error'])
# plt.xticks(decile_errors.index, decile_errors['deciles'], rotation='vertical')
# plt.xlabel('Deciles')
# plt.ylabel('Mean Absolute Error')
# plt.title('Mean Absolute Error by Deciles')
#
# # Save the plot to your computer as a PNG file
# plt.savefig('mean_absolute_error_by_deciles.png', dpi=300, bbox_inches='tight')
#
# plt.show()

# Plot the true values vs. the predicted values
# plt.scatter(predictions_df['y_true'], predictions_df['y_pred'], alpha=0.5)
# plt.xlabel("True Values")
# plt.ylabel("Predicted Values")
# plt.title("True Values vs. Predicted Values")
# plt.show()



# Calculate the mean squared error of the predictions
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error: ", mse)
#%%
predictions_df['y_pred'].hist()

#%%
display(y_pred_train)

print("------------")

display(df2['total_sales_price'].head().reset_index())

predictions_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})

# Take the sum of the unkown variables. If variable is unkown more often than not, remove


#%%
# df['unknown_count'] = df.apply(lambda row: (row == 'unknown').sum(), axis=1)
#
# # Sort the DataFrame based on 'unknown_count' column in descending order
# sorted_df = df.sort_values(by='unknown_count', ascending=False)
#
# # Display the sorted DataFrame
# display(sorted_df)
#
# # Filter rows with 'unknown_count' less than or equal to 6
# filtered_df = df[df['unknown_count'] <= 6]
#
# display(df)
# # Display the filtered DataFrame
# display(filtered_df.sort_values(by='unknown_count', ascending=False))