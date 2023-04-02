import pandas as pd
import numpy as np
from IPython.core.display_functions import display

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
    uniq_cnt = df[i].nunique()  # Use df[i] instead of cols[c]
    # print(i, uniq_cnt)
    var_dist_cnts[i] = uniq_cnt  # Use i instead of c


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

list_of_continuous = ['carat_weight','depth_percent','table_percent']

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

print("df2 columns:\n", df2.columns)
print("df2 describe:\n", df2.describe().T)






