import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# [The rest of the code remains the same until the model creation part]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)

# Make predictions
y_pred = model.predict(X_test_scaled).flatten()

# [The rest of the code remains the same]


# Create a new DataFrame to store the predictions
predictions_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})

predictions_df["deciles"] = pd.cut(predictions_df["y_pred"], bins=20)

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

decile_errors = predictions_df.groupby('deciles')['abs_error'].mean().reset_index()

print(decile_errors)

plt.figure(figsize=(12, 6))
plt.bar(decile_errors.index, decile_errors['abs_error'])
plt.xticks(decile_errors.index, decile_errors['deciles'], rotation='vertical')
plt.xlabel('Deciles')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error by Deciles')

# Save the plot to your computer as a PNG file
plt.savefig('mean_absolute_error_by_deciles.png', dpi=300, bbox_inches='tight')

plt.show()

average_target = df2['total_sales_price'].mean()

print("average target mean ", average_target)



# Plot the true values vs. the predicted values
# plt.scatter(predictions_df['y_true'], predictions_df['y_pred'], alpha=0.5)
# plt.xlabel("True Values")
# plt.ylabel("Predicted Values")
# plt.title("True Values vs. Predicted Values")
# plt.show()



# Calculate the mean squared error of the predictions
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error: ", mse)





