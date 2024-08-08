import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, confusion_matrix  
from sklearn.feature_selection import SelectKBest, f_regression  
from sklearn.preprocessing import StandardScaler  
from sklearn.datasets import fetch_california_housing  
import matplotlib.pyplot as plt  
  
# Load the California Housing dataset  
california = fetch_california_housing()  
  
# Convert the dataset into a pandas DataFrame  
df = pd.DataFrame(california.data, columns=california.feature_names)  
df['MEDV'] = california.target  
  
# Split the dataset into features (X) and target (y)  
X = df.drop('MEDV', axis=1)  
y = df['MEDV']  
  
# Split the dataset into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
  
# Create a linear regression model  
model = LinearRegression()  
  
# Train the model on the entire dataset  
model.fit(X_train, y_train)  
  
# Predict the target values for the test set  
y_pred = model.predict(X_test)  
  
# Evaluate the model using Mean Squared Error (MSE)  
mse = mean_squared_error(y_test, y_pred)  
print(f'Mean Squared Error (without feature selection or scaling): {mse:.2f}')  
  
# Convert the predicted values to binary classes (for confusion matrix)  
y_pred_class = np.where(y_pred > np.median(y_test), 1, 0)  
y_test_class = np.where(y_test > np.median(y_test), 1, 0)  
  
# Calculate the confusion matrix  
conf_mat = confusion_matrix(y_test_class, y_pred_class)  
print("\nConfusion Matrix (without feature selection or scaling):")  
print(conf_mat,"\n")  
  
# Feature selection using SelectKBest  
selector = SelectKBest(f_regression, k=5)  
X_train_selected = selector.fit_transform(X_train, y_train)  
X_test_selected = selector.transform(X_test)  
  
# Train the model on the selected features  
model.fit(X_train_selected, y_train)  
  
# Predict the target values for the test set with feature selection  
y_pred_selected = model.predict(X_test_selected)  
  
# Evaluate the model with feature selection using MSE  
mse_selected = mean_squared_error(y_test, y_pred_selected)  
print(f'Mean Squared Error (with feature selection): {mse_selected:.2f}')  
  
# Convert the predicted values to binary classes (for confusion matrix)  
y_pred_class_selected = np.where(y_pred_selected > np.median(y_test), 1, 0)  
  
# Calculate the confusion matrix  
conf_mat_selected = confusion_matrix(y_test_class, y_pred_class_selected)  
print("\nConfusion Matrix (with feature selection):")  
print(conf_mat_selected,"\n")  
  
# Feature scaling using StandardScaler  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  
  
# Train the model on the scaled features  
model.fit(X_train_scaled, y_train)  
  
# Predict the target values for the test set with feature scaling  
y_pred_scaled = model.predict(X_test_scaled)  
  
# Evaluate the model with feature scaling using MSE  
mse_scaled = mean_squared_error(y_test, y_pred_scaled)  
print(f'Mean Squared Error (with feature scaling): {mse_scaled:.2f}')  
  
# Convert the predicted values to binary classes (for confusion matrix)  
y_pred_class_scaled = np.where(y_pred_scaled > np.median(y_test), 1, 0)  
  
# Calculate the confusion matrix  
conf_mat_scaled = confusion_matrix(y_test_class, y_pred_class_scaled)  
print("\nConfusion Matrix (with feature scaling):")  
print(conf_mat_scaled,"\n")  
  
# Plot the confusion matrices  

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))  
ax1.imshow(conf_mat, cmap='Blues')  
ax1.set_title('Confusion Matrix (without feature selection or scaling)')  
ax1.set_xlabel('Predicted labels')  
ax1.set_ylabel('True labels')  
  
ax2.imshow(conf_mat_selected, cmap='Blues')  
ax2.set_title('Confusion Matrix (with feature selection)')  
ax2.set_xlabel('Predicted labels')  
ax2.set_ylabel('True labels')  
  
ax3.imshow(conf_mat_scaled, cmap='Blues')  
ax3.set_title('Confusion Matrix (with feature scaling)')  
ax3.set_xlabel('Predicted labels')  
ax3.set_ylabel('True labels') 
  
plt.show()