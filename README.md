# California_Housing_Price_Prediction

The code will run in vscode or pycharm

# Explanation:

## **Data Loading and Preparation**

The script starts by importing necessary libraries, including NumPy, Pandas, Scikit-learn, and Matplotlib.
The California Housing dataset is loaded using fetch_california_housing() from Scikit-learn.
The dataset is converted into a Pandas DataFrame, and the target variable MEDV is added as a column.

## **Data Splitting**

The dataset is split into features (X) and target (y) variables.
The dataset is further split into training and testing sets using train_test_split() with a test size of 0.2 and a random state of 42.

## **Linear Regression Model**

A linear regression model is created using LinearRegression() from Scikit-learn.
The model is trained on the entire dataset using fit().

## **Model Evaluation (without Feature Selection or Scaling)**

The model is evaluated using Mean Squared Error (MSE) between the predicted and actual target values.
The predicted values are converted to binary classes (0 or 1) based on the median of the target values.
A confusion matrix is calculated using confusion_matrix() from Scikit-learn.

## **Feature Selection**

Feature selection is performed using SelectKBest() from Scikit-learn, selecting the top 5 features based on the F-value.
The selected features are used to train the model, and the predicted values are evaluated using MSE.
The predicted values are converted to binary classes, and a confusion matrix is calculated.

## **Feature Scaling**

Feature scaling is performed using StandardScaler() from Scikit-learn.
The scaled features are used to train the model, and the predicted values are evaluated using MSE.
The predicted values are converted to binary classes, and a confusion matrix is calculated.

## **Visualization**

The confusion matrices for each scenario (without feature selection or scaling, with feature selection, and with feature scaling) are visualized using Matplotlib.
The plots are displayed side-by-side for comparison.

The output of the code look like this:

![Capture 2](https://github.com/user-attachments/assets/64c48fd8-9a40-4173-a531-9a222789195c)

![Capture 3](https://github.com/user-attachments/assets/ce5654e3-a715-431f-a4ed-85a42e24c166)
