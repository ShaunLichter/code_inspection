from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

# Load the Linnerud dataset
linnerud = datasets.load_linnerud()
X = linnerud.data
y = linnerud.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit a linear regression model with non-default hyperparameters
regr = LinearRegression(fit_intercept=False)
regr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regr.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

# Print the results
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Explained Variance Score: {evs}')

# Use cross-validation to evaluate the model
scores = cross_val_score(regr, X, y, cv=5)
print(f'Cross-validation scores: {scores}')
