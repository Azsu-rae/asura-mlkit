
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate sample data for the example
# Ensure Random value are predictable
np.random.seed(0)
X = 2 * np.random.rand(100, 1)

#Y = AX + B
Y = 3 * X + 4 + np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)
Y_train_pred = model.predict(X_train)

# 1st Graph
# Plot the training data and the regression line
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, Y_train_pred, color='red') # Regression Line
plt.title('Linear Regression - Training Set')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 2nd Graph
# Plot the test data and the regression line
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red') # Regression Line
plt.title('Linear Regression - Test Set')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
