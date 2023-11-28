import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# Generate synthetic data for training
np.random.seed(42)
num_samples = 1000


heights = np.random.uniform(1, 10, num_samples)
radii = np.random.uniform(1, 5, num_samples)

volumes = np.pi * radii**2 * heights

# Reshape the data for polynomial regression
X = np.column_stack((heights, radii))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, volumes, test_size=0.2, random_state=42)

# Create and train the polynomial regression model
degree = 3  # Set the degree of the polynomial
model = make_pipeline(StandardScaler(), PolynomialFeatures(degree), Ridge(alpha=0.1))
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')

# Plot the results
plt.scatter(radii, volumes, label='Actual Data')
sorted_indices = X_test[:, 1].argsort()
plt.scatter(radii[sorted_indices], y_pred[sorted_indices], label='Predicted Data', color='red')
plt.xlabel('Radius')
plt.ylabel('Volume')
plt.legend()
plt.show()

# User input for predicting cylinder volume
user_radius = float(input("Enter the radius of the cylinder: "))
user_height = float(input("Enter the height of the cylinder: "))

# Reshape the user input for prediction
user_input = np.array([[user_height, user_radius]])

# Make a prediction for the user input
predicted_volume = model.predict(user_input)[0]
print(f'Predicted Volume of the Cylinder: {predicted_volume}')
