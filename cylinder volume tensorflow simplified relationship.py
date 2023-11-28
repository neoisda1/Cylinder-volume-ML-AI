import tensorflow as tf
import numpy as np

# Generate synthetic data for a cylinder with a simpler relationship
np.random.seed(42)
num_samples = 5000  # Increased data size for better learning

heights = np.random.uniform(1, 10, num_samples)
radii = np.random.uniform(1, 5, num_samples)

# Use a simplified relationship closer to the cylinder volume formula
volumes = np.pi * radii**2 * heights * 1.5  # Adjusted for simplicity

# Normalize data for better training
normalized_heights = heights / 10.0
normalized_radii = radii / 5.0
normalized_volumes = volumes / (np.pi * (5.0**2) * 10.0 * 1.5)  # Adjusted for normalization

# Create a linear regression model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[2])
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model for more epochs
model.fit(x=np.column_stack((normalized_radii, normalized_heights)), y=normalized_volumes, epochs=300)

# Save the trained model
model.save('cylinder_volume_model')

# Get user input for radius and height
user_radius = float(input("Enter the radius of the cylinder: "))
user_height = float(input("Enter the height of the cylinder: "))

# Normalize user input
normalized_user_radius = user_radius / 5.0
normalized_user_height = user_height / 10.0

# Load the trained model
loaded_model = tf.keras.models.load_model('cylinder_volume_model')

# Make a prediction using the loaded model
normalized_predicted_volume = loaded_model.predict([[normalized_user_radius, normalized_user_height]])[0, 0]

# Denormalize the predicted volume
predicted_volume_denormalized = normalized_predicted_volume * (np.pi * (5.0**2) * 10.0 * 1.5)  # Adjusted for denormalization

# Print the results
print("Predicted Volume:", predicted_volume_denormalized, "cubic units")


