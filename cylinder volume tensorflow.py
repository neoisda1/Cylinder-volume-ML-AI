# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:43:58 2023

@author: jafar
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for a cylinder
np.random.seed(42)
num_samples = 100

heights = np.random.uniform(1, 10, num_samples)
radii = np.random.uniform(1, 5, num_samples)

# Calculate cylinder volumes using the formula: V = π * r^2 * h
volumes = np.pi * radii**2 * heights

# Normalize data for better training
normalized_heights = heights / 10.0
normalized_radii = radii / 5.0
normalized_volumes = volumes / (np.pi * (5.0**2) * 10.0)

# Create a linear regression model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[2])
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x=np.column_stack((normalized_radii, normalized_heights)), y=normalized_volumes, epochs=100)

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
predicted_volume_denormalized = normalized_predicted_volume * (np.pi * (5.0**2) * 10.0)

# Print the results
print("Predicted Volume:", predicted_volume_denormalized, "cubic units")
