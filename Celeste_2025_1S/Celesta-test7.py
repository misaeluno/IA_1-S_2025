import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

# Generate some sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]

# Parameters for gradient descent
eta = .5  # learning rate
n_iterations = 100
theta = np.random.randn(2, 1)  # random initialization

# For cost function history
cost_history = []

# For 3D plot of cost function
theta0_vals = np.linspace(0, 8, 100)
theta1_vals = np.linspace(0, 6, 100)
theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)
J_vals = np.zeros_like(theta0_mesh)
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        theta_ij = np.array([theta0_vals[i], theta1_vals[j]]).reshape(-1, 1)
        J_vals[i, j] = (1/(2*len(X_b))) * np.sum((X_b.dot(theta_ij) - y)**2)

# Create all figures upfront
plt.ion()  # Turn on interactive mode

# Figure 3: 3D surface of cost function
fig3 = plt.figure(figsize=(12, 8))
ax3 = fig3.add_subplot(111, projection='3d')
surf = ax3.plot_surface(theta0_mesh, theta1_mesh, J_vals, cmap=cm.coolwarm, alpha=0.6)
ax3.set_title('Cost Function Surface')
ax3.set_xlabel(r'$\theta_0$')
ax3.set_ylabel(r'$\theta_1$')
ax3.set_zlabel('Cost')
point = ax3.scatter([theta[0]], [theta[1]], [0], c='r', s=100)  # Starting point

# Perform gradient descent and update plots
for iteration in range(n_iterations):
    gradients = (1/len(X_b)) * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

    # Calculate current cost
    cost = (1/(2*len(X_b))) * np.sum((X_b.dot(theta) - y)**2)
    cost_history.append(cost)

    # Update Figure 3: 3D surface with current theta position
    # Remove old point and add new one
    point.remove()
    point = ax3.scatter([theta[0]], [theta[1]], [cost], c='r', s=100)
    fig3.canvas.draw()

    # Pause briefly to allow updates to display
    plt.pause(0.05)

    # Clear output for the next plot if in a notebook
    # display.clear_output(wait=True)
    # display.display(fig1, fig2, fig3)

plt.ioff()  # Turn off interactive mode
plt.show()
