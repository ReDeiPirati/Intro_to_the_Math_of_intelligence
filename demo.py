"""
This is a small Demo to show gradient descent on a linear regression task.
The optimal values of m and b can be actually calculated with way less effort than doing a linear regression!
"""

from __future__ import print_function
from numpy import *

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    """Compute The Loss(MSE) for the linear regression model"""
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]  # distance
        y = points[i, 1]  # calories burned
        totalError += (y - (m * x + b)) ** 2  # Squared Error
    return totalError / float(len(points))  # Mean Squared Error(MSE)


def step_gradient(b_current, m_current, points, learningRate):
    """Update the m and b using the gradients with respect to the Loss"""
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # b Gradient wrt the Loss over the full dataset
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        # m Gradient wrt the Loss over the full dataset
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    # GD Update Rulr for b and m
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    """Training with Batch Gradients Descent"""
    b = starting_b
    m = starting_m
    # Training for num_iterations
    for i in range(num_iterations):
        # Update b and m after a full dataset training step
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


def run():
    # Get the points from the mounted dataset
    points = genfromtxt("/dataset/data.csv", delimiter=",")
    learning_rate = 0.0001 # Step lenght of gradient descent algo
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(
        initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(
        num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    run()
