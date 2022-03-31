import numpy as np
import matplotlib.pyplot as plt


def grad_descent(X, y, alpha, first_theta, iterations, gamma=0):
    theta = np.copy(first_theta)
    Vt = np.zeros(len(theta))

    L_log = []

    for i in range(iterations):
        # calculate the loss
        hx_minus_y = X.T @ theta - y
        L = np.sum(hx_minus_y ** 2) / len(y)
        L_log.append(L)

        # calculate the gradient
        grad_L = 2 * X @ hx_minus_y / len(y)

        # Update weights
        Vt = gamma * Vt + alpha * grad_L
        theta -= Vt

    return theta, L_log


if __name__ == "__main__":

    x0 = np.ones(3)
    x1 = np.array([0, 1, 2], dtype=float)
    x2 = x1 ** 2
    x_mat = np.array([x0, x1, x2])

    y = np.array([1, 3, 7], dtype=float)

    start_theta = np.array([2, 2, 0], dtype=float)

    iterations = 200
    gamma = 0.9
    alpha = 0.1

    theta, L_log = grad_descent(x_mat, y, alpha, start_theta, iterations, gamma)

    print(f"The loss after 200 iterations with alpha = {alpha} and gamma = {gamma} is {L_log[-1]}.")

    plt.figure()
    plt.plot(L_log)
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.title(f'gamma = {gamma}, alpha = {alpha}')
    plt.xlim(0, iterations)
    plt.ylim(0, max(L_log))
    plt.show()
