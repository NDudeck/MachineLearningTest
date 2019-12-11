# -*- coding: utf-8 -*-
import numpy as np


# global variables to customize

class Optimize:
    m = 5  # degree of polynomial used -1
    tol = 10 ** -3  # tolerance for gradient
    max_iter = 100000  # max number of iterations before termination

    # if using line-search, use these alphas
    # alpha = np.empty((2,1))
    # alpha[0,1] = 0 #initial alpha value
    # alpha_max = 10**-2 #max step length
    # normalized direction of -gradient

    def __init__(self, func, w, grad_func, method="SD", tol=tol, max_iter=max_iter):

        self.func = func  # Function to be optimized
        self.w = w  # Variables to be optimized
        self.grad_func = grad_func  # Gradient of optimized function
        self.tol = tol  # tolerance of minimum value
        self.max_iter = max_iter  # maximum number of allowed iterations
        (self.n, self.m) = np.shape(w)  # number of variables being optimized

        if self.n > 0 and self.m > 0:
            is_matrix = True
        else:
            is_matrix = False

        if method == "SD":
            self.steepest_decent()

    def steepest_decent(self):
        def line_search():
            """Setting up line-search"""
            alpha = np.array([0, np.random.rand(1) / 1000])
            alpha_max = 1  # max step length
            c1 = 10 ** -4
            c2 = .9

            def phi(alpha_p):
                p = -self.grad_func(self.w) / np.linalg.norm(self.grad_func(self.w))
                return self.func(self.w + alpha_p * p)

            def phi_grad(alpha_pg):
                p = -self.grad_func(self.w) / np.linalg.norm(self.grad_func(self.w))
                return self.grad_func(self.w + alpha_pg * p)

            def zoom(alpha_lo, alpha_hi):
                while True:
                    # quadratic interpolation to find new alpha (alpha_j)
                    alpha_j = - (phi_grad(alpha_lo) * alpha_hi ** 2) / \
                              (2 * (phi(alpha_hi) - phi(alpha_lo) - phi_grad(alpha_lo) * alpha_hi))
                    phi_a_zoom = phi(alpha_j)
                    if phi_a_zoom > phi(0) + c1 * alpha_j * phi_grad(0) \
                            or phi_a_zoom >= phi(alpha_lo):
                        alpha_hi = alpha_j
                    else:
                        phi_ap_zoom = phi_grad(alpha_j)
                        if np.linalg.norm(phi_ap_zoom) <= -c2 * phi_grad(0):
                            return alpha_j
                        if phi_ap_zoom * (alpha_hi - alpha_lo) >= 0:
                            alpha_hi = alpha_lo
                            alpha_lo = alpha_j
            i_a = 1
            while True:
                phi_a = phi(alpha[i])

                # alpha violates the sufficient decrease condition
                if np.all(phi_a > phi(0) + c1 * alpha[i_a] * phi_grad(0)) \
                        or (np.all(phi_a >= phi(alpha[i_a - 1])) and i > 1):
                    alpha_min = zoom(alpha[i_a - 1], alpha[i_a])  # alpha_min is used for next iteration of eval
                    break
                # phi(alpha) > old phi(alpha)
                if np.linalg.norm(phi_grad(alpha[i_a])) <= -c2 * phi_grad(0):
                    alpha_min = alpha[i_a]
                    break
                # phi'(alpha) >= 0
                if np.all(phi_grad(alpha[i_a]) >= 0):
                    alpha_min = zoom(alpha[i_a], alpha[i_a - 1])
                    break
                alpha = np.append(alpha, 0.5 * (alpha[i_a] + alpha_max))
                i_a += 1

            return alpha_min

        """Solving using steepest descent"""
        iterations = 0
        while np.linalg.norm(self.grad_func) > self.tol and iterations < self.max_iter:
            for i in range(0, self.m):
                for j in range(0, self.n):
                    self.w[i, j] = self.w[i, j] - line_search() * self.grad_func[i]
            iterations += 1
