# -*- coding: utf-8 -*-
import numpy as np
import math

# global variables to customize

class Optimize:
    m = 5  # degree of polynomial used -1
    tol = 10 ** -3  # tolerance for gradient
    max_iter = 100  # max number of iterations before termination

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
        def line_search(alpha_hi):
            """Setting up line-search"""
            alpha_max = 10  # max step length
            alpha_lo = 10**-6 # min guess
            c1 = 10 ** -3
            c2 = .9

            def phi(alpha_p):
                p = -self.grad_func(self.w) / np.linalg.norm(self.grad_func(self.w))
                return np.reshape(self.func(self.w + alpha_p * p),(1,1))

            def phi_grad(alpha_pg):
                p = -self.grad_func(self.w) / np.linalg.norm(self.grad_func(self.w))
                return np.matmul(p.T,self.grad_func(self.w + alpha_pg * p))
            
            def interpolate(alpha_l,alpha_r):
#                alpha_l = 10**-6
#                alpha_r = .05
                inter_grad = phi_grad(alpha_l)
                inter = phi(alpha_l)
                inter_a = phi(alpha_r)
                print(alpha_l,alpha_r,inter_grad,inter,inter_a)
                
                if inter_a <= inter + c1*alpha_r*inter_grad:
                    alpha_new = alpha_r
                    return alpha_new
                # quadratic interpolation
                else:
                    alpha_quad = -inter_grad*alpha_r**2/(2*(inter_a-inter-inter_grad*alpha_r))
                    print(alpha_quad)
                    
                if inter_a <= inter + c1*alpha_quad*inter_grad:
                    # if it works, return quad
                    print('quad int worked')
                    return alpha_quad
                
                else: # doesn't work then do cubic int
                    m = alpha_r**2*alpha_quad**2*(alpha_quad-alpha_r)
                    dummy1 = phi(alpha_quad) - inter - inter_grad*alpha_quad
                    dummy2 = inter_a - inter - inter_grad*alpha_r
                    a = (alpha_r**2*dummy1 - alpha_quad**2*dummy2)/m
                    b = (-alpha_r**3*dummy1 + alpha_quad**3*dummy2)/m
                    alpha_cub = (-b+math.sqrt(b**2 - 3*a*inter_grad))/(3*a)
                    if math.isnan(alpha_cub):
                        print('alpha is', alpha_quad)
                        return alpha_quad
                    else:
                        print('alpha is', alpha_cub)                        
                        return alpha_cub
            #Bisect because inter doesnt like to work
            def bisect(alpha_l, alpha_r):
                max_bisect = 100
                for i in range(max_bisect):
                    bis = (alpha_l+alpha_r)/2
                    der = phi_grad(bis)
                    if der > 0:
                        alpha_r = bis
                    elif der < 0:
                        alpha_l = bis
                    else:
                        return bis
                return bis
                    #zoom to find better alpha within bounds
            def zoom(alpha_lo, alpha_hi):
                zoom_count = 0
                while zoom_count <= 100:
                    zoom_count += 1
                    # interpolation to find new alpha_int
                    alpha_int = bisect(alpha_lo, alpha_hi)
                    print(alpha_lo, alpha_hi,alpha_int)
                    phi_a_zoom = phi(alpha_int)
                    
                    if phi_a_zoom > phi(0) + c1 * alpha_int * phi_grad(0) \
                        or phi_a_zoom >= phi(alpha_lo):
                        print('cond 1')
                        alpha_hi = alpha_int
                    else:
                        phi_ap_zoom = phi_grad(alpha_int)
                        if abs(phi_ap_zoom) <= -c2 * phi_grad(0):
                            print('cond 2')
                            return alpha_int
                        elif phi_ap_zoom * (alpha_hi - alpha_lo) >= 0:
                            print('cond 3')
                            alpha_hi = alpha_lo
                            alpha_lo = alpha_int
                try:
                    return alpha_min
                except NameError:
                    print('none found redo zoom')
                    return (alpha_lo+alpha_hi)/2
                
            i_a = 1
            while alpha_hi <= alpha_max: #what's while True in original? changed to this
                phi_a = phi(alpha_hi)
                phi_grad_a = phi(alpha_hi)

                # alpha violates the sufficient decrease condition
                if phi_a > phi(0) + c1 * alpha_hi * phi_grad(0) \
                        or (phi_a >= phi(alpha_lo) and i > 1):
                    print('zoom lo to hi')
                    alpha_min = zoom(alpha_lo, alpha_hi)  # alpha_min is used for next iteration of eval
                    print(alpha_min)
                    break
                # phi'(alpha) < old phi'(alpha)
                if abs(phi_grad_a) <= -c2*phi_grad(0):
                    print('worked', alpha_min)
                    alpha_min = alpha_hi
                    break
                # phi'(alpha) >= 0
                if phi_grad_a >= 0:
                    print('zoom hi to lo')
                    alpha_min = zoom(alpha_lo, alpha_hi)
                    print(alpha_min)
                    break
                else:
                    alpha_lo,alpha_hi = alpha_hi, 2*alpha_hi
                i_a += 1
                
                if alpha_hi > alpha_max:
                    return 'None'

            return alpha_min

        """Solving using steepest descent"""
        iterations = 0
        while np.linalg.norm(self.grad_func(self.w)) > self.tol and iterations < self.max_iter:
            alpha_hi = np.random.rand(1)/10 #bigger initial guess
            alpha = line_search(alpha_hi)
            for j in range(0, self.m):
                self.w[:, j] = self.w[:, j] - alpha * self.grad_func(self.w[:,j])
                    
            print(self.w)
            iterations += 1
























