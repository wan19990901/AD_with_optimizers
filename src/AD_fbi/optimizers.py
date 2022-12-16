# Authors: Wenxu Liu, Queenie Luo, Guangya Wan, Mengyao Zheng, Di Zhen          #
# Course: AC207/CS107                                                           #
# File: optimzers.py                                                            #
# Description: Create three different optimization techniques that leverage     #
# forward mode automatic differentiation, enabling a user to find the minimum   #
# value of a function, location of the minimum value, and wall clock time to    #
# find these values.                                                            #
#################################################################################

import numpy as np
from .forward_mode import ForwardMode
import time

class Optimizer:
    """
    A class containing three different optimizer methods that leverage forward mode
    automatic differentiation, allowing a user to to find the minimum value of a function,
    location of the minimum value, and wall clock time required to find these values

    Examples
    --------
    # sample use case of instantiating and using a momentum optimizer
    >>> x = 2
    >>> fx = lambda x: x**4
    >>> print(Optimizer.momentum(x, fx, 1000))
    (0.038931846618652344, 2.0278841795288425e-05, -0.06710591258339543)
    
    # sample use case of instantiating and using a momentum optimizer
    >>> x = 1
    >>> fx = lambda x: (-1 * x.log()) + (x.exp() * x**4) / 10
    >>> Optimizer.momentum(x, fx, 1000)    
    (0.1545562744140625, 0.26172998379097046, 0.9423331580331616)

    # sample use case of instantiating and using a gradient descent optimizer
    >>> x = 1
    >>> fx = lambda x: (-1 * x.log()) + (x.exp() * x**4) / 10
    >>> Optimizer.gradient_descent(x, fx, 1000)
    (0.13236427307128906, 0.2617300604953795, 0.9424960556340723)
    
    """
    @staticmethod
    def momentum(x, fx, num_iter = 10000, alpha=0.01, beta=.9, verbose = False):
        """
        Parameters
        ----------
        x: the variable input (can be in either scalar or vector form)
        fx: the scalar function you would like to obtain the minimum for
        num_iter: the number of interations to perform (default 10,000)
        alpha: learning rate for the gradiant descent (default 0.01)
        beta: exponential decay rate (default 0.9)
        verbose: if verbose = True, output the intermediary positions (vals) and values (currvals) for every iteration, if verbose = False, only output the final results (default False)

        Returns
        -------
        opt_time: The time it takes to run the optimizer in seconds
        val: the position of the minimum value
        curr_val: the minimum value (can be in either scalar or vector form)
        vals: the intermediary positions of input variables for every iteration (only returns when verbose = False)
        currvals: the intermediary values of the function for every iteration (only returns when verbose = False)

        Examples
        --------
        >>> x = 1
        >>> fx = lambda x: (-1 * x.log()) + (x.exp() * x**4) / 10
        >>> Optimizer.momentum(x, fx, 1000)
        (0.030438899993896484, 2.0278841795288425e-05, -0.06710591258339543)

        >>> x = np.array([1, -1])
        >>> fx = lambda x, y:x**3 + y**2
        >>> Optimizer.momentum(x, fx, 1000)
        (0.08369302749633789, 2.7605629377339922e-05, array([ 3.02226506e-02, -3.30135704e-12]))

        >>> x = 2
        >>> fx = lambda x: (x - 1)**2 + 5
        >>> Optimizer.momentum(x, fx, 50, verbose = True)
        (0.009505748748779297,
        [5.843960615554842,
        5.597026644618029,
        5.387908529728599,
        5.240901124503859,
        5.1458759748109655],
        [1.9186732909771798,
        1.7726749928773602,
        1.6228230324326476,
        1.4908167932170397,
        1.3819371346320828])
       

        """
        vals=[]
        currvals=[]
        # start the timer
        start = time.time()
        # decay rate must be great than or equal to 0 and less than 1
        if 0 <= beta < 1 and 0 < alpha < 1:
            mt, curr_val = 0, x
            fm = ForwardMode(x, fx)
            val, x_der = fm.get_fx_value(), fm.get_derivative()
            vals.append(val)
            currvals.append(curr_val)
            # perform momentum optimization for the number of iterations specified
            for t in range(1, num_iter + 1):
                # calculate momentum
                mt = beta * mt + (1 - beta) * x_der
                
                # compute the new variation to update the current x location
                variation = alpha * mt
                curr_val = curr_val - variation
                
                # recalculate the function value and derivative at the updated value
                fm = ForwardMode(curr_val, fx)
                val, x_der = fm.get_fx_value(), fm.get_derivative()
                # store val and curr_val
                vals.append(val)
                currvals.append(curr_val)
        # raise the appropriate error for beta value not between 0 to 1
        elif beta>=1 or beta < 0:
            raise ValueError("Beta Values must be within the range of [0,1)")
        # raise the appropriate error for alpha value not between 0 to 1
        elif alpha >=1 or alpha <= 0:
            raise ValueError("Learning rate alpha must be within the range of (0,1)")
        # end the timer and compute wall clock time
        end = time.time()
        opt_time = end - start
        
        if verbose == False:
            return opt_time, val, curr_val
        else:
            return opt_time, vals, currvals
    
    
    
    @staticmethod
    def gradient_descent(x, fx, num_iter = 10000, alpha=0.001, verbose = False):
        """
        Parameters
        ----------
        x: the starting point to find the minimum point
        fx: the scalar function you would like to obtain the minimum point
        num_iter: the number of interations to perform (default 10,000)
        alpha: learning rate for the gradiant descent (default 0.001)
        verbose: if verbose = True, output the intermediary positions (vals) and values (currvals) for every iteration, if verbose = False, only output the final results (default False)


        Returns
        -------
        opt_time: The time it takes to run the optimizer in seconds
        val: the position of the minimum value
        curr_val: the minimum value (can be in either scalar or vector form)
        vals: the intermediary positions of input variables for every iteration (only returns when verbose = False)
        currvals: the intermediary values of the function for every iteration (only returns when verbose = False)
        

        Examples
        --------
        >>> x = 1
        >>> fx = lambda x: (-1 * x.log()) + (x.exp() * x**4) / 10
        >>> Optimizer.gradient_descent(x, fx, 1000)
        (0.13236427307128906, 0.2617300604953795, 0.9424960556340723)
       
        >>> x = np.array([1, -1])
        >>> fx = lambda x, y:x**3 + y**2
        >>> Optimizer.gradient_descent(x, fx, 1000)
        (0.042717695236206055, 0.03381871354483734, array([ 0.24973993, -0.13506452]))
        
        >>> x = 2
        >>> fx = lambda x: (x - 1)**2 + 5
        >>> Optimizer.gradient_descent(x, fx, 50, verbose = True)
       (0.0025625228881835938,
        [5.960750957026343,
        5.9230424014270335,
        5.886813870546916,
        5.852007274832185,
        5.8185668046884285],
        [array([1.98017904]),
        array([1.96075096]),
        array([1.94170795]),
        array([1.9230424]),
        array([1.90474682])])
        
        """
        # initiate the array to store the function values
        vals=[]
        # initiate the array to store the intermediate values for the input variable(s)
        currvals=[]
        # start the timer
        start = time.time()
        # learning rate value must be great than or equal to 0 and less than 1
        if 0 < alpha < 1:
            curr_val = x
            fm = ForwardMode(x, fx)
            val, x_der = fm.get_fx_value(), fm.get_derivative()
            vals.append(val)
            currvals.append(curr_val)
            # perform gradient descent for the number of iterations specified
            for t in range(1, num_iter + 1):
                # compute the new variation to update the current x location
                variation = alpha * x_der
                
                curr_val = curr_val - variation
                # recalculate the function value and derivative at the updated value
                fm = ForwardMode(curr_val, fx)
                val, x_der = fm.get_fx_value(), fm.get_derivative()
                # store val and curr_val
                vals.append(val)
                currvals.append(curr_val)
        # raise the appropriate error for alpha value not between 0 to 1
        else:
            raise ValueError("Learning rate alpha must be within the range of (0,1)")
        # end the timer and compute wall clock time
        end = time.time()
        opt_time = end - start
        
        if verbose == False:
            return opt_time, val, curr_val
        else:
            return opt_time, vals, currvals
        
    

    
    @staticmethod
    def ADAGRAD(x, fx, num_iter=10000, alpha=0.01, epsilon=1e-8, verbose=False):
        """
        Parameters
        ----------
        x: The variable input (can be in either scalar or vector form)
        fx: The function you would like to obtain the minimum for
        num_iter: Number of interations to perform (defualt 10000)
        alpha: Learning rate for the gradiant descent (default 0.01)
        epsilon: Denominator value to assure that ZeroDivisionError is not raised (default 1e-8)
        verbose: Boolean to whether return the full trace of result at each step. Default to False.
        
        Returns
        -------
        opt_time: The time it takes to run the optimizer in seconds
        fx_val: The minimum value
        fx_vals: List of values at each step
        x_val: Position of the minimum value (can be in either scalar or vector form)
        x_vals: List of the position at each step
        
        Examples
        --------
        >>> x = 1
        >>> fx = lambda x: (-1 * x.log()) + (x.exp() * x**4) / 10
        >>> Optimizer.ADAGRAD(x, fx, 1000)
        (0.13465595245361328, 0.2617299837909705, 0.9423331580331631)

        >>> x = np.array([1, -1])
        >>> fx = lambda x, y:x**3 + y**2
        >>> Optimizer.ADAGRAD(x, fx, 1000)
        (0.1048738956451416, 0.35386341347850286, 
        array([ 0.51915345, -0.46253758]))

        >>> x = 2
        >>> fx = lambda x: (x - 1)**2 + 5
        >>> Optimizer.ADAGRAD(x, fx, 1000)
        (0.0962369441986084, 5.213941015895882, array([1.46253758]))

        >>> fx = lambda x, y: (1 - x)**2 + 100*(y - x**2)**2
        >>> Optimizer.ADAGRAD(np.array([1,2]), fx, num_iter=10000, alpha=0.2)
        (1.715116024017334, 0.0004431195291550218, 
        array([1.02103907, 1.04258986]))
        """
        # store results as lists
        fx_vals, x_vals = [], []

        # start the timer
        start = time.time()

        # learning rate value must be great than 0 and less or equal to 1
        if 0 < alpha <=1:
            x_val = x
            fm = ForwardMode(x, fx)
            fx_val, x_der = fm.get_fx_value(), fm.get_derivative()
            fx_vals.append(fx_val)
            x_vals.append(x)
            G = x_der**2
            # perform ADAGRAD optimization for the number of iterations specified
            for t in range(1, num_iter + 1):
                # compute the new variation to update the current x location
                variation = alpha / np.sqrt(G+epsilon) * x_der
                x_val = x_val - variation
                # recalculate the function value and derivative at the updated value
                fm = ForwardMode(x_val, fx)
                fx_val, x_der = fm.get_fx_value(), fm.get_derivative()
                fx_vals.append(fx_val)
                x_vals.append(x_val)
                G = G + x_der**2
        # raise the appropriate error for negative learning rate
        else:
            raise ValueError("Learning rate alpha must be within the range of (0,1]")

        # end the timer and compute wall clock time
        end = time.time()
        opt_time = end - start

        if verbose:
            return opt_time, fx_vals, x_vals
        else:
            return opt_time, fx_val, x_val

