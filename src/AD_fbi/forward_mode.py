# Authors: Wenxu Liu, Queenie Luo, Guangya Wan, Mengyao Zheng, Di Zhen                 #
# Course: AC207/CS107                                                                  #
# File: forward_mode.py                                                                 #
# Description: Perform forward mode automatic differentiation method, enabling a user  #
# to output just the function values, just the derivative values, or both the function #
# and derviative values in a tuple
########################################################################################


import numpy as np
from .dual_number import DualNumbers


class ForwardMode:
    """
    A class to perform forward mode automatic differentiation mode, enabling a user
    to output just the function values evaluated at the evaluation point, just the derivative values, 
    or both the function and derviative values in a tuple.
    
    Instance Variables
    ----------
    input_values: a scalar or a vector which indicates the evaluation point
    input_function: a scalar function or a vector of functions 
    seed: a seed vector (optional parameter: default value = 1 or np.ones(len(self.inputs))
    
    Examples
    --------
    # get function value
    >>> func = lambda x: x**2 + 1
    >>> fm = ForwardMode(1, func, -1)
    >>> fm.get_fx_value()
    2.0
    
    # get function derivative
    >>> fm.get_derivative()
    array([-2.])
    
    # get function value and derivative
    >>> fm.calculate_dual_number()
    (2, array([-2.]))
    
    # get univariate vector function value and directional derivative
    # the function takes a scalar input and outputs an array
    >>> func = lambda x: (x + 1, x**3)
    >>> fm = ForwardMode(1, func, -1)
    >>> fm.calculate_dual_number()
    (array([2., 1.]), array([[-1.],
                             [-3.]]))

        
    
    """

    def __init__(self, input_values, input_function, seed = "default seed"):
        self.inputs = input_values
        self.functions = input_function
        
        # if there is no input value for seed
        if seed == 'default seed':
            # if the input variable is a scalar
            if np.isscalar(self.inputs):
                self.seed = 1
            
            # if the input variable is an array
            else:
                self.seed = np.ones(len(self.inputs))
                
        # if seed is specified by the user
        else:
            self.seed = seed

    def get_fx_value(self):
        """
        Parameters
        ----------
        None
        
        Returns
        -------
        value of the input function at the evaluation point
        
        Examples
        --------
        # get univariate scalar function value
        >>> func = lambda x: x
        >>> fm = ForwardMode(1, func, -1)
        >>> fm.get_fx_value()
        1.0
        
        # get univariate vector function value and directional derivative
        # the function takes a scalar input and outputs an array
        >>> func = lambda x: (x + 1, x**3)
        >>> fm = ForwardMode(1, func, -1)
        >>> fm.get_fx_value()
        array([2., 1.])
        
        # get multivariate scalar function value and directional derivative
        # the function takes an array input and outputs a scalar
        >>> func = lambda x, y: 2*x + y
        >>> fm = ForwardMode(np.array([1, 1]), func, [2, -1])
        >>> fm.get_fx_value()
        3
        
        # get multivariate array function value and directional derivative
        # the function takes an array input and outputs an array
        >>> func = lambda x, y: (2*x + y, 3*y + x**2)
        >>> fm = ForwardMode(np.array([1, 1]), func, [2, -1])
        >>> fm.get_fx_value()
        array([3., 4.]
        
        """

        return self.calculate_dual_number()[0]
    

    def get_derivative(self):
        """
        Parameters
        ----------
        None
        
        Returns
        -------
        the derivative of the input function at the evaluation point
        
        Examples
        --------
        # get univariate scalar function value and the directional derivative
        # the function takes a scalar input and outputs a scalar
        >>> func = lambda x: x + 1
        >>> fm = ForwardMode(1, func, -1)
        >>> fm.get_derivative()
        -1.0
        
        # get univariate vector function value and directional derivative
        # the function takes a scalar input and outputs an array
        >>> func = lambda x: (x + 1, x**3)
        >>> fm = ForwardMode(1, func, -1)
        >>> fm.get_derivative()
        array([[-1.],
                [-3.]])
        
        # get multivariate scalar function value and directional derivative
        # the function takes an array input and outputs a scalar
        >>> func = lambda x, y: 2*x + y
        >>> fm = ForwardMode(np.array([1, 1]), func, [2, -1])
        >>> fm.get_derivative()
        array([ 4., -1.])
        
        # get multivariate array function value and directional derivative
        # the function takes an array input and outputs an array
        >>> func = lambda x, y: (2*x + y, 3*y + x**2)
        >>> fm = ForwardMode(np.array([1, 1]), func, [2, -1])
        >>> fm.get_derivative()
        array([[ 4., -1.],
                [ 4., -3.]])
    
        """

        return self.calculate_dual_number()[1]
    
    @staticmethod
    def fuse_multiple_inputs(functions, n_col):
        
        func_num = len(functions)
        
        # initialize the arrays to store the function and directional derivatives for the input functions
        func_val = np.empty(func_num)  
        func_der = np.empty([func_num, n_col])    
        
        # compute the function and directional derivative value for each input functions
        for i, funct in enumerate(functions):
            func_val[i] = funct.val
            func_der[i] = funct.derv
        
        return func_val, func_der
        
    
    def calculate_dual_number(self):
        """
        Parameters
        ----------
        None
        
        Returns
        -------
        evaluated value and derivative of the input function at the evaluation point
        
        Examples
        --------
        # get univariate scalar function value and the directional derivative
        # the function takes a scalar input and outputs a scalar
        >>> func = lambda x: x + 1
        >>> fm = ForwardMode(1, func, -1)
        >>> fm.calculate_dual_number()
        (2.0, -1.0)
        
        # get univariate vector function value and directional derivative
        # the function takes a scalar input and outputs an array
        >>> func = lambda x: (x + 1, x**3)
        >>> fm = ForwardMode(1, func, -1)
        >>> fm.calculate_dual_number()
        (array([2., 1.]), array([[-1.],
                                 [-3.]]))
        
        # get multivariate scalar function value and directional derivative
        # the function takes an array input and outputs a scalar
        >>> func = lambda x, y: 2*x + y
        >>> fm = ForwardMode(np.array([1, 1]), func, [2, -1])
        >>> fm.calculate_dual_number()
        (3, array([ 4., -1.]))
        
        # get multivariate array function value and directional derivative
        # the function takes an array input and outputs an array
        >>> func = lambda x, y: (2*x + y, 3*y + x**2)
        >>> fm = ForwardMode(np.array([1, 1]), func, [2, -1])
        >>> fm.calculate_dual_number()
        (array([3., 4.]), array([[ 4., -1.],
                                 [ 4., -3.]]))
        
        """
        
        
            
        # check if the input is a scalar
        if np.isscalar(self.inputs):
            # enforce the self.inputs to become an array
            self.inputs = np.array([self.inputs]) 
            input_num = 1
        elif len(self.inputs) == 1:
            input_num = 1
        else:
            input_num = len(self.inputs)
            

        # initialize the list of dual numbers
        dual_list = [0] * input_num
        
        # get the corresponding seed vector       
        def get_seed_vector(index):
            
            seed_vector = np.zeros(input_num)
            
            # if self.seed is a scalar
            if np.isscalar(self.seed):
                seed_vector[index] = self.seed
            else:
                # if self.seed is an array
                if input_num != len(self.seed):
                    raise ValueError("ERROR: Seed vector length mismatchs with the number of input variables.")
                elif len(self.seed) == 1:
                    seed_vector[index] = self.seed[0]
                else:
                    seed_vector[index] = self.seed[index]

            return seed_vector
        
        for i in range(input_num):
            dual_list[i] = DualNumbers(self.inputs[i], get_seed_vector(i))
        
        z = self.functions(*dual_list)
        
        try:
            # input function is a scalar function
            if len(z.derv) == 1: # the input is a scalar
                return float(z.val), float(z.derv)
            else:
                return z.val, z.derv
        except AttributeError:
            # input function is an array function
            return self.fuse_multiple_inputs(z, input_num)
            





    