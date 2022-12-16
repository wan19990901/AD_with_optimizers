# Authors: Wenxu Liu, Queenie Luo, Guangya Wan, Mengyao Zheng, Di Zhen          #
# Course: AC207/CS107                                                                #
# File: dual_number.py                                                          #
# Description: This class defines the dual number object to be used in forward mode 
# automatic differentiation. It contains methods to initialize the object, set and get    #
# the function and derivative value of the object, overload elementary          #
# operations, and define elementary functions.                                  #
#################################################################################

import numpy as np

def is_numeric(x):
    r"""Method to check whether input x contains numeric elements or not
    
    Parameters
    ----------
    x: An object to be checked if the values contained within it are either integers or floats
    
    Returns
    -------
    True if a scaler is a float/integer, or all of the elements within object x are either integers or floats. False otherwise.

    Examples
    --------
    >>> x = 6
    >>> print(is_numeric(x))
    True
    >>> x = 'e'
    >>> print(is_numeric(x))
    False
    >>> x = [1,2]
    >>> print(is_numeric(x))
    True
    >>> x = [1,'cs107']
    >>> print(is_numeric(x))
    False
    """
    # if the input object is not a character but a scalar, it must be numeric
    if not isinstance(x, str) and np.isscalar(x):
        return True
    # if the input object is an array, check all contents
    else:
        # if an element is numeric, move on to the next, otherwise it's not valid
        for i in x:
            if isinstance(i, (int, np.int32, np.int64, float, np.float64)):
                pass
            else:
                return False
        return True
    
class DualNumbers:
    r"""A class representing a variable object to be used in automatic differentiation
    
    The class contains methods to initialize the object, set and get the function and 
    derivative value of the object, overload elementary operations, and define elementary 
    functions that are used in forward mode calculation
    
    Instance Variables
    ----------
    val: value of the DualNumbers object
    derv: derivative(s) of the DualNumbers object
    
    Returns
    -------
    A DualNumbers object that contains the value and derivative
    
    Examples
    --------
    >>> z1 = DualNumbers(1, -1)
    >>> z2 = DualNumbers(2.2, 0)
    >>> print(z_1 + z_2)
    Values: 3.2, Derivatives: -1
    """
    
    def __init__(self, val, derv):
        r"""A constructor to create DualNumbers object with a value and a derivative
        
        Parameters
        ----------
        val: integer or float object that represents the value of DualNumbers object
        derv_seed: integer or float object that represents the seed value for the derivative of DualNumbers object 
        
        Returns
        -------
        None
        """
        if is_numeric(val):
            self._val = val
        else:
            raise TypeError('Error: Input value should be an int or float')
        if is_numeric(derv):
            self._derv = derv
        # in the case of a 1D array of derivatives, check each element individually
        elif isinstance(derv, np.ndarray) and len(derv.shape) == 1:
            try:
                derv = derv.astype(float)
            except ValueError:
                raise ValueError('Error: Input value should be an int or float')
            self._derv = derv
        # for all other non-numeric cases, raise the appropriate value error
        else:
            raise TypeError('Error: Input value must be an array of ints/floats or be a scalar int/float')

    @property
    def val(self):
        r"""A method to retrieve the value attribute of DualNumbers object
        
        Parameters
        ----------
        None
        
        Returns
        -------
        val attribute of DualNumbers object
        
        Examples
        --------
        >>> z = DualNumbers(1, 2)
        >>> print(z.val)
        1
        """
        return self._val

    @val.setter
    def val(self, val):
        r"""A method to set the value attribute of DualNumbers object
        
        Parameters
        ----------
        val: float or integer object that represents the value of the DualNumbers object
        
        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            If input is a non-integer or non-float value or a 1D numpy array of non-integer or non-float values
        
        Examples
        --------
        >>> z = DualNumbers(1, 2)
        >>> z.val = 2
        >>> print(z.val)
        2
       """
        if is_numeric(val):
            self._val = val
        else:
            raise TypeError('Error: Input value should be an int or float')

    @property
    def derv(self):
        r"""A method to retrieve the derivative attribute of DualNumbers object
        
        Parameters
        ----------
        None
        
        Returns
        -------
        derv attribute of DualNumbers object
        
        Examples
        --------
        >>> z = DualNumbers(1, 2)
        >>> print(z.derv)
        2
        """
        return self._derv

    @derv.setter
    def derv(self, derv):
        r"""A method to set the derivative attribute of DualNumbers object
        
        Parameters
        ----------
        derv: float/integer object or 1D array of float/integer objects that represents DualNumbers derivative
        
        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            If input is a non-integer or non-float value or contains a 1D numpy array of non-integer or non-float values
        
        Examples
        --------
        >>> z = DualNumbers(1, 2)
        >>> z.derv = 1
        >>> print(z.derv)
        1
        """
        if is_numeric(derv):
            self._derv = derv
        # in the case of a 1D array of derivatives, check each element individually
        elif isinstance(derv, np.ndarray) and len(derv.shape) == 1:
            try:
                derv = derv.astype(float)
            except ValueError:
                raise ValueError('Error: Input value should be an int or float')
            self._derv = derv
        # for all other non-numeric cases, raise the appropriate value error
        else:
            raise TypeError('Error: Input value must be an array of ints/floats or be a scalar int/float')

    def __repr__(self):
        r"""A method to overload the string representation for DualNumbers object
        
        Parameters
        ----------
        None
        
        Returns
        -------
        A string representation of DualNumbers object
        
        Examples
        --------
        >>> print(DualNumbers(1, 2))
        Values: 1, Derivatives: 2
        """
        return f'Values: {self.val}, Derivatives: {self.derv}'
    
    def __add__(self, other):
        r"""A method to perform addition operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object or DualNumbers object
        
        Returns
        -------
        A DualNumbers object as the result of the addition operation
        
        Examples
        --------
        >>> z1 = DualNumbers(1, -1)
        >>> z2 = DualNumbers(1, 2)
        >>> print(z1 + z2)
        Values: 2, Derivatives: 1
        
        >>> x1 = DualNumbers(1, np.array([-1, 0]))
        >>> x2 = DualNumbers(1, np.array([0, 2]))
        >>> print(x1 + x2)
        Values: 2, Derivatives:[-1  2]
        """
        # perform addition if other is a dual number
        try:
            f = self.val + other.val
            f_prime = self.derv + other.derv
        # perform addition if other is a real number
        except AttributeError:
            f = self.val + other
            f_prime = self.derv
        return DualNumbers(f, f_prime)
    
    def __radd__(self, other):
        r"""A method to perform reverse addition operation on a DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object
        
        Returns
        -------
        A DualNumbers object as the result of reverse addition operation
        
        Examples
        --------
        >>> z1 = 2
        >>> z2 = DualNumbers(1, -1)
        >>> print(z1 + z2)
        Values: 3, Derivatives: -1

        >>> x1 = 2
        >>> x2 = DualNumbers(1, np.array([0, -1]))
        >>> print(x1 + x2)
        Values: 2, Derivatives: [0  -1]
        """
        return self + other
    
    def __sub__(self, other):
        r"""A method to perform subtraction operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: A float/integer object or DualNumbers object
        
        Returns
        -------
        A DualNumbers object as the result of the subtraction operation
        
        Examples
        --------
        >>> z1 = DualNumbers(1, -1)
        >>> z2 = DualNumbers(1, 2)
        >>> print(z1 - z2)
        Values: 0, Derivatives: -3

        >>> x = DualNumbers(1, np.array([-1, 0]))
        >>> y = DualNumbers(1, np.array([0, 2]))
        >>> print(x - y)
        Values: 0, Derivatives: [-1 -2]
        """
        # perform subtraction if other is a dual number
        try:
            f = self.val - other.val
            f_prime = self.derv - other.derv
        # perform subtraction if other is a real number
        except AttributeError:
            f = self.val - other
            f_prime = self.derv
        return DualNumbers(f, f_prime)
    
    def __rsub__(self, other):
        r"""A method to perform reverse subtraction operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object
        
        Returns
        -------
        A DualNumbers object as the result of the reverse subtraction operation
        
        Examples
        --------
        >>> z1 = 1
        >>> z2 = DualNumbers(1, -1)
        >>> print(z1 - z2)
        Values: 0, Derivatives: 1

        >>> x1 = 1
        >>> x2 = DualNumbers(1, np.array([0, 2]))
        >>> print(x1 - x2)
        Values: 0, Derivatives: [0  -2]
        """
        return other + (-self)
    
    def __mul__(self, other):
        r"""A method to perform multiplication operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object or DualNumbers object
        
        Returns
        -------
        A DualNumbers object as the result of the multiplication operation
        
        Examples
        --------
        >>> z1 = DualNumbers(1, -1)
        >>> z2 = DualNumbers(1, 2)
        >>> print(z1 * z2)
        Values: 1, Derivatives: 1

        >>> x1 = DualNumbers(1, np.array([-1, 0]))
        >>> x2 = DualNumbers(1, np.array([0, 2]))
        >>> print(x1 * x2)
        Values: 1, Derivatives: [-1  2]
        """
        # perform multiplication if other is a dual number
        try:
            f = self.val * other.val
            f_prime = self.val * other.derv + self.derv * other.val
        # perform multiplication if other is a real number
        except AttributeError:
            f = self.val * other
            f_prime = self.derv * other
        return DualNumbers(f, f_prime)
    
    def __rmul__(self, other):
        r"""A method to perform the reverse multiplication operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object
        
        Returns
        -------
        A DualNumbers object as the result of the reverse multiplication operation
        
        Examples
        --------
        >>> z1 = 1
        >>> z2 = DualNumbers(1, -1)
        >>> print(z1 * z2)
        Values: 1, Derivatives: -1

        >>> x1 = 1
        >>> x2 = DualNumbers(1, np.array([-1, 0]))
        >>> print(x1 * x2)
        Values: 1, Derivatives: [-1  0]
        """
        return self * other
    
    def __truediv__(self, other):
        r"""A method to perform division operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object or DualNumbers object
        
        Returns
        -------
        A DualNumbers object as the result of the division operation
        
        Raises
        ------
        ZeroDivisionError if denominator in division is zero
        
        Examples
        --------
        >>> z1 = DualNumbers(1, -1)
        >>> z2 = DualNumbers(1, 2)
        >>> print(z1 / z2)
        Values: 1.0, Derivatives: -3.0

        >>> x = DualNumbers(1, np.array([-1, 0]))
        >>> y = DualNumbers(1, np.array([0, 2]))
        >>> print(x / y)
        Values: 1.0, Derivatives: [-1. -2.]

        >>> z1 = DualNumbers(1, -1)
        >>> z2 = DualNumbers(0, 2)
        >>> print(z1 / z2)
        ZeroDivisionError: Error: Denominator in division should not be 0

        >>> z1 = DualNumbers(1, -1)
        >>> z2 = 0
        >>> print(z1 / z2)
        ZeroDivisionError: Error: Denominator in division should not be 0
        """
        # perform division if other is a dual number
        try:
            # avoid zero division
            if other.val == 0:
                raise ZeroDivisionError("Error: Denominator in division should not be 0")
            f = self.val / other.val
            f_prime = (self.derv * other.val - self.val * other.derv) / (other.val ** 2)
            return DualNumbers(f, f_prime)
        # perform division if other is a real number
        except AttributeError:
            # avoid zero division
            if other == 0:
                raise ZeroDivisionError("Error: Denominator in division should not be 0")
            f = self.val / other
            f_prime = self.derv / other
            return DualNumbers(f, f_prime)
        
    def __rtruediv__(self, other):
        r"""A method to perform the reverse division operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object
        
        Returns
        -------
        A DualNumbers object as the result of the reverse division operation
        
        Raises
        ------
        ZeroDivisionError if denominator in division is zero
        
        Examples
        --------
        >>> z1 = 1
        >>> z2 = DualNumbers(1, -1)
        >>> print(z1 / z2)
        Values: 1.0, Derivatives: 1.0

        >>> x1 = 1
        >>> x2 = DualNumbers(1, np.array([0, -1]))
        >>> print(x1 / x2)
        Values: 1.0, Derivatives: [0.  1.]

        >>> z1 = 1
        >>> z2 = DualNumbers(0, -1)
        >>> print(z1 / z2)
        ZeroDivisionError: Error: Denominator in division should not be 0
        """
        if self.val == 0:
            raise ZeroDivisionError("Error: Denominator in division should not be 0")
        f = other / self.val
        f_prime = (- other * self.derv) / (self.val ** 2)
        return DualNumbers(f, f_prime)

    def __pow__(self, other):
        r"""A method to perform power operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object or DualNumbers object
        
        Returns
        -------
        A DualNumbers object as the result of the power operation
        
        Raises
        ------
        ValueError
            If negative number is raised to a fraction power with an even denominator
            If the power is less than 1 and differentiation occurs at 0
        
        Examples
        --------
        >>> z1 = DualNumbers(1, -1)
        >>> z2 = DualNumbers(1, 2)
        >>> print(z1 ** z2)
        Values: 1, Derivatives: -1.0

        >>> x = DualNumbers(1, np.array([-1, 0]))
        >>> y = DualNumbers(1, np.array([0, 2]))
        >>> print(x ** y)
        Values: 1, Derivatives: [-1. 0.]

        >>> z1 = DualNumbers(-1, -1)
        >>> z2 = DualNumbers(0.2, 2)
        >>> print(z1 ** z2)
        ValueError: Error: Attempted to raise a negative number to a fraction power with even denominator

        >>> z1 = DualNumbers(0, -2)
        >>> z2 = DualNumbers(0.2, 2)
        >>> print(z1 ** z2)
        ValueError: Error: Attempted to find derivative at 0 when power is less than 1
        """
        # perform power operation if other is a dual number
        try:
            # avoid raising a negative number to a fraction power with an even denominator
            if self.val < 0 and other.val % 1 != 0 and other.val.as_integer_ratio()[1] % 2 == 0:
                raise ValueError("Error: Attempted to raise a negative number to a fraction power with even denominator")
            # avoid having a 0 derivative when the power is less than 1
            if self.val == 0 and other.val < 1:
                raise ValueError("Error: Attempted to find derivative at 0 when the power is less than 1")

            f = self.val ** other.val
            f_prime = (self.val ** (other.val - 1)) * self.derv * other.val + (
                    self.val ** other.val) * other.derv * np.log(self.val)
            return DualNumbers(f, f_prime)

        # perform power operation if other is a real number
        except AttributeError:
            # avoid raising a negative number to a fraction power with an even denominator
            if self.val < 0 and other % 1 != 0 and other.as_integer_ratio()[1] % 2 == 0:
                raise ValueError("Error: Attempted to raise a negative number to a fraction powerwith even denominator")
            # avoid having a 0 derivative when the power is less than 1
            if self.val == 0 and other < 1:
                raise ValueError("Error: Attempted to find derivative at 0 when power is less than 1")

            f = self.val ** other
            f_prime = other * self.val ** (other - 1)
            return DualNumbers(f, self.derv * f_prime)
        
    def __rpow__(self, other):
        r"""A method to perform the reverse power operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object
        
        Returns
        -------
        A DualNumbers object as the result of the reverse power operation
        
        Examples
        --------
        >>> z1 = 1
        >>> z2 = DualNumbers(1, -1)
        >>> print(z1 ** z2)
        Values: 1, Derivatives: -0.0

        >>> x1 = 1
        >>> x2 = DualNumbers(1, np.array([0, -1]))
        >>> print(x1 ** x2)
        Values: 1, Derivatives: [0.  -0.]
        """
        f = other ** self.val
        f_prime = (other ** self.val) * self.derv * np.log(other)
        return DualNumbers(f, f_prime)
    
    def __neg__(self):
        r"""A method to perform the negation operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        None
        
        Returns
        -------
        A DualNumbers object as the result of the negation operation
        
        Examples
        --------
        >>> z = DualNumbers(1, -1)
        >>> print(-z)
        Values: -1, Derivatives: 1

        >>> x = DualNumbers(1, np.array([-1, 0]))
        >>> print(-x)
        Values: -1, Derivatives: [1  0]
        """
        return DualNumbers(-1 * self.val, -1 * self.derv)
    
    def __eq__(self, other):
        r"""A method to check whether the DualNumbers objects are equal to each other
        
        Parameters
        ----------
        other: A DualNumbers object
        
        Returns
        -------
        A tuple of boolean variables where the first variable indicates whether the function values are equal and the second
        variable indicates if the derivative values are equal
        
        Examples
        --------
        >>> z1 = DualNumbers(1, 10)
        >>> z2 = DualNumbers(1, 10)
        >>> print(z1 == z2)
        (True, True)

        >>> z1 = DualNumbers(1, 10)
        >>> z2 = DualNumbers(2, 10)
        >>> print(z1 == z2)
        (False, True)

        >>> z1 = DualNumbers(2, 1)
        >>> z2 = DualNumbers(1, 2)
        >>> print(z1 == z2)
        (False, False)
        """
        # check if DualNumbers values are equal
        try:
            is_val_eq = all(self.val == other.val)
        except TypeError:
            is_val_eq = True if self.val == other.val else False

        # check if DualNumbers derivatives are equal
        try:
            is_derv_eq = all(self.derv == other.derv)
        except TypeError:
            is_derv_eq = True if self.derv == other.derv else False
        return is_val_eq, is_derv_eq
    
    def __ne__(self, other):
        r"""A method to check whether the DualNumbers objects are not equal to each other
        
        Parameters
        ----------
        other: A DualNumbers object
        
        Returns
        -------
        A tuple of boolean variables where the first variable indicates if the function values are not equal and
        the second variable indicates if the derivative values are not equal
        
        Examples
        --------
        >>> z1 = DualNumbers(1, 10)
        >>> z2 = DualNumbers(1, 10)
        >>> print(z1 != z2)
        (False, False)

        >>> z1 = DualNumbers(1, 10)
        >>> z2 = DualNumbers(2, 10)
        >>> print(z1 != z2)
        (True, False)

        >>> z1 = DualNumbers(2, 1)
        >>> z2 = DualNumbers(1, 2)
        >>> print(z1 != z2)
        (True, True)
        """
        # check if DualNumbers values are not equal
        try:
            is_val_eq = all(self.val != other.val)
        except TypeError:
            is_val_eq = True if self.val != other.val else False

        # check if DualNumbers derivatives are not equal
        try:
            is_derv_eq = all(self.derv != other.derv)
        except TypeError:
            is_derv_eq = True if self.derv != other.derv else False

        return is_val_eq, is_derv_eq
    def sqrt(self):
        """
        method to compute the value and derivative of the square root function of the DualNumbers objects

        Parameters
        ----------
        None

        Returns
        -------
        A dualnumbers object that contains the value and derivative of the square root function

        Examples
        --------
        # square root of variable with scalar derivative
        >>> x = dual_number(4, -1)
        >>> print(x.sqrt())
        Values:2.0, Derivatives:-0.5

        # square root of variable with vector derivative
        >>> x = val_derv(1, np.array([-1, 0]))
        >>> print(x.sqrt())
        Values:1.0, Derivatives:[-0.5  0. ]

        """
        return self.__pow__(0.5)

    def log(self, base=None):
        """
        method to compute the value and derivative of logarthmic of the DualNumbers objects

        Parameters
        ----------
        base: A float object that represents the base of the logarithm (default logarithmic base is None)

        Returns
        -------
        A val_derv object that contains the value and derivative of the logarithmic function

        Raises
        ------
        ValueError
            If self.val is less than or equal to zero
            If input base is less than or equal to zero
            If input base is equal to one

        Examples
        --------
        # ValueError if self.val is less than or equal to zero
        >>> x = DualNumbers(0, -1)
        >>> print(x.log())
        ValueError: ERROR: Value for log should be greater than 0

        # ValueError if input base is less than or equal to zero or equal to 1
        >>> x = DualNumbers(1, 1)
        >>> print(x.log(base = -1))
        ValueError: ERROR: LOG base should be greater than 0 and not equal to 1

        # logarithm of variable with scalar derivative
        >>> x = DualNumbers(1, -1)
        >>> print(x.log())
        Values:0.0, Derivatives:-1.0

        # logarithm of variable with vector derivative
        >>> x = DualNumbers(1, np.array([-1, 0]))
        >>> print(x.log())
        Values:0.0, Derivatives:[-1.  0.]
        """

        # ensure the value is greater than zero so that log is correctly defined
        if self.val <= 0:
            raise ValueError("ERROR: Value for log should be greater than 0")
        # if the default base is used, proceed with default base numpy log funtion
        if base is None:
            return DualNumbers(np.log(self.val), self.derv / self.val)
        # ensure the user specifies a valid base before computing the log value and derivative
        else:
            if base <= 0 or base == 1:
                raise ValueError("ERROR: LOG base should be greater than 0 and not equal to 1")
            return DualNumbers(np.log(self.val) / np.log(base), self.derv / (self.val * np.log(base)))

    def exp(self):
        """
        method to compute the value and derivative of exponential function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the exponential function

        Examples
        --------
        # exponential of variable with scalar derivative
        >>> x = val_derv(0, -2)
        >>> print(x.exp())
        Values:1.0, Derivatives:-2.0

        # exponential of variable with vector derivative
        >>> x = val_derv(0, np.array([-1, 0]))
        >>> print(x.exp())
        Values:1.0, Derivatives:[-1.  0.]
        """
        # compute the value and derivative of the exponential function for any input
        return DualNumbers(np.exp(self.val), self.derv * np.exp(self.val))

    def sin(self):
        """
        method to compute the value and derivative of the sine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the sine function

        Examples
        --------
        # sine of variable with scalar derivative
        >>> x = val_derv(0, -1)
        >>> print(x.sin())
        Values:0.0, Derivatives:-1.0

        # sine of variable with vector derivative
        >>> x = val_derv(0, np.array([-1, 0]))
        >>> print(x.sin())
        Values:0.0, Derivatives:[-1.  0.]
        """
        # compute the value and derivative of the sine function for any input
        return DualNumbers(np.sin(self.val), self.derv * np.cos(self.val))

    def cos(self):
        """
        method to compute the value and derivative of the cosine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the cosine function

        Examples
        --------
        # cosine of variable with scalar derivative
        >>> x = val_derv(0, -1)
        >>> print(x.cos())
        Values:1.0, Derivatives:0.0

        # cosine of variable with vector derivative
        >>> x = val_derv(0, np.array([-1, 0]))
        >>> print(x.cos())
        Values:1.0, Derivatives:[ 0. -0.]

        """
        # compute the value and derivative of the cosine function for any input
        return DualNumbers(np.cos(self.val), -self.derv * np.sin(self.val))

    def tan(self):
        """
        method to compute the value and derivative of the tangent function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the tangent function

        Raises
        ------
        ValueError if input is an odd multiple of pi/2

        Examples
        --------
        # tangent of variable with scalar derivative
        >>> x = val_derv(0, -1)
        >>> print(x.tan())
        Values:0.0, Derivatives:-1.0

        # tangent of variable with vector derivative
        >>> x = val_derv(0, np.array([-1, 0]))
        >>> print(x.tan())
        Values:0.0, Derivatives:[-1.  0.]

        # ValueError if input is an odd multiple of pi/2
        >>> x = val_derv(np.pi / 2, -1)
        >>> print(x.tan())
        ValueError: ERROR: Input to tan should not be an odd mutiple of pi/2

        """

        # ensure the user does not input an odd multiple of pi divided by 2
        if (self.val / (np.pi / 2)) % 2 == 1:
            raise ValueError("ERROR: Input to tan should not be an odd mutiple of pi/2")

        # compute the value and derivative of the tangent function for a valid input
        return DualNumbers(np.tan(self.val), self.derv * 1 / np.cos(self.val) ** 2)

    def sinh(self):
        """
        method to compute the value and derivative of the hyperbolic sine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the hyperbolic sine function

        Examples
        --------
        # hyperbolic sine of variable with scalar derivative
        >>> x = val_derv(0, -1)
        >>> print(x.sinh())
        Values:0.0, Derivatives:-1.0

        # hyperbolic sine of variable with vector derivative
        >>> x = val_derv(0, np.array([-1, 0]))
        >>> print(x.sinh())
        Values:0.0, Derivatives:[-1.  0.]

        """
        return DualNumbers(np.sinh(self.val), self.derv * np.cosh(self.val))

    def cosh(self):
        """
        method to compute the value and derivative of the hyperbolic cosine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the hyperbolic cosine function

        Examples
        --------
        # hyperbolic cosine of variable with scalar derivative
        >>> x = val_derv(0, -1)
        >>> print(x.cosh())
        Values:1.0, Derivatives:-0.0

        # hyperbolic cosine of variable with vector derivative
        >>> x = val_derv(0, np.array([-1, 0]))
        >>> print(x.cosh())
        Values:1.0, Derivatives:[-0.  0.]

        """

        # compute the value and derivative of the hyperbolic cosine function for any input
        return DualNumbers(np.cosh(self.val), self.derv * np.sinh(self.val))

    def tanh(self):
        """
        method to compute the value and derivative of the hyperbolic tangent function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the hyperbolic tangent function

        Examples
        --------
        # hyperbolic tangent of variable with scalar derivative
        >>> x = val_derv(0, -1)
        >>> print(x.tanh())
        Values:0.0, Derivatives:-1.0

        # hyperbolic tangent of variable with vector derivative
        >>> x = val_derv(0, np.array([-1, 0]))
        >>> print(x.tanh())
        Values:0.0, Derivatives:[-1.  0.]

        """

        # compute the value and derivative of the hyperbolic tangent function for any input
        return DualNumbers(np.tanh(self.val), self.derv * 1 / (np.cosh(self.val) ** 2))

    def arcsin(self):
        """
        method to compute the value and derivative of the inverse sine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the inverse sine function

        Raises
        ------
        ValueError if input is not contained within the interval [-1,1]

        Examples
        --------
        # inverse sine of variable with scalar derivative
        >>> x = val_derv(0, -1)
        >>> print(x.arcsin())
        Values:0.0, Derivatives:-1.0

        # inverse sine of variable with vector derivative
        >>> x = val_derv(0, np.array([-1, 0]))
        >>> print(x.arcsin())
        Values:0.0, Derivatives:[-1.  0.]

        # ValueError for input outside of the interval -1 to 1
        >>> x = val_derv(2, -1)
        >>> print(x.arcsin())
        ValueError: ERROR: Input to arcsin() should be between -1 and 1

        """

        # ensure the user passes in an input between -1 and 1
        if -1 >= self.val or self.val >= 1:
            raise ValueError("ERROR: Input to arcsin() should be between -1 and 1")
        # compute the value and derivative of the inverse sine function for a valid input
        return DualNumbers(np.arcsin(self.val), self.derv * 1 / (1 - self.val ** 2) ** 0.5)

    def arccos(self):
        """
        method to compute the value and derivative of the inverse cosine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the inverse cosine function

        Raises
        ------
        ValueError will be raised if input is not contained within the interval [-1,1]

        Examples
        --------
        # inverse cosine of variable with scalar derivative
        >>> x = val_derv(0, -1)
        >>> print(x.arccos())
        Values:1.5707963267948966, Derivatives:1.0

        # inverse cosine of variable with vector derivative
        >>> x = val_derv(0, np.array([-1, 0]))
        >>> print(x.arccos())
        Values:1.5707963267948966, Derivatives:[ 1. -0.]

        # ValueError for input outside of the interval -1 to 1
        >>> x = val_derv(2, -1)
        >>> print(x.arccos())
        ValueError: ERROR: Input to arccos() should be between -1 and 1

        """

        # ensure the user passes in an input between -1 and 1
        if -1 >= self.val or self.val >= 1:
            raise ValueError("ERROR: Input to arccos() should be between -1 and 1")
        # compute the value and derivative of the inverse cosine function for a valid input
        return DualNumbers(np.arccos(self.val),  - self.derv / (1 - self.val ** 2) ** 0.5)

    def arctan(self):
        """
        method to compute the value and derivative of the inverse tangent function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the inverse tangent function

        Examples
        --------
        # inverse tangent of variable with scalar derivative
        >>> x = val_derv(1, -1)
        >>> print(x.arctan())
        Values:0.7853981633974483, Derivatives:-0.5

        # inverse tangent of variable with vector derivative
        >>> x = val_derv(1, np.array([-1, 0]))
        >>> print(x.arctan())
        Values:0.7853981633974483, Derivatives:[-0.5  0. ]

        """
        # compute the value and derivative of the inverse tangent function for a valid input
        return DualNumbers(np.arctan(self.val), self.derv / (1 + self.val ** 2))

    def logistic(self):
        """
        method to compute the value and derivative of the logistic function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the logistic function

        Examples
        --------
        # logistic of variable with scalar derivative
        >>> x = val_derv(1, -1)
        >>> print(x.logistic())
        Values:0.7310585786300049, Derivatives:-0.19661193324148188

        # logistic of variable with vector derivative
        >>> x = val_derv(1, np.array([-1, 0]))
        >>> print(x.logistic())
        Values:0.7310585786300049, Derivatives:[-0.19661193 -0.        ]

        """

        return 1 / ((-self).exp() + 1)