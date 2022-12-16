import pytest
import numpy as np
from AD_fbi.dual_number import DualNumbers, is_numeric

z0 = DualNumbers(1, 2)
z1 = DualNumbers(1, -1)
z2 = DualNumbers(2.2, 0)
z3 = DualNumbers(2, 6.036)
z4 = DualNumbers(3, 100000)
z5 = DualNumbers(0.5, -2)
z6 = DualNumbers(0, 9)

x1 = DualNumbers(1, np.array([0, 2]))
x2 = DualNumbers(1, np.array([-1, 0]))
x3 = DualNumbers(1, np.array([0, -1]))
x4 = DualNumbers(0, np.array([-2, -1]))


class TestDualNumbers:
    """Test class for DualNumbers module"""
    def test_val_derv_repr(self):
        assert "Values: 1, Derivatives: -1" == z1.__repr__()
        assert "Values: 3, Derivatives: 100000" == z4.__repr__()
        assert "Values: 1, Derivatives: [ 0 -1]" == x3.__repr__()

    # test is_numeric function
    def test_is_numeric(self):
        x1, x2, x3, x4 = 6, 'a', 'dual', ['n', 'u', 'm', 'b', 'e', 'r']
        x5, x6, x7, x8 = [3, 's', 9, 'i', 'x'], [1, 2, 3, 4, 5], \
                                   ["test", 2, 7, 'c'], [0.2, 0.5, -6]

        assert is_numeric(x1) == True
        assert is_numeric(x2) == False
        assert is_numeric(x3) == False
        assert is_numeric(x4) == False
        assert is_numeric(x5) == False
        assert is_numeric(x6) == True
        assert is_numeric(x7) == False
        assert is_numeric(x8) == True
        
    # test attribute initialization
    def test_init(self):
        assert z0.val == 1
        assert z0.derv == 2

        assert x1.val == 1
        assert x1.derv[0] == 0
        assert x1.derv[1] == 2

    def test_init_Error(self):
        try:
            z_err = DualNumbers('k', 2)
            assert False
        except TypeError:
            assert True
        try:
            z_err = DualNumbers(2, [1,'k'])
            assert False
        except TypeError:
            assert True
            
    # test attribute setter
    def test_setter(self):
        z0.val = 2
        z0.derv = 1
        assert z0.val == 2
        assert z0.derv == 1
    def test_setter_Error(self):
        try:
            z0.val = 'o'
            assert False
        except TypeError:
            assert True
        try:
            z0.derv = ['']
            assert False
        except TypeError:
            assert True
    
    # test scalar addition operation
    def test_scalar_add(self):
        out = z1 + z2
        assert out.val == 3.2
        assert out.derv == -1
        out = z1 + 2
        assert out.val == 3
        assert out.derv == -1
        out = 2 + z1
        assert out.val == 3
        assert out.derv == -1
    
    # test array addition operation
    def test_1d_add(self):
        out = x1 + x2
        assert out.val == 2 
        assert all(out.derv == [-1, 2])
        out = x1 + 2
        assert out.val == 3
        assert all(out.derv == [0, 2])
        out = 2 + x1
        assert out.val == 3
        assert all(out.derv == [0, 2])
    
    # test scalar subtraction operation
    def test_scalar_sub(self):
        out = z1 - z2
        assert out.val == 1-2.2
        assert out.derv == -1
        out = z1 - 2
        assert out.val == -1
        assert out.derv == -1
        out = 2 - z1
        assert out.val == 1
        assert out.derv == 1
    
    # test array subtraction operation
    def test_1d_sub(self):
        out = x2 - x1
        assert out.val == 0
        assert all(out.derv == [-1, -2])
        out = x1 - 2
        assert out.val == -1
        assert all(out.derv == [0, 2])
        out = 2 - x1
        assert out.val == 1
        assert all(out.derv == [0, -2])
    
    # test scalar multiplization operation
    def test_scalar_mul(self):
        out = z1 * z2
        assert out.val == 2.2
        assert out.derv == -2.2
        out = z1 * 2
        assert out.val == 2
        assert out.derv == -2
        out = 2 * z1
        assert out.val == 2
        assert out.derv == -2
        
    # test array multiplication operation
    def test_1d_mul(self):
        out = x1 * x2
        assert out.val == 1
        assert all(out.derv == [-1, 2])
        out = x1 * 2
        assert out.val == 2
        assert all(out.derv == [0, 4])
        out = 2 * x1
        assert out.val == 2
        assert all(out.derv == [0, 4])

    # test scalar division operation
    def test_scalar_div(self):
        out = z1 / z3
        assert out.val == 0.5
        assert out.derv == -8.036 / 4
        out = z1 / 2
        assert out.val == 0.5
        assert out.derv == -0.5
        out = 2 / z1
        assert out.val == 2
        assert out.derv == 2
        
    # test array division operation
    def test_1d_div(self):
        out = x2 / x1
        assert out.val == 1
        assert all(out.derv == [-1, -2])
        out = x1 / 2
        assert out.val == 0.5
        assert all(out.derv == [0, 1])
        out = 2 / x1
        assert out.val == 2
        assert all(out.derv == [0, -4])
    
    # test error handling in scalar division operation
    def test_scalar_div_zeroError(self):
        with pytest.raises(ZeroDivisionError) as e:
            z1 / 0
        with pytest.raises(ZeroDivisionError) as e:
            z1 / z6
        with pytest.raises(ZeroDivisionError) as e:
            4 / z6
    
    # test error handling in array division operation
    def test_1d_div_zeroError(self):
        with pytest.raises(ZeroDivisionError) as e:
            x2 / 0
        with pytest.raises(ZeroDivisionError) as e:
            x2 / x4
        with pytest.raises(ZeroDivisionError) as e:
            4 / x4
    
    # test scalar power operation
    def test_scalar_pow(self):
        out = z5 ** z4
        assert out.val == 0.5 ** 3
        assert out.derv == (0.5 ** 3) * (100000 * np.log(0.5) + (-2) * 3 / 0.5)
        out = z4 ** 2
        assert out.val == 3 ** 2
        assert out.derv == (3 ** 2) * (100000 * 2 / 3)
        out = 2 ** z4
        assert out.val == 2 ** 3
        assert out.derv == (2 ** 3) * (100000 * np.log(2))
    def test_pow_error(self):
        temp = DualNumbers(0,0)
        with pytest.raises(ValueError) as e:
            temp**DualNumbers(0.5,0)
        
    # test array power operation
    def test_1d_pow(self):
        out = x2 ** x1
        assert out.val == 1
        assert all(out.derv == [-1, 0])
        
    # test scalar negation operation
    def test_scalar_neg(self):
        out = -z1
        assert out.val == -1
        assert out.derv == 1
        
    # test array negation operation
    def test_1d_neg(self):
        out = -x1
        assert out.val == -1
        assert all(out.derv == [0, -2])
        
    # test scalar equality operation
    def test_scalar_eq(self):
        tmp = DualNumbers(1, -1)
        assert (tmp == z1) == (True, True)
        tmp = DualNumbers(1, 0)
        assert (tmp == z1) == (True, False)
        tmp = DualNumbers(2, -1)
        assert (tmp == z1) == (False, True)
        tmp = DualNumbers(2, -2)
        assert (tmp == z1) == (False, False)
        
    # test array equality operation
    def test_1d_eq(self):
        tmp = DualNumbers(1, np.array([0, 2]))
        assert (tmp == x1) == (True, True)
        tmp = DualNumbers(1, np.array([0, 1]))
        assert (tmp == x1) == (True, False)
        tmp = DualNumbers(0, np.array([0, 2]))
        assert (tmp == x1) == (False, True)
        tmp = DualNumbers(0, np.array([1, 3]))
        assert (tmp == x1) == (False, False)
    
    # test scalar inequality operation
    def test_scalar_ne(self):
        tmp = DualNumbers(2,-2)
        assert (tmp != z1) == (True, True)
        tmp = DualNumbers(2,-1)
        assert (tmp != z1) == (True, False)
        tmp = DualNumbers(1,-2)
        assert (tmp != z1) == (False, True)
        tmp = DualNumbers(1,-1)
        assert (tmp != z1) == (False, False)
    
    # test array inequality operation
    def test_1d_ne(self):
        tmp = DualNumbers(0, np.array([1, 3]))
        assert (tmp != x1) == (True, True)
        tmp = DualNumbers(0, np.array([0, 2]))
        assert (tmp != x1) == (True, False)
        tmp = DualNumbers(1, np.array([1, 3]))
        assert (tmp != x1) == (False, True)
        tmp = DualNumbers(1, np.array([0, 2]))
        assert (tmp != x1) == (False, False) 
    
    def test_sin_scalar(self):
        sin_scalar = z3.sin()
        assert pytest.approx(np.sin(2)) == sin_scalar.val
        assert pytest.approx(np.cos(2) * 6.036) == sin_scalar.derv
        sin_scalar_const = z2.sin()
        assert pytest.approx(np.sin(2.2)) == sin_scalar_const.val
        assert pytest.approx(0) == sin_scalar_const.derv

    # test scalar cosine operation
    def test_cos_scalar(self):
        cos_scalar = z3.cos()
        assert pytest.approx(np.cos(2)) == cos_scalar.val
        assert pytest.approx((-np.sin(2) * 6.036)) == cos_scalar.derv
        cos_scalar_const = z2.cos()
        assert pytest.approx(np.cos(2.2)) == cos_scalar_const.val
        assert pytest.approx(0) == cos_scalar_const.derv

    # test scalar tangent operation
    def test_tan_scalar(self):
        tan_scalar = z3.tan()
        assert pytest.approx(np.tan(2)) == tan_scalar.val
        assert pytest.approx(6.036 / (np.cos(2) ** 2)) == tan_scalar.derv
        tan_scalar_const = z2.tan()
        assert pytest.approx(np.tan(2.2)) == tan_scalar_const.val
        assert pytest.approx(0) == tan_scalar_const.derv

    # test vector tangent operation

    # test erorr handling in scalar tangent operation
    def test_tan_scalar_error(self):
        with pytest.raises(ValueError) as e:
            var = DualNumbers(3 * np.pi / 2, -1)
            var.tan()

    # test scalar hyperbolic sine operation
    def test_sinh_scalar(self):
        sinh_scalar = z3.sinh()
        assert pytest.approx(np.sinh(2)) == sinh_scalar.val
        assert pytest.approx(np.cosh(2) * 6.036) == sinh_scalar.derv
        sinh_scalar_const = z2.sinh()
        assert pytest.approx(np.sinh(2.2)) == sinh_scalar_const.val
        assert pytest.approx(0) == sinh_scalar_const.derv

    # test scalar hyperbolic cosine operation
    def test_cosh_scalar(self):
        cosh_scalar = z3.cosh()
        assert pytest.approx(np.cosh(2)) ==  cosh_scalar.val
        assert pytest.approx(np.sinh(2) * 6.036) == cosh_scalar.derv
        cosh_scalar_const = z2.cosh()
        assert pytest.approx(np.cosh(2.2)) ==  cosh_scalar_const.val
        assert pytest.approx(0) ==  cosh_scalar_const.derv

    # test scalar hyperbolic tangent operation
    def test_tanh_scalar(self):
        tanh_scalar = z3.tanh()
        assert pytest.approx(np.tanh(2)) == tanh_scalar.val
        assert pytest.approx((1 - np.tanh(2) ** 2) * 6.036) == tanh_scalar.derv
        tanh_scalar_const = z2.tanh()
        assert pytest.approx(np.tanh(2.2)) ==  tanh_scalar_const.val
        assert pytest.approx(0) == tanh_scalar_const.derv

    # test scalar logarithmic operation
    def test_log_scalar(self):
        log_scalar = z3.log()
        assert pytest.approx(np.log(2)) == log_scalar.val
        assert pytest.approx(6.036 / 2) == log_scalar.derv
        log_scalar_base10 = z3.log(10)
        assert pytest.approx(np.log(2) / np.log(10)) == log_scalar_base10.val
        assert pytest.approx((1 / (2 * np.log(10))) * 6.036) == log_scalar_base10.derv

    # test error handling in scalar logarithmic operation
    def test_log_scalar_invalid_value(self):
        
        with pytest.raises(ValueError) as e:
            z3.log(1)
        with pytest.raises(ValueError) as e:
            z3.log(0)
        with pytest.raises(ValueError) as e:
            z3.log(-1)
   
    def test_exp_scalar(self):
        exp_scalar_const = z2.exp()
        assert pytest.approx(np.exp(2.2)) == exp_scalar_const.val
        assert pytest.approx(0) == exp_scalar_const.derv

    # test scalar inverse sine operation
    def test_scalar_arcsin(self):
        arc_sin_scalar = z5.arcsin()
        assert pytest.approx(np.arcsin(0.5)) == arc_sin_scalar.val
        assert pytest.approx(-2 / np.sqrt((1 - 0.5 ** 2))) == arc_sin_scalar.derv

    # test scalar inverse cosine operation
    def test_scalar_arccos(self):
        arc_cos_scalar = z5.arccos()
        assert pytest.approx(np.arccos(0.5)) == arc_cos_scalar.val
        assert pytest.approx(2 / np.sqrt((1 - 0.5 ** 2))) == arc_cos_scalar.derv


    # test scalar inverse tangent operation
    def test_scalar_arctan(self):
        arc_tan_scalar = z5.arctan()
        assert pytest.approx(np.arctan(0.5)) == arc_tan_scalar.val
        assert pytest.approx(-2 / (1 + 0.5 ** 2)) == arc_tan_scalar.derv

    # test scalar power operation
    def test_scalar_power(self):
        # integer base float power
        power_scalar_int_float = z4 **  z5
        assert pytest.approx(3 ** 0.5) == power_scalar_int_float.val
        assert pytest.approx((3 ** 0.5) * (-2 * np.log(3) + 100000 * 0.5 / 3)) == power_scalar_int_float.derv

        # integer base integer power
        power_scalar_int_int =  z4 ** z4
        assert pytest.approx(3 ** 3) == power_scalar_int_int.val
        assert pytest.approx((3 ** 3) * (100000 * np.log(3) + 100000 * 3 / 3)) == power_scalar_int_int.derv


    # test error handling in scalar power operation
    def test_scalar_power_error(self):
        temp1 = DualNumbers(0, 1)
        temp2 = DualNumbers(-3, -100000)
        with pytest.raises(ValueError) as e:
            temp1 **  z5

        with pytest.raises(ValueError) as e:
            temp2 ** z5

        with pytest.raises(ValueError) as e:
            temp2 ** 0.5

        with pytest.raises(ValueError) as e:
            temp1 ** 0.5

    # test scalar logistic function
    def test_logistic_scalar(self):
        logi_scalar2 = z3.logistic()
        assert pytest.approx((1 / (1 + np.exp(-2)))) == logi_scalar2.val
        assert pytest.approx((6.036 * np.exp(-2) / ((np.exp(-2) + 1) ** 2))) == logi_scalar2.derv

