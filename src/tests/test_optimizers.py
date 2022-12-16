# Authors: Wenxu Liu, Queenie Luo, Guangya Wan, Mengyao Zheng, Di Zhen          #
# Course: AC207/CS107                                                           #
# File: test_optimizers.py                                                      #
# Description: Test the optimizer class functionality by creating a testing     #
# testing suite using pytest                                                    #
#################################################################################

import pytest
import numpy as np
from AD_fbi.optimizers import Optimizer

x = 1
f_x = lambda x: (x + 1) ** 2
xy = np.array([0, 0])
f_xy = lambda x, y: (x - 1) ** 2 + (y + 1) ** 2

class TestOptimizer:
    # test univariate function for momentum optimizer
    def test_uni_momentum(self):
        opt_time, val, curr_val = Optimizer.momentum(x, f_x, 1000)
        assert val == pytest.approx(0)
        assert curr_val == pytest.approx(-1)
        
    # test multivariate function for momentum optimizer
    def test_multi_momentum(self):
        opt_time, val, curr_val = Optimizer.momentum(xy, f_xy, 1000)
        assert val == pytest.approx(0)
        assert curr_val[0] == pytest.approx(1)
        assert curr_val[1] == pytest.approx(-1)
        
    # test invalid hyper parameters for momentum
    def test_momentum_beta_invalid(self):
        try:
            Optimizer.momentum(xy, f_xy, 1000, beta = 1.5)
            assert False
        except ValueError:
            assert True
            
    # test invalid hyper parameters for momentum
    def test_momentum_alpha_invalid(self):
        try:
            Optimizer.momentum(xy, f_xy, 1000, alpha = 10)
            assert False
        except ValueError:
            assert True

            
    # test univariate function for gradient descent optimizer
    def test_uni_gradient_descent(self):
        opt_time, val, curr_val = Optimizer.gradient_descent(x, f_x, 10000)
        assert val == pytest.approx(0)
        assert curr_val == pytest.approx(-1)
        
    # test multivariate function for gradient descent optimizer
    def test_multi_gradient_descent(self):
        opt_time, val, curr_val = Optimizer.gradient_descent(xy, f_xy, 10000)
        assert val == pytest.approx(0)
        assert curr_val[0] == pytest.approx(1)
        assert curr_val[1] == pytest.approx(-1)
        
            
    # test invalid hyper parameters for gradient descent
    def test_gradient_descent_alpha_invalid(self):
        try:
            Optimizer.gradient_descent(xy, f_xy, 1000, alpha = 10)
            assert False
        except ValueError:
            assert True


    # test univariate function for ADAGRAD optimizer
    def test_uni_ADAGRAD(self):
        opt_time, val, curr_val = Optimizer.ADAGRAD(x, f_x, 10000, alpha=0.1)
        assert val == pytest.approx(0)
        assert curr_val == pytest.approx(-1)
        
    # test multivariate function for ADAGRAD optimizer
    def test_multi_ADAGRAD(self):
        opt_time, val, curr_val = Optimizer.ADAGRAD(xy, f_xy, 10000, alpha=0.1)
        assert val == pytest.approx(0)
        assert curr_val[0] == pytest.approx(1)
        assert curr_val[1] == pytest.approx(-1)
            
    # test invalid alpha hyperparameters for ADAGRAD
    def test_ADAGRAD_alpha_invalid(self):
        try:
            Optimizer.ADAGRAD(xy, f_xy, 1000, alpha = 10)
            assert False
        except ValueError:
            assert True
    

