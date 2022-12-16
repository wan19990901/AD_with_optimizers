import pytest
import numpy as np

from AD_fbi.forward_mode import ForwardMode


##initialize ForwardMode objects

func1 = lambda x: x
fm1 = ForwardMode(1, func1)
fm2 = ForwardMode(1, func1, -1)

func2 = lambda x: x**2 + 2
fm3 = ForwardMode(2, func2)
fm4 = ForwardMode(3, func2, -2)

func3 = lambda x, y: 2*x + y
fm5 = ForwardMode(np.array([1, 2]), func3)
fm6 = ForwardMode(np.array([1, 1]), func3, [2, -1])



func4 = lambda x, y: (2*x + y, 3*y + x**2)
fm7 = ForwardMode(np.array([1, 2]), func4)
fm8 = ForwardMode(np.array([1, 1]), func4, [2, -1])




class TestForwardMode:
    """Test class for ForwardMode module"""
    
    # test attribute initialization
    def test_init(self):
        assert fm1.inputs == 1
        assert fm1.functions == func1
        assert fm1.seed == 1
        
        assert fm2.inputs == 1
        assert fm2.functions == func1
        assert fm2.seed == -1
        
        assert fm3.inputs == 2
        assert fm3.functions == func2
        assert fm3.seed == 1
        
        assert fm4.inputs == 3
        assert fm4.functions == func2
        assert fm4.seed == -2
        
        assert (fm5.inputs == np.array([1, 2])).all()
        assert fm5.functions == func3
        assert (fm5.seed == np.ones(2)).all()
        
        assert (fm6.inputs == np.array([1, 1])).all()
        assert fm6.functions == func3
        assert fm6.seed == [2, -1]
        
        
        assert (fm7.inputs == np.array([1, 2])).all()
        assert fm7.functions == func4
        assert (fm7.seed == np.ones(2)).all()
        
        assert (fm8.inputs == np.array([1, 1])).all()
        assert fm8.functions == func4
        assert fm8.seed == [2, -1]
        
        
    def test_get_fx_value(self):
        assert fm1.get_fx_value() == 1
        assert fm2.get_fx_value() == 1
        assert fm3.get_fx_value() == 6
        assert fm4.get_fx_value() == 11
        
        assert fm5.get_fx_value() == 4
        assert fm6.get_fx_value() == 3
        assert (fm7.get_fx_value() == np.array([4., 7.])).all()
        assert (fm8.get_fx_value() == np.array([3., 4.])).all()
        
    def test_get_derivative(self):
        assert fm1.get_derivative() == 1 
        assert fm2.get_derivative() == -1 
        assert fm3.get_derivative() == 4
        assert fm4.get_derivative() == -12
        
        assert (fm5.get_derivative() == np.array([2., 1.])).all()
        assert (fm6.get_derivative() == np.array([4., -1.])).all()
        assert (fm7.get_derivative() == np.array([[2., 1.], [2., 3.]])).all()
        assert (fm8.get_derivative() == np.array([[4., -1.],[4., -3.]])).all()
        
    def test_calculate_dual_number(self):
        assert fm1.calculate_dual_number() == (1, 1)
        assert fm2.calculate_dual_number() == (1, -1)
        assert fm3.calculate_dual_number() == (6, 4)
        assert fm4.calculate_dual_number() == (11, -12)
        assert fm5.calculate_dual_number()[0] == 4
        assert (fm5.calculate_dual_number()[1] == [2., 1.]).all()
        
        assert fm6.calculate_dual_number()[0] == 3
        assert (fm6.calculate_dual_number()[1] == [4., -1.]).all()
        
        assert (fm7.calculate_dual_number()[0] == np.array([4., 7.])).all()
        assert (fm7.calculate_dual_number()[1] == np.array([[2., 1.], [2., 3.]])).all()
        
        assert (fm8.calculate_dual_number()[0] == np.array([3., 4.])).all()
        assert (fm8.calculate_dual_number()[1] == np.array([[4., -1.],[4., -3.]])).all()

        
        
