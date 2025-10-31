"""
Defintions for problem 0
"""

import numpy as np
import scipy.integrate
from scipy.integrate import DenseOutput
from scipy.interpolate import interp1d
from warnings import warn


class ForwardEulerOutput(DenseOutput):
    """
    Dense output for Forward Euler method.
    """
    def __init__(self, t_old, t, y_old, y):
        super().__init__(t_old, t)
        self.y_old = y_old
        self.y = y
        
        # Create interpolation function
        self.interpolant = interp1d(
            [t_old, t], 
            np.vstack([y_old, y]).T,
            kind='linear',
            axis=1,
            bounds_error=False,
            fill_value=(y_old, y)
        )
    
    def _call_impl(self, t):
        return self.interpolant(t)



class ForwardEuler(scipy.integrate.OdeSolver):
    """
    Forward Euler method ODE solver.
    
    y_{n+1} = y_n + h * f(t_n, y_n)
    
    Parameters
    ----------
    fun : callable
        Right-hand side of the system: dy/dt = fun(t, y)
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it.
    vectorized : bool
        Whether `fun` is implemented in a vectorized fashion.
    support_complex : bool, optional
        Whether integration in a complex domain should be supported.
        Generally determined by a derived solver class capabilities.
        Default is False.
    h : float, optional
        Step size. If not provided, defaults to (t_bound - t0) / 100.
    """
    
    def __init__(self, fun, t0, y0, t_bound, vectorized=False, support_complex=False, h=None):
        # Initialize using super()
        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex)
        
        # Set direction
        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
        
        # Set step size
        if h is None:
            self.h = (t_bound - t0) / 100.0
        else:
            if h <= 0:
                raise ValueError("Step size h should be positive")
            self.h = h
        
        # Store previous values for dense output
        self.t_old = t0
        self.y_old = y0.copy()
        
        # Jacobian statistics remain at 0
        self.njev = 0
        self.nlu = 0


    def _step_impl(self):
        self.t_old = self.t
        self.y_old = self.y.copy()

        f = self.fun(self.t, self.y)

        # Forward Euler update
        t_new = self.t + self.h

        # Check the boundary
        if t_new > self.t_bound:
            # Adjust step to exactly reach t_bound
            h_actual = self.t_bound - self.t
            t_new = self.t_bound
        else:
            h_actual = self.h
        
        # Update state
        y_new = self.y + h_actual * f
        self.t = t_new
        self.y = y_new

        return True, None
    
    def _dense_output_impl(self):
        """
        Return dense output for the last successful step.
        """
        return ForwardEulerOutput(self.t_old, self.t, self.y_old, self.y)

