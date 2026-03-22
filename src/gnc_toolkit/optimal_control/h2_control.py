"""
H2 Optimal Controller (LQG equivalent).
"""

import numpy as np
from .lqg import LQG

class H2Controller(LQG):
    """
    H2 Optimal Controller.
    
    The H2 control problem is equivalent to the LQG problem for a linear system
    subject to white Gaussian noise, where the objective is to minimize the 
    H2 norm of the transfer function from disturbances to regulated outputs.
    
    This class inherits from LQG as the implementation is identical for standard
    state-space systems. It provides an alias and can be extended for more 
    general H2 problems.
    """
    def __init__(self, A, B, C, Q_lqr, R_lqr, Q_lqe, R_lqe, G_lqe=None):
        """
        Initialize H2 Controller (LQG equivalent).
        """
        super().__init__(A, B, C, Q_lqr, R_lqr, Q_lqe, R_lqe, G_lqe)

    def solve(self):
        """
        Solve both LQR and LQE sub-problems.
        """
        self.lqr.solve()
        self.lqe.solve()
        self.K = self.lqr.compute_gain()
        self.L = self.lqe.compute_gain()
        return self.K, self.L
