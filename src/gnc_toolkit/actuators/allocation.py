import numpy as np
from abc import ABC, abstractmethod

class ControlAllocator(ABC):
    """
    Abstract base class for control allocation.
    Maps a desired generalized force (force/torque) to actuator commands.
    """
    def __init__(self, actuator_matrix):
        """
        Args:
            actuator_matrix (np.array): Matrix A (m x n) where Y = A * u.
                                       m = degrees of freedom (e.g. 3 for torque).
                                       n = number of actuators.
        """
        self.A = np.array(actuator_matrix)
        self.m, self.n = self.A.shape

    @abstractmethod
    def allocate(self, desired_output):
        """
        Args:
            desired_output (np.array): Desired force/torque (m,).
            
        Returns:
            np.array: Actuator commands (n,).
        """
        pass

class PseudoInverseAllocator(ControlAllocator):
    """
    Standard pseudo-inverse allocator: u = A^T * (A * A^T)^-1 * Y.
    Minimizes the L2-norm of the actuator commands (energy optimal).
    """
    def __init__(self, actuator_matrix):
        super().__init__(actuator_matrix)
        # Precompute pseudo-inverse: A+ = A^T (A A^T)^-1
        self.A_pinv = np.linalg.pinv(self.A)

    def allocate(self, desired_output):
        return self.A_pinv @ np.array(desired_output)

class SingularRobustAllocator(ControlAllocator):
    """
    Singular Robust Inverse (SR-Inverse) for CMGs.
    Adds a regularization term to avoid high gimbal rates near singularities.
    u = A^T * (A * A^T + lambda * I)^-1 * Y
    """
    def __init__(self, actuator_matrix, epsilon=0.01, lambda0=0.01):
        super().__init__(actuator_matrix)
        self.epsilon = epsilon
        self.lambda0 = lambda0

    def allocate(self, desired_output, A_current=None):
        """
        Args:
            desired_output (np.array): Desired torque.
            A_current (np.array): Current Jacobian matrix if time-varying.
        """
        A = A_current if A_current is not None else self.A
        
        # Calculate manipulability measure (determinant of AA^T)
        m_measure = np.linalg.det(A @ A.T)
        
        # Regularization parameter lambda
        if m_measure < self.epsilon:
            lam = self.lambda0 * (1 - m_measure/self.epsilon)**2
        else:
            lam = 0.0
            
        # SR-Inverse: A^T (A A^T + lam * I)^-1
        m = A.shape[0]
        inv_term = np.linalg.inv(A @ A.T + lam * np.eye(m))
        u = A.T @ inv_term @ desired_output
        return u

class NullMotionManager:
    """
    Manages null-motion to redistribute actuator states without affecting net output.
    u_net = u_alloc + u_null
    u_null = (I - A+ * A) * z
    """
    def __init__(self, actuator_matrix):
        self.A = np.array(actuator_matrix)
        self.m, self.n = self.A.shape
        self.I = np.eye(self.n)

    def get_null_projection(self, A_current=None):
        """Returns the null-space projection matrix (I - A+ * A)."""
        A = A_current if A_current is not None else self.A
        A_pinv = np.linalg.pinv(A)
        return self.I - A_pinv @ A

    def apply_null_command(self, u_base, z, A_current=None):
        """
        Adds null-motion component to base command.
        
        Args:
            u_base (np.array): Base actuator commands (e.g. from pseudo-inverse).
            z (np.array): Desired secondary goal (e.g. move gimbals away from limits).
            A_current (np.array): Current Jacobian.
        """
        P = self.get_null_projection(A_current)
        return u_base + P @ z
