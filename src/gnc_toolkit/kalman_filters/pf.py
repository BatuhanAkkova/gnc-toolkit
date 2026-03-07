import numpy as np

class ParticleFilter:
    """
    Bootstrap Particle Filter (Sequential Importance Resampling).
    Handles non-Gaussian distributions and highly non-linear models.
    """
    def __init__(self, dim_x, dim_z, num_particles=1000):
        """
        Initialize the Particle Filter.
        dim_x: Dimension of the state vector
        dim_z: Dimension of the measurement vector
        num_particles: Number of particles (N)
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.N = num_particles
        
        # Particles: shape (num_particles, dim_x)
        self.particles = np.zeros((self.N, dim_x))
        # Weights: shape (num_particles,)
        self.weights = np.ones(self.N) / self.N
        
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    def initialize_particles(self, x_mean, P):
        """
        Initialize particles from a Gaussian distribution.
        """
        self.particles = np.random.multivariate_normal(x_mean, P, self.N)
        self.weights = np.ones(self.N) / self.N

    def predict(self, dt, fx, Q=None, **kwargs):
        """
        Predict step (Proposal distribution).
        dt: Time step
        fx: State transition function f(x, dt, **kwargs) -> x_new
        Q: Optional process noise covariance
        """
        if Q is None: Q = self.Q
        
        # Propagate each particle through the model and add noise
        for i in range(self.N):
            self.particles[i] = fx(self.particles[i], dt, **kwargs)
            noise = np.random.multivariate_normal(np.zeros(self.dim_x), Q)
            self.particles[i] += noise

    def update(self, z, hx, R=None, **kwargs):
        """
        Update step (Weighting and Resampling).
        z: Measurement vector
        hx: Measurement function h(x, **kwargs) -> z_pred
        R: Optional measurement noise covariance
        """
        if R is None: R = self.R
        
        # Update weights based on measurement likelihood
        inv_R = np.linalg.inv(R)
        det_R = np.linalg.det(R)
        norm_factor = 1.0 / np.sqrt((2 * np.pi)**self.dim_z * det_R)
        
        for i in range(self.N):
            zp = hx(self.particles[i], **kwargs)
            diff = z - zp
            # Multivariate Gaussian likelihood
            prob = norm_factor * np.exp(-0.5 * np.dot(diff.T, np.dot(inv_R, diff)))
            self.weights[i] *= prob
            
        # Normalize weights
        self.weights += 1e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)
        
        # Resample if effective number of particles is too low
        if self.neff() < self.N / 2:
            self.resample()

    def resample(self):
        """Resample particles using Systematic Resampling."""
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.  # Ensure last element is 1
        
        # Systematic Resampling
        positions = (np.arange(self.N) + np.random.random()) / self.N
        indexes = np.zeros(self.N, 'i')
        
        i, j = 0, 0
        while i < self.N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                
        self.particles = self.particles[indexes]
        self.weights = np.ones(self.N) / self.N

    def neff(self):
        """Returns the effective number of particles."""
        return 1.0 / np.sum(np.square(self.weights))

    @property
    def x(self):
        """Returns the weighted mean state."""
        return np.average(self.particles, weights=self.weights, axis=0)

    @property
    def P(self):
        """Returns the weighted covariance matrix."""
        x_mean = self.x
        diff = self.particles - x_mean
        return np.dot(self.weights * diff.T, diff)
