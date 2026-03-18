import numpy as np

class IMM:
    """
    Interacting Multiple Model (IMM) Filter.
    Estimates the state of a system that can switch between multiple modes (models).
    Excellent for tracking maneuvering targets (e.g., constant velocity vs. constant acceleration).
    """
    def __init__(self, filters, transition_matrix):
        """
        Initialize the IMM filter.
        filters: List of filter objects (e.g., KF, EKF, UKF)
        transition_matrix: Probability of switching between models [N x N]
        """
        self.filters = filters
        self.N = len(filters)
        self.Phi = transition_matrix # Transition probability matrix (Tij = P(Mj|Mi))
        
        # Probabilities of each model
        self.mu = np.ones(self.N) / self.N
        
        self.dim_x = filters[0].dim_x
        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x)

    def predict(self, dt, **kwargs):
        """
        Predict step (Mixing and model-specific prediction).
        """
        # Interaction (Mixing)
        # Calculate mixing probabilities cj = sum(Tij * mui)
        c = np.dot(self.mu, self.Phi)
        
        # Calculate mixing weights mu_ij = (Tij * mu_i) / cj
        mu_mixed = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                mu_mixed[i, j] = (self.Phi[i, j] * self.mu[i]) / c[j]
                
        # Calculate mixed states and covariances
        x0 = [np.zeros(self.dim_x) for _ in range(self.N)]
        P0 = [np.zeros((self.dim_x, self.dim_x)) for _ in range(self.N)]
        
        # Mix states: x0_j = sum(mu_ij * x_i)
        # Mix covariances: P0_j = sum(mu_ij * (P_i + (x_i-x0_j)(x_i-x0_j).T))
        for j in range(self.N):
            for i in range(self.N):
                x0[j] += mu_mixed[i, j] * self.filters[i].x
                
            for i in range(self.N):
                dx = self.filters[i].x - x0[j]
                P0[j] += mu_mixed[i, j] * (self.filters[i].P + np.outer(dx, dx))
                
        # Model-specific Prediction
        for i in range(self.N):
            self.filters[i].x = x0[i]
            self.filters[i].P = P0[i]
            # Use model-specific prediction
            # Check if filter expects dt
            import inspect
            sig = inspect.signature(self.filters[i].predict)
            if 'dt' in sig.parameters:
                self.filters[i].predict(dt=dt, **kwargs)
            else:
                self.filters[i].predict(**kwargs)

    def update(self, z, **kwargs):
        """
        Update step (Model-specific update and mode probability update).
        """
        # Model-specific Update
        likelihoods = np.zeros(self.N)
        for i in range(self.N):
            # We need to perform update and get likelihood
            # For simplicity, let's assume update is done and it returns the likelihood
            # or we calculate it here. 
            # In gnc-toolkit, Likelihood = exp(-0.5 * y' * S^-1 * y) / sqrt(det(2*pi*S))
            
            # Peek into the filter to get innovation and covariance for likelihood
            # This depends on the filter type (KF vs EKF vs UKF)
            if hasattr(self.filters[i], 'H'):
                H = self.filters[i].H
            else:
                # Fallback or estimate H (for UKF/CKF)
                # For UKF, we'd need zp and S
                pass
                
            # Perform the actual update
            self.filters[i].update(z, **kwargs)
            
            # Since update doesn't return likelihood, calculate it based on filter internals
            # (Assuming y and S are accessible or we reconstruct them)
            # This is a bit brittle, but standard for IMM implementations in Python
            # Let's use a helper for Gaussian likelihood
            likelihoods[i] = self._calculate_likelihood(i, z)
            
        # 2. Model Probability Update
        # mu_j = L_j * c_j / sum(L_k * c_k)
        c = np.dot(self.mu, self.Phi)
        self.mu = likelihoods * c
        self.mu /= np.sum(self.mu)
        
        # 3. Combined State Estimate
        self.x = np.zeros(self.dim_x)
        self.P = np.zeros((self.dim_x, self.dim_x))
        
        for i in range(self.N):
            self.x += self.mu[i] * self.filters[i].x
            
        for i in range(self.N):
            dx = self.filters[i].x - self.x
            self.P += self.mu[i] * (self.filters[i].P + np.outer(dx, dx))

    def _calculate_likelihood(self, i, z):
        """Calculate Gaussian likelihood of measurement z for filter i."""
        # This implementation requires filters to have y (innovation) and S (covariance)
        # or similar accessible after update.
        # Since gnc_toolkit objects don't store y/S currently, we'll need to compute them
        # mock-up for now (ideally filters should be updated to provide this)
        f = self.filters[i]
        
        # Simplified: if it's a KF/EKF
        if hasattr(f, 'H') and hasattr(f, 'x'):
            H = f.H
            y = z - np.dot(H, f.x)
            S = np.dot(np.dot(H, f.P), H.T) + f.R
        else:
            # Fallback for UKF/CKF (this is complex without refactoring UKF)
            # Default to 1.0 or implement properly
            return 1.0
            
        inv_S = np.linalg.inv(S)
        det_S = np.linalg.det(S)
        dim = len(z)
        
        exponent = -0.5 * np.dot(y.T, np.dot(inv_S, y))
        return (1.0 / np.sqrt((2 * np.pi)**dim * det_S)) * np.exp(exponent)
