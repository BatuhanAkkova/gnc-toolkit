"""
Terrain-Relative Navigation (TRN) feature matching and localization update.
"""

import numpy as np


class FeatureMatchingTRN:
    def __init__(self, map_database):
        """
        Terrain-Relative Navigation (TRN) feature matcher.
        """
        self.map = map_database  # List of (position, descriptor)

    def match_features(self, observed_features):
        """
        Matches observed camera/LIDAR features to the map.

        Returns
        -------
            list of (map_pos, observed_pos)
        """
        matches = []
        for obs_pos in observed_features:
            # Simple nearest neighbor search
            best_match = None
            min_dist = float("inf")
            for map_pos in self.map:
                dist = np.linalg.norm(obs_pos - map_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_match = map_pos
            if min_dist < 10.0:
                matches.append((best_match, obs_pos))
        return matches


def map_relative_localization_update(x, P, matches, R):
    """
    EKF measurement update for TRN.

    Args:
        x (np.ndarray): State estimate [pos, vel].
        P (np.ndarray): Covariance.
        matches (list): (map_pos, observed_pos) pairs.
        R (np.ndarray): Measurement noise covariance.

    Returns
    -------
        tuple: (updated_x, updated_P)
    """
    if not matches:
        return x, P

    # Z = observed_pos, H = identity (mapping state position to observation)
    for map_pos, obs_pos in matches:
        z = obs_pos
        h = map_pos

        H = np.zeros((3, len(x)))
        H[:, :3] = np.eye(3)

        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x = x + K @ y
        P = (np.eye(len(x)) - K @ H) @ P

    return x, P
