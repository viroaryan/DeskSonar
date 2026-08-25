"""
DeskSonar Kinematic Kalman Filter & Multi-Target Track Manager
Implements optimal state estimation (Range, Radial Velocity) for acoustic targets.
"""
import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


class TargetTrack:
    """
    State estimation for a single acoustic target track using a 2-state Kalman Filter.
    State: [Range (m), Radial Velocity (m/s)]^T
    """

    def __init__(self, track_id: int, init_range: float, init_velocity: float, timestamp: float):
        self.track_id = track_id
        self.created_at = timestamp
        self.last_updated = timestamp
        self.hit_count = 1
        self.miss_count = 0
        self.is_confirmed = False

        # State vector: [range_m, velocity_m_s]
        self.x = np.array([init_range, init_velocity], dtype=np.float64)

        # State covariance matrix P
        self.P = np.array([
            [0.05 ** 2, 0.0],
            [0.0, 0.20 ** 2]
        ], dtype=np.float64)

        # Process noise parameter (acceleration variance in m/s^2)
        self.sigma_a = 0.8

        # Measurement noise covariance R
        self.R = np.array([
            [0.03 ** 2, 0.0],
            [0.0, 0.08 ** 2]
        ], dtype=np.float64)

        self.snr_db = 10.0
        self.magnitude = 1.0

    def predict(self, dt: float) -> None:
        """
        Kalman prediction step: x_k|k-1 = F * x_k-1, P_k|k-1 = F * P * F^T + Q
        """
        if dt <= 0:
            return

        F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ], dtype=np.float64)

        Q = (self.sigma_a ** 2) * np.array([
            [0.25 * (dt ** 4), 0.5 * (dt ** 3)],
            [0.5 * (dt ** 3), dt ** 2]
        ], dtype=np.float64)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.miss_count += 1

    def update(self, meas_range: float, meas_velocity: float, snr_db: float, magnitude: float, timestamp: float) -> None:
        """
        Kalman update step: K = P * H^T * (H * P * H^T + R)^-1, x = x + K * y
        """
        z = np.array([meas_range, meas_velocity], dtype=np.float64)
        H = np.eye(2, dtype=np.float64)

        # Innovation (residual)
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        self.P = (np.eye(2) - (K @ H)) @ self.P

        self.snr_db = snr_db
        self.magnitude = magnitude
        self.last_updated = timestamp
        self.hit_count += 1
        self.miss_count = 0

        if self.hit_count >= 3:
            self.is_confirmed = True

    @property
    def range_m(self) -> float:
        return float(self.x[0])

    @property
    def velocity_m_s(self) -> float:
        return float(self.x[1])


class MultiTargetTracker:
    """
    Multi-target tracking manager with nearest-neighbor data association (GNN)
    and automatic track lifecycle management (Creation, Confirmation, Deletion).
    """

    def __init__(self, gate_distance_m: float = 0.15, max_coasting_misses: int = 5):
        self.gate_dist = gate_distance_m
        self.max_misses = max_coasting_misses
        self.tracks: List[TargetTrack] = []
        self._next_id = 1
        self._last_time: float = time.time()

    def update_tracks(
        self,
        measurements: List[Tuple[float, float, float, float]],  # list of (range_m, velocity_m_s, snr_db, magnitude)
        timestamp: float
    ) -> List[TargetTrack]:
        dt = max(0.001, min(0.1, timestamp - self._last_time))
        self._last_time = timestamp

        # 1. Predict all existing tracks
        for trk in self.tracks:
            trk.predict(dt)

        # 2. Association (Global Nearest Neighbor / Greedy Euclidean Gating)
        unmatched_meas = list(range(len(measurements)))
        matched_tracks = set()

        for t_idx, trk in enumerate(self.tracks):
            best_m_idx = None
            best_dist = float('inf')

            for m_idx in unmatched_meas:
                m_range, m_vel, _, _ = measurements[m_idx]
                # Mahalanobis / Euclidean distance
                dist = abs(trk.range_m - m_range) + 0.3 * abs(trk.velocity_m_s - m_vel)
                if dist < self.gate_dist and dist < best_dist:
                    best_dist = dist
                    best_m_idx = m_idx

            if best_m_idx is not None:
                m_r, m_v, snr, mag = measurements[best_m_idx]
                trk.update(m_r, m_v, snr, mag, timestamp)
                matched_tracks.add(t_idx)
                unmatched_meas.remove(best_m_idx)

        # 3. Create new tentative tracks for unmatched valid measurements
        for m_idx in unmatched_meas:
            m_r, m_v, snr, mag = measurements[m_idx]
            if snr > 4.0:  # Only spawn for decent SNR
                new_track = TargetTrack(self._next_id, m_r, m_v, timestamp)
                new_track.snr_db = snr
                new_track.magnitude = mag
                self.tracks.append(new_track)
                self._next_id += 1

        # 4. Prune dead tracks (miss count exceeded)
        self.tracks = [t for t in self.tracks if t.miss_count <= self.max_misses]

        # Return confirmed tracks sorted by SNR
        confirmed = [t for t in self.tracks if t.is_confirmed]
        confirmed.sort(key=lambda t: t.snr_db, reverse=True)
        return confirmed
