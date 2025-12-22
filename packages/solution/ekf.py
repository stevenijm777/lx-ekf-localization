#!/usr/bin/env python3

import numpy as np
from multiprocessing import Lock


def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi

class EKF:
    def __init__(self, q_0: np.ndarray, P_0: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.q = q_0
        self.P = P_0
        self.Q = Q
        self.R = R
        self.q_mutex = Lock()
    def predict(self, dX, dT):
        with self.q_mutex:
            # Guardamos el angulo actual para los Jacobianos
            theta = self.q[2]

            # Step 1: update the pose estimate using the kinematic model
            self.q[0] = self.q[0] + dX * np.cos(theta)
            self.q[1] = self.q[1] + dX * np.sin(theta)
            self.q[2] = wrap_angle(self.q[2] + dT)

            # Step 2: Calculate the process model Jacobians (usando theta anterior)
            F = np.array([[1, 0, -dX * np.sin(theta)], 
                          [0, 1,  dX * np.cos(theta)], 
                          [0, 0,  1]])

            W = np.array([[np.cos(theta), 0], 
                          [np.sin(theta), 0], 
                          [0, 1]])

            # Step 3: update the covariance estimate
            self.P = F @ self.P @ F.T + W @ self.Q @ W.T

    def update(self, z: np.ndarray, tag_xy: np.ndarray):
        with self.q_mutex:
            # Step 1: calculate the predicted range and bearing measurements
            dx = tag_xy[0] - self.q[0] 
            dy = tag_xy[1] - self.q[1]
            r2 = dx**2 + dy**2
            r = np.sqrt(r2)
            if r < 0.01: return
            
            bearing_pred = wrap_angle(np.arctan2(dy, dx) - self.q[2])
            z_pred = np.array([r, bearing_pred])

            # Step 2: Calculate the innovation
            y = z - z_pred
            y[1] = wrap_angle(y[1])

            # Step 3: Calculate the measurement Jacobian
            H = np.array([[-dx/r, -dy/r,  0],
                          [dy/r2, -dx/r2, -1]])

            # Step 4: Calculate the Kalman gain
            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T @ np.linalg.inv(S)

            # Step 5: Update the state and covariance estimates
            self.q = self.q + K @ y
            self.q[2] = wrap_angle(self.q[2])
            
            I = np.eye(3)
            self.P = (I - K @ H) @ self.P




