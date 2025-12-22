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

            # Step 1: update the pose estimate using the kinematic model
            # TODO: Update these equations
            self.q[0] = self.q[0] + dX*np.cos(self.q[2])
            self.q[1] = self.q[1] + dX*np.sin(self.q[2])
            self.q[2] = self.q[2] + dT

            self.q[2] = wrap_angle(self.q[2])

            # Step 2: Calculate the process model Jacobians
            # TODO: Define F and W
            F = np.array([[1, 0, -dX*np.sin(self.q[2])], 
                        [0, 1, dX*np.cos(self.q[2])], 
                        [0, 0, 1]])

            W = np.array([[np.cos(self.q[2]), 0], [np.sin(self.q[2]), 0], [0, 1]])

            # Step 3: update the covariance estimate
            # TODO: Update this equation
            self.P = np.dot(np.dot(F, self.P), np.transpose(F)) + np.dot(np.dot(W, self.Q), np.transpose(W))

    def update(self, z: np.ndarray, tag_xy: np.ndarray):
        # z is the measurement in the form [range, bearing]
        # tag_xy is the tag location of the tag in world coordinates [tag_x, tag_y]

        with self.q_mutex:

            # Step 1: calculate the predicted range and bearing measurements
            # TODO: update the equations below
            dx = tag_xy[0] - self.q[1]
            dy = tag_xy[1] - self.q[1]
            r2 = dx**2 + dy**2
            rng_pred = np.sqrt(r2)
            bearing_pred = wrap_angle(np.arctan2(dy, dx) - self.q[2])
            z_pred = np.array([rng_pred, bearing_pred])

            # Step 2: Calculate the innovation
            # TODO: Define y
            y = z - z_pred
            y[1] = wrap_angle(y[1])

            # Step 3: Calculate the measurement Jacobian
            # TODO: Define H
            H = np.array([[-dx/r, -dy/r, 0],
                        [dy/r2, -dx/r2, -1]])

            # Step 4: Calculate the Kalman gain
            # TODO: Define K
            S = H @ self.P @ np.transpose(H) + self.R
            K = self.P @ np.transpose(H) @ np.linalg.inv(S)

            # Step 5: Update the state and covariance estimates
            # TODO: Update these equations
            self.q = self.q + K @ y
            self.q[2] = wrap_angle(self.q[2])
            I = np.eye(3)
            self.P =(I - K @ H) @ self.P






