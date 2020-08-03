import numpy as np


class KalmanFilter:
    # Kalman filter for object bounding box

    def __init__(self, initial_state, initial_error, delta_t):
        self.pre_state = initial_state # x_top_left, y_top_left, width, height, v_x, v_y, v_width, v_height
        self.cur_state = np.zeros((8, 1)) # x_top_left, y_top_left, width, height, v_x, v_y, v_width, v_height
        self.delta_t = delta_t
        self.state_transition = np.eye(8)
        rng = np.arange(4)
        self.state_transition[rng, rng + 4] = self.delta_t
        self.state_transition_noise_covariance = np.diag(np.zeros(8)+0.1)
        self.measurement_noise_covariance = np.diag(np.zeros(4)+10)
        self.pre_cov = initial_error
        self.cur_cov = np.zeros((8, 8))
        self.H = np.zeros((4, 8))
        self.H[rng, rng] = 1
        self.gain = np.zeros((4, 4))

    def predict_new_state(self):
        w = np.random.multivariate_normal(mean=np.zeros(8),
                                          cov=self.state_transition_noise_covariance)
        self.cur_state = self.state_transition @ self.pre_state + w[:, np.newaxis]

    def predict_new_cov(self):
        self.cur_cov = self.state_transition @ self.pre_cov @ self.state_transition.T \
                       + self.state_transition_noise_covariance

    def update_gain(self):
        self.gain = self.cur_cov @ self.H.T @ np.linalg.inv(self.H @ self.cur_cov @ self.H.T
                                                            + self.measurement_noise_covariance)

    def update_estimate(self, measurement):
        self.pre_state = self.cur_state + self.gain @ (measurement - self.H @ self.cur_state)

    def update_covariance(self):
        self.pre_cov = (np.eye(8) + self.gain @ self.H) @ self.cur_cov

    def predict(self):
        # Predict next state and process covariance
        self.predict_new_state()
        self.predict_new_cov()

        return self.H @ self.cur_state

    def get_cur_state(self):
        # Return the current state
        return self.H @ self.cur_state

    def update(self, measurement):
        # Update state according to measurement
        self.update_gain()
        self.update_estimate(measurement)
        self.update_covariance()




