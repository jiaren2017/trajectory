import sys
import numpy as np
from matplotlib import pyplot as plt

class SimulatedWalker(object):
    r""" Object for dynamic Walk-Simulation"""
    def __init__(self, trace_x, trace_y, config):
        r""" Initialization

        TODO: Config-Parameter erklaeren. Erweitern auf v_history!

        Parameters
        ----------
        trace_x, trace_y : *array_like*
            Lists containing the x- and y-coordinates of the original (noisy) Trace.
        config : *dict*
            Dictionary with all necessary Informations for the Simulation-run, namely:
            /mass (float), dt (float), vmax (float) and rgoal (float)/

        Returns
        -------
        x_history, y_history:
            Lists containing all Positions during the Simulation-Runs

        """

        self.trace_x = np.array(trace_x, dtype=float)
        self.trace_y = np.array(trace_y, dtype=float)
        self.idx = 0
        self.pos = np.array([self.trace_x[0], self.trace_y[0]])
        self.current_target = np.array([self.trace_x[1], self.trace_y[1]])

        self.heading = self.normalize(self.current_target - self.pos)
        self.vel = np.array([0,0], dtype=float)
        self.mass = config["mass"]
        self.dt = config["dt"]
        self.max_speed = config["vmax"]
        self.arrive_radius2 = config["rgoal"] ** 2
        self.x_history = []
        self.y_history = []


    def update(self):
        r"""Updates the position by calculating the steering-force, then
        the acceleration, and finaly the velocity of the simulated object.
        """
        steering_force = self.calc_steering()
        acc = steering_force / self.mass
        self.vel += (acc * self.dt)
        self.vel = np.clip(self.vel, -1*self.max_speed, self.max_speed)
        self.pos += (self.vel * self.dt)
        self.heading = self.normalize(self.vel)
        self.x_history.append(self.pos[0])
        self.y_history.append(self.pos[1])

    def normalize(self, vector):
        r"""Normalizes a vector and throws an exception for norms close to zero.
        """
        norm = np.linalg.norm(vector)
        if not np.allclose(0, norm):
            return vector / norm
        else:
            msg = "Warning: Normalizing Heading Failed"
            raise ValueError(msg)

    def calc_steering(self):
        r"""Erweitern auf mehr Modi...
        """
        return self.seek()

    def run_simulation(self):
        r"""Main-Simulationloop."""
        count = 0
        while count < 1e6:
            self.update()
            if self.arrived():
                self.idx += 1
                if self.idx == len(self.trace_x):
                    print("Finished in %i steps..."%count)
                    return self.x_history, self.y_history
                else:
                    self.current_target = np.array([self.trace_x[self.idx],\
                                                self.trace_y[self.idx]])
            count += 1
        print("Could not Finish Simulation in less than 1e6 steps...")
        return None, None

    def arrived(self):
        r"""Check if the object is close to the next waypoint."""
        dr = self.current_target - self.pos
        if np.sum(np.power(dr, 2)) <= self.arrive_radius2:
            return True
        else:
            return False

    def seek(self):
        r"""(Very) Simple calculation of the Steering Force."""
        desired_vel = self.normalize(self.current_target-self.pos)\
                      *self.max_speed
        return desired_vel - self.vel
