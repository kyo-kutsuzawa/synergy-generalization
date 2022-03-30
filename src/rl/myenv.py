import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.myenv import *
del sys.path[-1]


class ArmReachingFixedPoints(ArmReaching):
    """Reaching task to eight target points on a plane.

    The arm should reach to a given target at a certain time and finally return to the center position.
    The initial condition is fixed.
    """
    def reset_model(self):
        # Reset the initial states
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        # Reset environmental states
        self.t = 0.0  # sec
        self.cnt = 0

        # Set the center position to the initial finger-tip position
        fingertip_position = self.get_body_com("fingers")
        self.target_center = fingertip_position.copy()

        # Reset the target position
        self.target_position = self._get_new_target_position()
        target_id = self.sim.model.body_name2id("target")
        self.sim.model.body_pos[target_id] = self.target_position

        # Get the first observation
        observation = self._get_obs()

        return observation

    def _get_new_target_position(self):
        """Get a new target position.

        This method is assumed to be called at the beginning of the task.
        """
        n_targets = 8

        # Horizontal plane
        if self.target_mode == 0:
            r = 0.15  # m
            theta = 2*np.pi * np.random.randint(n_targets) / n_targets
            target_position = self.target_center + r * np.array([-np.sin(theta), -np.cos(theta), 0.0])

        # Vertical plane
        if self.target_mode == 1:
            r = 0.15  # m
            theta = 2*np.pi * np.random.randint(n_targets) / n_targets
            target_position = self.target_center + r * np.array([-np.sin(theta), 0.0, -np.cos(theta)])

        return target_position
