import os
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np


class ArmReaching(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, arm_type=0):
        utils.EzPickle.__init__(self)

        # Setup time-related constants
        dt = 0.01  # Hard-coded simulation interval [s]. This value is defined in arm_7dof.xml
        self.T = 1.0
        self.cnt_target = int(self.T / (dt) * 0.5)
        self.cnt_finish = int(self.T / (dt))

        # Initialize time variables
        self.t = 0.0  # sec
        self.cnt = 0

        if arm_type == 0:
            filename = "arm_7dof.xml"
            self.target_center = np.array([0.21, 0.0, 0.24])  # m
        elif arm_type == 1:
            filename = "arm_7dof_1.xml"
            self.target_center = np.array([0.16, 0.0, 0.29])  # m
        elif arm_type == 2:
            filename = "arm_7dof_2.xml"
            self.target_center = np.array([0.26, 0.0, 0.19])  # m

        # Setup target position parameters
        self.target_mode = 0
        self.target_position = self._get_new_target_position()

        # Setup reward coefficients
        self.alpha1 = 1.0
        self.alpha2 = 1.0

        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        mujoco_env.MujocoEnv.__init__(self, xml_path, 1)

    def step(self, action):
        # Step the simulation
        self.do_simulation(action, 1)
        self.cnt += 1
        self.t = self.cnt * self.dt

        # Get an observation
        observation = self._get_obs()

        # Get the finger tip position
        self.fingertip_position = self.get_body_com("fingers")
        self.fingertip_velocity = self.data.get_body_xvelp("fingers")

        # Compute a reward
        reward_tracking = 0.0
        reward_vel = 0.0
        reward_energy = -np.square(action).sum() * self.dt
        if self.cnt == self.cnt_target:
            reward_tracking = -np.square(self.target_position - self.fingertip_position).sum()
            reward_vel = -np.square(self.sim.data.qvel).sum()
        if self.cnt == self.cnt_finish:
            reward_tracking = -np.square(self.target_center - self.fingertip_position).sum()
            reward_vel = -np.square(self.sim.data.qvel).sum()
        reward = reward_tracking + self.alpha1 * reward_vel + self.alpha2 * reward_energy

        # Judge the task finish
        if self.cnt == self.cnt_finish:
            done = True
        else:
            done = False

        # Make information
        info = {
            "tracking reward": reward_tracking,
            "velocity reward": reward_vel,
            "energy reward": reward_energy
        }

        return observation, reward, done, info

    def reset_model(self):
        # Reset the initial states
        qpos = self.init_qpos + np.random.uniform(-0.1,   0.1,   size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(-0.005, 0.005, size=self.model.nv)
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

    def viewer_setup(self):
        """Setup the camera position.
        """
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.azimuth = 135
        self.viewer.cam.elevation = -30
        self.viewer.cam.lookat[0] = 0.22
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.26
        self.viewer._hide_overlay = True
        self.frames = []

    def record_setup(self, width, height):
        self.render(mode="rgb_array", width=width, height=height)
        self.video_width  = width
        self.video_height = height

    def record(self):
        if hasattr(self, "frames"):
            frame = self.render(mode="rgb_array", width=self.video_width, height=self.video_height)
            frame = frame[:, :, [2, 1, 0]]
            self.frames.append(frame)

    def save_video(self, filename):
        if hasattr(self, "frames"):
            import cv2
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            size = self.frames[0].shape[0:2][::-1]

            video = cv2.VideoWriter(filename, codec, 100, size)
            for frame in self.frames:
                video.write(frame)
            video.release()

    def _get_obs(self):
        """Get an observation.
        
        An observation consists of
        1) seven-dimensional joint angle,
        2) seven-dimensional joint angular velocity,
        3) three-dimensional target position, and
        4) one-dimensional time-encoding value.

        Observations include time encoding for feed-forward neural network policies to provide time information.
        It enables us to employ feed-forward neural network policies, although this task requires temporal information.
        """
        joint_angle = self.sim.data.qpos
        joint_vel   = self.sim.data.qvel
        t_enc = [self.t / self.T]
        observation = np.concatenate([joint_angle, joint_vel, self.target_position, t_enc])

        return observation

    def _get_new_target_position(self):
        """Get a new target position.

        This method is assumed to be called at the beginning of the task.
        """
        if self.target_mode == 0:
            r = np.random.uniform(0.0, 0.2)  # m
            theta = np.random.uniform(-np.pi, np.pi)
            target_position = self.target_center + r * np.array([-np.sin(theta), -np.cos(theta), 0.0])

        if self.target_mode == 1:
            r = np.random.uniform(0.0, 0.2)  # m
            theta = np.random.uniform(-np.pi, np.pi)
            target_position = self.target_center + r * np.array([-np.sin(theta), 0.0, -np.cos(theta)])

        if self.target_mode == 2:
            self.target_center = np.array([0.3, 0.0, 0.6-0.25]) + np.array([0.0, 0.0, -0.2])
            r = np.random.uniform(0.0, 0.2)  # m
            theta = np.random.uniform(-np.pi, np.pi)
            target_position = self.target_center + r * np.array([-np.sin(theta), -np.cos(theta), 0.0]) + np.array([0.0, 0.0, -0.1])

        if self.target_mode == 3:
            r = np.random.uniform(0.0, 0.2)  # m
            theta = np.random.uniform(-np.pi, np.pi)
            target_position = self.target_center + r * np.array([-np.sin(theta), -np.cos(theta)*np.cos(np.pi/4), -np.cos(theta)*np.sin(np.pi/4)])

        return target_position


class ArmReachingDeterministic(ArmReaching):
    """This class is a deterministic version of ArmReaching; initial states and target positions are always the same when resetting.

    The target position can be set by the set_target_position method.
    """
    def __init__(self, target_position=None, qpos=None, qvel=None, arm_type=0):
        ArmReaching.__init__(self, arm_type)
        utils.EzPickle.__init__(**locals())

        self.target_position = target_position

        if qpos is None:
            self.qpos0 = self.init_qpos
        else:
            self.qpos0 = qpos

        if qvel is None:
            self.qvel0 = self.init_qvel
        else:
            self.qvel0 = qvel

    def reset_model(self):
        # Reset the initial states
        qpos = self.qpos0
        qvel = self.qvel0
        self.set_state(qpos, qvel)

        # Reset environmental states
        self.t = 0.0  # sec
        self.cnt = 0

        # Set the center position to the initial finger-tip position
        fingertip_position = self.get_body_com("fingers")
        self.target_center = fingertip_position.copy()

        # Reset the target position
        target_id = self.sim.model.body_name2id("target")
        self.sim.model.body_pos[target_id] = self.target_position

        plane_body_id = self.sim.model.body_name2id("target-plane-body")
        plane_geom_id = self.sim.model.geom_name2id("target-plane-geom")
        self.sim.model.geom_size[plane_geom_id] = [0.2, 0.001, 0.2]
        self.sim.model.body_pos[plane_body_id][2] = 0.0

        # Get the first observation
        observation = self._get_obs()

        return observation

    def set_target_position(self, pos):
        self.target_position = pos
        self._ezpickle_kwargs["target_position"] = self.target_position

    def set_initial_states(self, qpos, qvel):
        self.qpos0 = qpos
        self.qvel0 = qvel
        self._ezpickle_kwargs["qpos"] = self.qpos0
        self._ezpickle_kwargs["qvel"] = self.qvel0

    def _get_new_target_position(self):
        target_position = np.zeros(3)
        return target_position


def _example():
    env = ArmReaching(2)
    env.reset()

    while True:
        env.render()

        #action = env.action_space.sample()
        action = [0.0]*env.action_space.shape[0]
        obs, _, _, _ = env.step(action)

        for o in obs:
            print("{:8.3f}".format(o), end="  ")
        print("", end="\r")


if __name__ == "__main__":
    _example()
