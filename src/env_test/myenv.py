import os
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import mujoco_py


class ArmReaching(mujoco_env.MujocoEnv):
    def __init__(self):
        self.target_center = np.array([0.3, 0.0, 0.6-0.25])  # m
        self.cnt = 0
        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_7dof_1.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, 1)

    def step(self, action):
        self.do_simulation(action, 1)

        self.cnt += 1
        n = np.floor(self.cnt / 200) % 4

        for i in range(8):
            target_id = self.sim.model.body_name2id("target-{}".format(i+1))
            self.sim.model.body_pos[target_id] = [0, 0, 100]

        for i in range(8):
            r = 0.15
            theta = 2 * np.pi * i / 8

            if n == 0:
                target_id = self.sim.model.body_name2id("target-{}".format(i+1))
                target_position = self.target_center + r * np.array([-np.sin(theta), -np.cos(theta), 0.0])
                self.sim.model.body_pos[target_id] = target_position

                plane_body_id = self.sim.model.body_name2id("target-plane-body")
                plane_geom_id = self.sim.model.geom_name2id("target-plane-geom")
                self.sim.model.geom_size[plane_geom_id] = [0.2, 0.2, 0.001]
                self.sim.model.body_pos[plane_body_id][2] = 0.0

            if n == 1:
                target_id = self.sim.model.body_name2id("target-{}".format(i+1))
                target_position = self.target_center + r * np.array([-np.sin(theta), 0.0, -np.cos(theta)])
                self.sim.model.body_pos[target_id] = target_position

                plane_body_id = self.sim.model.body_name2id("target-plane-body")
                plane_geom_id = self.sim.model.geom_name2id("target-plane-geom")
                self.sim.model.geom_size[plane_geom_id] = [0.2, 0.001, 0.2]
                self.sim.model.body_pos[plane_body_id][2] = 0.0

            if n == 2:
                target_id = self.sim.model.body_name2id("target-{}".format(i+1))
                target_position = self.target_center + r * np.array([0.0, -np.sin(theta), -np.cos(theta)])
                self.sim.model.body_pos[target_id] = target_position

                plane_body_id = self.sim.model.body_name2id("target-plane-body")
                plane_geom_id = self.sim.model.geom_name2id("target-plane-geom")
                self.sim.model.geom_size[plane_geom_id] = [0.001, 0.2, 0.2]
                self.sim.model.body_pos[plane_body_id][2] = 0.0

            if n == 3:
                target_id = self.sim.model.body_name2id("target-{}".format(i+1))
                target_position = self.target_center + r * np.array([-np.sin(theta), -np.cos(theta), 0.0])
                target_position += np.array([0.0, 0.0, 0.05])
                self.sim.model.body_pos[target_id] = target_position

                plane_body_id = self.sim.model.body_name2id("target-plane-body")
                plane_geom_id = self.sim.model.geom_name2id("target-plane-geom")
                self.sim.model.geom_size[plane_geom_id] = [0.2, 0.2, 0.001]
                self.sim.model.body_pos[plane_body_id][2] = 0.05

        return self._get_obs(), 0.0, False, None

    def reset_model(self):
        self.cnt = 0

        qpos = self.init_qpos
        qvel = np.zeros_like(self.init_qvel)
        self.set_state(qpos, qvel)

        target_id = self.sim.model.body_name2id("finger")
        fingertip_position = self.get_body_com("fingers")
        self.sim.model.body_pos[target_id] = fingertip_position
        print(fingertip_position)

        self.target_center = np.array([0.16, 0.0, 0.29])  # m

        target_id = self.sim.model.body_name2id("center")
        self.sim.model.body_pos[target_id] = self.target_center

        for i in range(8):
            target_id = self.sim.model.body_name2id("target-{}".format(i+1))
            self.sim.model.body_pos[target_id] = [0, 0, 0]

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.azimuth = 135
        self.viewer.cam.elevation = -30
        self.viewer.cam.lookat[2] += .2
        self.viewer._hide_overlay = True

    def _get_obs(self):
        observation = np.zeros(3)
        return observation


class ArmReachingControl(mujoco_env.MujocoEnv):
    def __init__(self):
        self.target_center = np.array([0.3, 0.0, 0.6-0.25])  # m
        self.cnt = 0

        self.Kp = 20.0
        self.Kd = 0.3

        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_7dof.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, 1)

    def step(self, action):
        target_id = self.sim.model.body_name2id("finger")
        self.fingertip_position = self.get_body_com("fingers")
        self.sim.model.body_pos[target_id] = self.fingertip_position

        if self.cnt % 1000 == 0:
            i = np.random.randint(8)
            n = np.random.randint(3)
            r = 0.2
            theta = 2 * np.pi * i / 8

            for j in range(8):
                target_id = self.sim.model.body_name2id("target-{}".format(j+1))
                self.sim.model.body_pos[target_id] = [0, 0, 100]

            if n == 0:
                target_id = self.sim.model.body_name2id("target-{}".format(i+1))
                self.target_position = self.target_center + r * np.array([-np.sin(theta), -np.cos(theta), 0.0])

            if n == 1:
                target_id = self.sim.model.body_name2id("target-{}".format(i+1))
                self.target_position = self.target_center + r * np.array([-np.sin(theta), 0.0, -np.cos(theta)])

            if n == 2:
                target_id = self.sim.model.body_name2id("target-{}".format(i+1))
                self.target_position = self.target_center + r * np.array([0.0, -np.sin(theta), -np.cos(theta)])

            self.sim.model.body_pos[target_id] = self.target_position

        if (self.cnt + 500) % 1000 == 0:
            self.target_position = self.target_center

        # Compute control input
        self.jacobian_matrix = self._get_jacobian_matrix()
        x_dev = self.target_position - self.fingertip_position
        v_ref = np.dot(self.jacobian_matrix, np.reshape(self.Kp * x_dev, (-1, 1))).flatten()
        u_ref = v_ref - self.Kd * self.sim.data.qvel
        u = action + u_ref

        self.do_simulation(u, 1)
        self.cnt += 1

        return self._get_obs(), 0.0, False, None

    def reset_model(self):
        self.cnt = 0

        qpos = self.init_qpos
        qvel = np.zeros_like(self.init_qvel)
        self.set_state(qpos, qvel)

        target_id = self.sim.model.body_name2id("finger")
        self.fingertip_position = self.get_body_com("fingers")
        self.sim.model.body_pos[target_id] = self.fingertip_position

        self.target_center = self.fingertip_position.copy()
        self.target_position = self.fingertip_position.copy()

        target_id = self.sim.model.body_name2id("center")
        self.sim.model.body_pos[target_id] = self.target_center

        for i in range(8):
            target_id = self.sim.model.body_name2id("target-{}".format(i+1))
            self.sim.model.body_pos[target_id] = [0, 0, 0]

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2

    def _get_obs(self):
        observation = np.zeros(3)
        return observation

    def _get_jacobian_matrix(self):
        j_pos = np.zeros((3 * 7))
        j_rot = np.zeros((3 * 7))
        mujoco_py.cymj._mj_jacBodyCom(self.model, self.sim.data, j_pos, j_rot, self.model.body_name2id("fingers"))

        jacobian_matrix = j_pos.reshape((3, 7)).T

        return jacobian_matrix


def _example():
    #env = ArmReachingControl()
    env = ArmReaching()
    env.reset()

    while True:
        env.render()
        env.step(np.zeros(env.action_space.shape[0]))


if __name__ == "__main__":
    _example()
