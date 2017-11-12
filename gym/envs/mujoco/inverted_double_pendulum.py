import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py
from mujoco_py import functions

class InvertedDoublePendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self,
                'inverted_double_pendulum.xml',
                frame_skip=1,
                width=300,
                height=300)
        utils.EzPickle.__init__(self)
        self.cam_done = False

    def _step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = (alive_bonus - dist_penalty - vel_penalty)
        done = bool(y <= 1)
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[:1],  # cart x pos
            np.sin(self.sim.data.qpos[1:]),  # link angles
            np.cos(self.sim.data.qpos[1:]),
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        if len(self.sim.render_contexts) and not self.cam_done:
            cam = self.sim.render_contexts[0].cam
            cam.type = 1 # Tracking
            cam.trackbodyid = 1
            cam.distance = self.model.stat.extent * 0.5
            cam.lookat[2] += 3  # v.model.stat.center[2]
            functions.mjv_updateCamera(self.sim.model, self.sim.data, cam, self.sim.render_contexts[0].scn)
            self.cam_done = True
