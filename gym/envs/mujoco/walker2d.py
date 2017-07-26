import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py
from mujoco_py import functions

class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self,
                "walker2d.xml",
                frame_skip=4,
                width=300,
                height=300)
        utils.EzPickle.__init__(self)
        self.cam_done = False

    def _step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        if len(self.sim.render_contexts) and not self.cam_done:
            cam = self.sim.render_contexts[0].cam
            cam.type = 1 # Tracking
            cam.trackbodyid = 1
            cam.distance = self.model.stat.extent * 0.5
            cam.lookat[2] += .8
            cam.elevation = -20
            functions.mjv_updateCamera(self.sim.model, self.sim.data, cam, self.sim.render_contexts[0].scn)
            self.cam_done = True
