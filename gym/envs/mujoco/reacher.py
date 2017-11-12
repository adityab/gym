import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py
from mujoco_py import functions

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self,
                'reacher.xml',
                frame_skip=5,
                width=300,
                height=300)
        self.cam_done = False

    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        if len(self.sim.render_contexts) and not self.cam_done:
            cam = self.sim.render_contexts[0].cam
            cam.trackbodyid = 0
            functions.mjv_updateCamera(self.sim.model, self.sim.data, cam, self.sim.render_contexts[0].scn)
            self.cam_done = True

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        lowdim = None
        pixels = None
        
        if self.obs_type in 'lowdim,both':
            theta = self.sim.data.qpos.flat[:2]
            lowdim = np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target")
            ])
            if self.obs_type in 'pixels,both':
                if self.render_ok:
                    pixels = self.sim.render(self.width, self.height)
                else:
                    pixels = np.zeros((self.width, self.height, 3))
        
        if self.obs_type == 'lowdim':
            return lowdim
        elif self.obs_type == 'pixels':
            return pixels
        else:
            return lowdim, pixels
