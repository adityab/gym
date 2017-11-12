import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py
from mujoco_py import functions

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self,
                'half_cheetah.xml',
                frame_skip=5,
                width=300,
                height=300)
        utils.EzPickle.__init__(self)
        self.cam_done = False

    def _step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        if len(self.sim.render_contexts) and not self.cam_done:
            cam = self.sim.render_contexts[0].cam
            cam.type = 1 # Tracking
            cam.distance = self.model.stat.extent * 0.5
            cam.trackbodyid = 1
            functions.mjv_updateCamera(self.sim.model, self.sim.data, cam, self.sim.render_contexts[0].scn)
            self.cam_done = True
