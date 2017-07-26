import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

import mujoco_py
from mujoco_py import functions

def mass_center(sim):
    mass = sim.model.body_mass
    rmass = mass.repeat(3).reshape(mass.shape[0], 3)
    xpos = sim.data.xipos
    return (np.sum(rmass * xpos, 0) / np.sum(mass))[0]

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self,
                'humanoid.xml',
                frame_skip=5,
                width=300,
                height=300)
        utils.EzPickle.__init__(self)
        self.cam_done = False

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def _step(self, a):
        pos_before = mass_center(self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        if len(self.sim.render_contexts) and not self.cam_done:
            cam = self.sim.render_contexts[0].cam
            cam.type = 1 # Tracking
            cam.trackbodyid = 1
            cam.distance = self.model.stat.extent * 1.0
            cam.lookat[2] += .8
            cam.elevation = -20
            functions.mjv_updateCamera(self.sim.model, self.sim.data, cam, self.sim.render_contexts[0].scn)
            self.cam_done = True
