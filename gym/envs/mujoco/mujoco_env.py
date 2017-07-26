import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

try:
    import mujoco_py
    from mujoco_py import load_model_from_path, MjSim
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip, width, height):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.width = width
        self.height = height
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = MjSim(self.model)
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def _reset(self):
        #mjlib.mj_resetData(self.model.ptr, self.data.ptr)
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer.autoscale()
        self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        #self.model._compute_subtree()  # pylint: disable=W0212
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        for _ in range(n_frames):
            self.sim.data.ctrl[:] = ctrl
            self.sim.step()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer().finish()
                self.viewer = None
            return
        img = self.sim.render(self.width, self.height)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    #def _get_viewer(self):
    #    if self.viewer is None:
    #        self.viewer = mujoco_py.MjViewer()
    #        self.viewer.start()
    #        self.viewer.set_model(self.model)
    #        self.viewer_setup()
    #    return self.viewer

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.sim.data.subtree_com[idx]

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(body_name)
        # FIXME: Not entirely sure, I have no idea what I'm doing,
        # this doesn't seem documented anywhere for mujoco 1.50
        #return self.model.body_comvels[idx]
        return np.concatenate((
            self.data.body_xvelp[idx],
            self.data.body_xvelr[idx]
            ))

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.data.body_xmat[idx].reshape((3, 3))

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
