from typing import Optional

import numpy as np

from mod_stable_baselines3.stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper


class VecExtractDictObs(VecEnvWrapper):
    """
    A vectorized wrapper for extracting dictionary observations.

    :param venv: The vectorized environment
    :param key: The key of the dictionary observation
    """

    def __init__(self, venv: VecEnv, key: str):
        self.key = key
        super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

    def reset(self, state: Optional[list] = None, rand_state: Optional[list] = None) -> np.ndarray:
        obs = self.venv.reset(state, rand_state)
        return obs[self.key]

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs[self.key], reward, done, info
