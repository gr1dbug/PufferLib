"""A minimal template for your own envs."""

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.gbgridenv import binding

OBS_WINDOW = 20
OBS_CHANNELS = 6


class GbGridEnv(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        render_mode=None,
        log_interval=128,
        size=100,
        agent_energy=50,
        buf=None,
        seed=0,
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_WINDOW, OBS_WINDOW, OBS_CHANNELS), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            size=size,
            agent_energy=agent_energy,
        )
        self.size = size

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))
        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

    def get_grid(self, env_id=0):
        return binding.vec_get_grid(self.c_envs, env_id)

    def put_grid(self, env_id=0, **kwargs):
        return binding.vec_put_grid(self.c_envs, env_id, **kwargs)


if __name__ == "__main__":
    N = 4096
    env = GbGridEnv(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 5, (CACHE, N))

    import time

    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % CACHE])
        steps += 1

    print("Squared SPS:", int(env.num_agents * steps / (time.time() - start)))
