import numpy as np

from pufferlib.ocean.gbgridenv.gbgridenv import GbGridEnv


OBS_WINDOW = 20
OBS_CHANNELS = 6


def _energy_from_obs(obs):
    # Scalar channels are most reliable at the center pixel (non-padding).
    return float(obs[OBS_WINDOW // 2, OBS_WINDOW // 2, 1])


def test_gbgridenv_observation_shape_range_and_channels():
    env = GbGridEnv(num_envs=8, size=30, agent_energy=25)
    obs, _ = env.reset(seed=7)

    assert obs.shape == (8, OBS_WINDOW, OBS_WINDOW, OBS_CHANNELS)
    assert obs.dtype == np.float32

    # Channel 0: normalized cell type in [0, 1]
    assert np.min(obs[..., 0]) >= 0.0
    assert np.max(obs[..., 0]) <= 1.0

    # Channel 1: normalized energy in [0, 1]
    assert np.min(obs[..., 1]) >= 0.0
    assert np.max(obs[..., 1]) <= 1.0

    # Channels 2,3: normalized goal direction in [-1, 1]
    assert np.min(obs[..., 2]) >= -1.0
    assert np.max(obs[..., 2]) <= 1.0
    assert np.min(obs[..., 3]) >= -1.0
    assert np.max(obs[..., 3]) <= 1.0

    # Channels 4,5: normalized distance and carried load in [0, 1]
    assert np.min(obs[..., 4]) >= 0.0
    assert np.max(obs[..., 4]) <= 1.0
    assert np.min(obs[..., 5]) >= 0.0
    assert np.max(obs[..., 5]) <= 1.0

    # Center pixel should always carry valid scalar features.
    center = obs[:, OBS_WINDOW // 2, OBS_WINDOW // 2, :]
    assert np.min(center[:, 1]) >= 0.0 and np.max(center[:, 1]) <= 1.0
    assert np.min(center[:, 2]) >= -1.0 and np.max(center[:, 2]) <= 1.0
    assert np.min(center[:, 3]) >= -1.0 and np.max(center[:, 3]) <= 1.0
    assert np.min(center[:, 4]) >= 0.0 and np.max(center[:, 4]) <= 1.0
    assert np.min(center[:, 5]) >= 0.0 and np.max(center[:, 5]) <= 1.0

    env.close()


def test_gbgridenv_exactly_one_depot_after_reset():
    env = GbGridEnv(num_envs=4, size=25, agent_energy=20)
    env.reset(seed=11)

    for env_id in range(4):
        grid = env.get_grid(env_id)
        cell_types = grid["cell_types"]
        depot_count = int(np.sum(cell_types == 2))
        assert depot_count == 1

    env.close()


def test_gbgridenv_terminal_is_observable():
    # Small energy makes this test short and deterministic.
    env = GbGridEnv(num_envs=1, size=12, agent_energy=5)
    env.reset(seed=13)

    saw_terminal = False
    for _ in range(16):
        # Deterministic action stream; exact action doesn't matter.
        obs, rewards, terminals, truncations, info = env.step(np.array([0], dtype=np.int32))
        if terminals[0]:
            saw_terminal = True
            break

    assert saw_terminal, "Expected at least one visible terminal before deferred reset"
    env.close()


def test_gbgridenv_energy_monotonic_until_terminal():
    env = GbGridEnv(num_envs=1, size=20, agent_energy=8)
    obs, _ = env.reset(seed=17)

    energies = [_energy_from_obs(obs[0])]
    for _ in range(20):
        obs, rewards, terminals, truncations, info = env.step(np.array([1], dtype=np.int32))
        energies.append(_energy_from_obs(obs[0]))
        if terminals[0]:
            break

    diffs = np.diff(np.array(energies, dtype=np.float32))
    assert np.all(diffs <= 1e-6), f"Energy increased unexpectedly: {energies}"
    assert any(e <= 0.0 for e in energies), "Energy should eventually reach zero before terminal"
    env.close()


def test_gbgridenv_invalid_move_has_no_movement_reward():
    # In a 3x3 map, the only interior cell becomes depot; all moves from center are invalid.
    env = GbGridEnv(num_envs=1, size=3, agent_energy=10)
    env.reset(seed=23)

    # Action 0 = left. From center this should be invalid (border is solid).
    obs, rewards, terminals, truncations, info = env.step(np.array([0], dtype=np.int32))
    reward = float(rewards[0])

    # Base step penalty is -0.01 and invalid move penalty is -0.02, so reward must be <= -0.02.
    # This guards against accidentally granting positive movement reward on invalid moves.
    assert reward <= -0.02, f"Invalid move appears rewarded: got reward {reward}"
    env.close()
