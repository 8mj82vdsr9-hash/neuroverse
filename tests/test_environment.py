from neuroverse.environment import SimpleEnvironment


def test_environment_step_and_reward():
    env = SimpleEnvironment(width=5, height=5)
    obs, reward, done, info = env.step('right')
    assert isinstance(obs, list)
    assert len(obs) == 3
    assert isinstance(reward, float)
    assert info['status'] in {'running', 'success', 'timeout'}
