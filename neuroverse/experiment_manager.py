from neuroverse.neural_core import FeedForwardNetwork


class ExperimentManager:
    def __init__(self, network=None, memory=None, env=None):
        self.network = network or FeedForwardNetwork([2, 5, 1])
        self.memory = memory
        self.env = env

    def run(self, mode='xor', epochs=3000):
        if mode == 'xor':
            history = self.network.train_xor(epochs=epochs, lr=0.25)
            return {
                'mode': mode,
                'epochs': epochs,
                'loss_history': history,
                'final_loss': history[-1] if history else None,
            }

        if mode == 'env':
            if self.env is None:
                raise ValueError('Environment is required for env mode.')
            self.env.reset()
            done = False
            info = {}
            while not done:
                obs = self.env.observe()
                action = self.env.ACTIONS[0]
                if obs[0] > 0:
                    action = 'right'
                elif obs[0] < 0:
                    action = 'left'
                elif obs[1] > 0:
                    action = 'down'
                elif obs[1] < 0:
                    action = 'up'

                _, reward, done, info = self.env.step(action)
                if self.memory is not None:
                    self.memory.add_experience({
                        'observation': obs,
                        'action': action,
                        'reward': reward,
                        'done': done,
                        'info': info,
                    })

            return {
                'mode': mode,
                'final_state': self.env.state(),
                'status': info.get('status', 'done'),
            }

        raise ValueError(f'Unsupported mode: {mode}')
