from neuroverse.neural_core import FeedForwardNetwork


def test_feedforward_shape():
    net = FeedForwardNetwork([2, 3, 1])
    out = net.forward([0.2, 0.8])
    assert len(out) == 1
    assert 0.0 <= out[0] <= 1.0


def test_xor_training_is_finite():
    net = FeedForwardNetwork([2, 5, 1])
    history = net.train_xor(epochs=150, lr=0.25)
    assert len(history) == 150
    assert all(isinstance(x, float) for x in history)
    assert history[-1] < 1.0
