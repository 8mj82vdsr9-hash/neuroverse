import math
import random


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)


class Neuron:
    def __init__(self, neuron_id, layer_index, bias=0.0):
        self.id = neuron_id
        self.layer = layer_index
        self.bias = bias
        self.activation = 0.0
        self.z = 0.0
        self.in_connections = []
        self.out_connections = []


class FeedForwardNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = list(layer_sizes)
        self.layers = []
        self.connections = []
        self._build_network()

    def _build_network(self):
        for layer_index, size in enumerate(self.layer_sizes):
            layer = []
            for neuron_index in range(size):
                neuron = Neuron(
                    f'L{layer_index}_N{neuron_index}',
                    layer_index,
                    random.uniform(-1.0, 1.0),
                )
                layer.append(neuron)
            self.layers.append(layer)

        for layer_index in range(len(self.layer_sizes) - 1):
            source_layer = self.layers[layer_index]
            target_layer = self.layers[layer_index + 1]
            for source in source_layer:
                for target in target_layer:
                    weight = random.uniform(-1.0, 1.0)
                    source.out_connections.append({'to': target.id, 'weight': weight})
                    target.in_connections.append({'from': source.id, 'weight': weight})
                    self.connections.append({'from': source.id, 'to': target.id, 'weight': weight})

    def get_neuron(self, neuron_id):
        for layer in self.layers:
            for neuron in layer:
                if neuron.id == neuron_id:
                    return neuron
        return None

    def forward(self, inputs):
        if len(inputs) != len(self.layers[0]):
            raise ValueError(f'Expected {len(self.layers[0])} input values, got {len(inputs)}')

        for idx, value in enumerate(inputs):
            self.layers[0][idx].activation = float(value)
            self.layers[0][idx].z = float(value)

        for layer_index in range(1, len(self.layers)):
            for neuron in self.layers[layer_index]:
                total = neuron.bias
                for conn in neuron.in_connections:
                    source = self.get_neuron(conn['from'])
                    total += source.activation * conn['weight']
                neuron.z = total
                neuron.activation = sigmoid(total)

        return [n.activation for n in self.layers[-1]]

    def state(self):
        neurons = []
        for layer in self.layers:
            for neuron in layer:
                neurons.append({
                    'id': neuron.id,
                    'layer': neuron.layer,
                    'activation': neuron.activation,
                    'bias': neuron.bias,
                    'z': neuron.z,
                    'in_connections': neuron.in_connections,
                    'out_connections': neuron.out_connections,
                })
        return {'layer_sizes': self.layer_sizes, 'neurons': neurons, 'connections': self.connections}

    def _update_connection(self, from_id, to_id, new_weight):
        target = self.get_neuron(to_id)
        for conn in target.in_connections:
            if conn['from'] == from_id:
                conn['weight'] = new_weight

        source = self.get_neuron(from_id)
        for conn in source.out_connections:
            if conn['to'] == to_id:
                conn['weight'] = new_weight

        for conn in self.connections:
            if conn['from'] == from_id and conn['to'] == to_id:
                conn['weight'] = new_weight

    def train_xor(self, epochs=3000, lr=0.25):
        data = [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0]),
        ]
        losses = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for x, y in data:
                self.forward(x)
                output_neuron = self.layers[-1][0]
                error = output_neuron.activation - y[0]
                epoch_loss += error ** 2

                delta_output = error * sigmoid_derivative(output_neuron.z)
                output_neuron.bias -= lr * delta_output

                for conn in output_neuron.in_connections:
                    source = self.get_neuron(conn['from'])
                    grad = delta_output * source.activation
                    new_weight = conn['weight'] - lr * grad
                    self._update_connection(source.id, output_neuron.id, new_weight)

                for hidden_neuron in self.layers[1]:
                    hidden_delta = 0.0
                    for out_conn in hidden_neuron.out_connections:
                        if out_conn['to'] == output_neuron.id:
                            hidden_delta = delta_output * out_conn['weight'] * sigmoid_derivative(hidden_neuron.z)
                            break

                    hidden_neuron.bias -= lr * hidden_delta
                    for conn in hidden_neuron.in_connections:
                        source = self.get_neuron(conn['from'])
                        grad = hidden_delta * source.activation
                        new_weight = conn['weight'] - lr * grad
                        self._update_connection(source.id, hidden_neuron.id, new_weight)

            losses.append(epoch_loss / len(data))

        return losses
