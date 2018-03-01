
from numpy import array, random, dot


class NeuralNetwork:

    def __init__(self):
        random.seed(1)
        self.weights = 2 * random.random((2, 1)) - 1

    def train(self, inputs, outputs, num):
        for iteration in range(num):
            output = self.think(inputs)
            error = outputs - output
            adjustment = 0.01 * dot(inputs.T, error)
            self.weights += adjustment

    def think(self, inputs):
        return dot(inputs, self.weights)


def main():
    network = NeuralNetwork()
    inputs = array([[2, 3], [1, 1], [5, 2], [12, 3]])
    outputs = array([[10, 4, 14, 30]]).T
    network.train(inputs, outputs, 10000)

    print(network.think(array([15, 2])))


if __name__ == '__main__':
    main()
