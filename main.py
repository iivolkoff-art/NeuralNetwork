import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

inputs = np.array([[0,0,1],
                  [1,1,1],
                  [1,0,1],
                  [0,1,1]])
training_outputs = np.array([[0],
                             [1],
                             [1],
                             [0]])
np.random.seed(1)
weight = 2 * np.random.random((3,1)) - 1

print("Start Weight: ")
print(weight)

for i in range(200000):
    outputs = sigmoid(np.dot(inputs, weight))

    err = training_outputs - outputs
    adjustments = np.dot(inputs.T, err * (outputs * (1 - outputs)))

    weight += adjustments

print("End Weight: ")
print(weight)

print("Result: ")
print(outputs)

#test1
test_inputs_1 = np.array([0,1,1])
outputs = sigmoid(np.dot(test_inputs_1, weight))
print("Test1: ")
print(outputs)

#test2
test_inputs_2 = np.array([1,0,1])
outputs = sigmoid(np.dot(test_inputs_2, weight))
print("Test2: ")
print(outputs)
