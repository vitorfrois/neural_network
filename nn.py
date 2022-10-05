import numpy as np
import sympy as sp
import pandas as pd
import gradio 

data = pd.read_csv('train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
data_test = data[0:1000].T
X_test = data_test[1:n].T
Y_test = data_test[0].T

data_train = data[1000:m].T 
Y_train = data_train[0].T
X_train = data_train[1:n].T

def one_hot(i, size):
    arr = np.zeros(size)
    arr.fill(-1)
    arr[i] = 1
    return arr

def wrap_image(m):
    return m/255

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def mnistize(img):
    img = img.reshape(1, 28, 28, 1)
    img = np.absolute(img-255)
    img = img.flatten()
    return img


class Layer:
    def __init__(self, input_size, output_size, learning_rate):
        self.learning_rate = learning_rate
        self.w = np.random.rand(output_size, input_size) - 0.5
        self.b = np.random.rand(output_size, 1) - 0.5

    def tanh(self,m):
        return np.tanh(m)

    def tanh_diff(self,m):
        return 1 - (np.tanh(m)**2)

    def forward(self, a1):
        self.a1 = a1
        self.Zl = np.matmul(self.w, self.a1) + self.b
        print("zl shape = ", self.Zl.shape)
        self.a2 = self.tanh(self.Zl)
        return self.a2

    def computate_weights(self, grad):
        print("grad shape ", grad.shape)
        dZ = np.multiply(grad, self.tanh_diff(self.Zl))
        print("dZ shape ", dZ.shape)
        dW = np.matmul(dZ, self.a1.T)
        dB = dZ
        return dW, dB

    def backward(self, grad):
        dW, dB = self.computate_weights(grad)
        self.w -= dW*self.learning_rate
        self.b -= dB*self.learning_rate
        print(grad.shape)
        print(self.w.shape)
        return np.matmul(self.w, grad)

class Network:
    def __init__(
            self, 
            arch, 
            learning_rate): 
        n_layers = len(arch)
        self.layers = []
        for info in arch:
            print(info)
            self.layers.append(Layer(info["input"], info["output"], learning_rate))
        self.loss = 0

    def forward_pass(self, a1):
        for layer in self.layers:
            a1 = layer.forward(a1)
        return a1

    def mse(self, activation, validation): 
        return np.sum((activation-validation)**2)

    def mse_grad(self, activation, validation):
        return 2 * (activation - validation) / np.size(validation)

    def backward_pass(self, grad):
        for layer in reversed(self.layers):
            print("here")
            grad = layer.backward(grad)
            print(grad.shape)

    def train(self, X_train, Y_train):
        for j in range(10):
            loss = 0
            for i in range(X_train.shape[0]):
                y = one_hot(Y_train[i], 10).reshape(10, 1)
                self.output = self.forward_pass(X_train[i])
                loss += self.mse(self.output, y)
                grad = self.mse_grad(self.output, y)
                print(grad.shape)
                self.backward_pass(grad)
            print(f"{j} epoch loss = {loss/X_train.shape[0]}")

    def testNN(self, X_test, Y_test):
        hit = 0
        for i in range(1000):
            self.output = X_test[i]
            for layer in self.layers:
                self.output = layer.forward(self.output)
            sm = softmax(self.output)
            index = np.argmax(sm)
            if(index == Y_test[i]):
                hit += 1
        return (hit * 100)/1000

    def predict(self, img):
        img = mnistize(img)
        a2 = self.layer.forward(img)
        sm = softmax(a2)
        LABELS = [0,1,2,3,4,5,6,7,8,9]
        confidences = {LABELS[i]: v.item() for i, v in zip(np.arange(10), sm)}
        return confidences


def main():
    first_layer = {
        "input": 784,
        "output": 10
    }
    hidden_layer = {
        "input": 16,
        "output": 16
    }
    output_layer = {
        "input": 10,
        "output": 10
    }
    arch = []
    arch.append(first_layer)
    # arch.append(output_layer)
    print(arch)
    n = Network(arch, 0.01)
    n.train(wrap_image(X_train), Y_train)
    print("Acurracy = ", n.testNN(X_test, Y_test))
    # image = gradio.components.Image(shape=(28,28),source="upload", invert_colors=True, image_mode="L")
    # gradio.Interface(fn=n.predict,
    #     inputs=image,
    #     outputs="label",
    #     live=True).launch(share=True)



main()



    
    