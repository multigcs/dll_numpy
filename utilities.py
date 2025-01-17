import json
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        self.computation_graph = []
        self.parameters = []

    def add(self, layer):
        self.computation_graph.append(layer)
        self.parameters += layer.getParams()

    def __innitializeNetwork(self):
        for f in self.computation_graph:
            if f.type == "linear":
                weights, bias = f.getParams()
                weights.data = 0.01 * np.random.randn(
                    weights.data.shape[0], weights.data.shape[1]
                )
                bias.data = 0.0

    def fit(self, data, target, batch_size, num_epochs, optimizer, loss_fn):
        loss_history = []
        self.__innitializeNetwork()
        data_gen = DataGenerator(data, target, batch_size)
        itr = 0
        for epoch in range(num_epochs):
            for X, Y in data_gen:
                optimizer.zeroGrad()
                for f in self.computation_graph:
                    X = f.forward(X)
                loss = loss_fn.forward(X, Y)
                grad = loss_fn.backward()
                for f in self.computation_graph[::-1]:
                    grad = f.backward(grad)
                loss_history += [loss]
                print(
                    "Loss at epoch = {} and iteration = {}: {}".format(
                        epoch, itr, loss_history[-1]
                    )
                )
                itr += 1
                optimizer.step()

        return loss_history

    def predict(self, data):
        X = data
        for f in self.computation_graph:
            X = f.forward(X)
        return X

    def save(self, tdata_file):
        tdata = {}
        for pn, param in enumerate(self.parameters):
            # print(pn, param)
            tdata[pn] = {
                "data": param.data.tolist(),
                "grad": param.grad.tolist(),
            }
        open(tdata_file, "w").write(json.dumps(tdata))

    def load(self, tdata_file):
        tdata = json.loads(open(tdata_file, "r").read())
        for pn in tdata:
            self.parameters[int(pn)].data = np.array(tdata[pn]["data"])
            self.parameters[int(pn)].grad = np.array(tdata[pn]["grad"])


def genSpiralData(points_per_class, num_classes):
    data = np.ndarray((points_per_class * num_classes, 2), np.float32)
    target = np.ndarray((points_per_class * num_classes,), np.uint8)
    r = np.linspace(0, 1, points_per_class)
    radians_per_class = 2 * np.pi / num_classes
    for i in range(num_classes):
        t = np.linspace(
            i * radians_per_class, (i + 1.5) * radians_per_class, points_per_class
        ) + 0.1 * np.random.randn(points_per_class)
        data[i * points_per_class : (i + 1) * points_per_class] = np.c_[
            r * np.sin(t), r * np.cos(t)
        ]
        target[i * points_per_class : (i + 1) * points_per_class] = i

    return data, target


def loadData(rdata_file, num_inputs=None, start_inputs=0):
    data = []
    target = []
    if rdata_file.endswith(".json"):
        rdata = json.loads(open(rdata_file, "r").read())
        dcols = len(rdata[0]) - 1
        if num_inputs:
            dcols = num_inputs
        for pn in rdata:
            data.append(pn[start_inputs:dcols])
            target.append(pn[-1])
        data = np.array(data)
    else:
        csv = open(rdata_file, "r").read().split("\n")
        dcols = len(csv[0].split(";")) - 1
        if num_inputs:
            dcols = num_inputs
        for line in csv:
            if line:
                sline = line.split(";")
                data.append(
                    [float(p) for p in sline[start_inputs : dcols + start_inputs]]
                )
                target.append(int(sline[-1]))
        data = np.array(data)
        target = np.array(target)
    return data, target


def saveData(rdata_file, data, target):
    rdata = []
    for n in range(len(data)):
        nc = []
        for c in data[n]:
            nc.append(float(c))
        nc.append(int(target[n]))
        rdata.append(nc)
    if rdata_file.endswith(".json"):
        open(rdata_file, "w").write(json.dumps(rdata))
    else:
        f = open(rdata_file, "w")
        for line in rdata:
            f.write(";".join([str(p) for p in line]))
            f.write("\n")
        f.close()


class DataGenerator:
    def __init__(self, data, target, batch_size, shuffle=True):
        self.shuffle = shuffle
        if shuffle:
            shuffled_indices = np.random.permutation(len(data))
        else:
            shuffled_indices = range(len(data))

        self.data = data[shuffled_indices]
        self.target = target[shuffled_indices]
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(data.shape[0] / batch_size))
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.num_batches:
            batch_data = self.data[
                self.counter * self.batch_size : (self.counter + 1) * self.batch_size
            ]
            batch_target = self.target[
                self.counter * self.batch_size : (self.counter + 1) * self.batch_size
            ]
            self.counter += 1
            return batch_data, batch_target
        else:
            if self.shuffle:
                shuffled_indices = np.random.permutation(len(self.target))
            else:
                shuffled_indices = range(len(self.target))

            self.data = self.data[shuffled_indices]
            self.target = self.target[shuffled_indices]

            self.counter = 0
            raise StopIteration


def plot2DData(data, target):
    plt.scatter(x=data[:, 0], y=data[:, 1], c=target, cmap=plt.cm.rainbow)
    plt.show()


def plot2DDataWithDecisionBoundary(data, target, model):
    x_min, x_max = np.min(data[:, 0]) - 0.5, np.max(data[:, 0]) + 0.5
    y_min, y_max = np.min(data[:, 1]) - 0.5, np.max(data[:, 1]) + 0.5
    X, Y = np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)
    XX, YY = np.meshgrid(X, Y)
    Z = np.argmax(model.predict(np.c_[XX.ravel(), YY.ravel()]), axis=1).reshape(
        XX.shape
    )
    plt.contourf(XX, YY, Z, cmap=plt.cm.seismic)
    plt.scatter(x=data[:, 0], y=data[:, 1], c=target, cmap=plt.cm.seismic)
    plt.show()
