import os
import dl_numpy as DL
import utilities
import numpy as np
import json


"""
csv = open("rdata.csv", "r").read().split("\n")
for ln, line in enumerate(csv, 1):
    if line:
        print(f"{255 / ln};{line}")

exit(0)
"""


if __name__ == "__main__":
    batch_size = 20
    num_epochs = 200
    samples_per_class = 100
    num_classes = 3
    start_inputs = 0
    num_inputs = 2
    hidden_units = 100
    rdata_file = "rdata.csv"
    tdata_file = "tdata.json"

    # build
    model = utilities.Model()
    model.add(DL.Linear(num_inputs, hidden_units))
    model.add(DL.ReLU())
    model.add(DL.Linear(hidden_units, num_classes))
    optim = DL.SGD(model.parameters, lr=1.0, weight_decay=0.001, momentum=0.9)
    loss_fn = DL.SoftmaxWithLoss()

    # load/gen data
    if not os.path.isfile(rdata_file):
        data, target = utilities.genSpiralData(samples_per_class, num_classes)
        utilities.saveData(rdata_file, data, target)
    else:
        data, target = utilities.loadData(rdata_file, num_inputs, start_inputs)

    # learn
    if not os.path.isfile(tdata_file):
        model.fit(data, target, batch_size, num_epochs, optim, loss_fn)
        model.save(tdata_file)
    else:
        model.load(tdata_file)

    # test
    predicted = model.predict(data)
    predicted_labels = np.argmax(predicted, axis=1)
    #for row in predicted:
    #    print(row.tolist()[0] > 0, row.tolist()[1] > 0, row.tolist()[2] > 0)
    #exit(0)

    accuracy = np.sum(predicted_labels == target) / len(target)
    print("Model Accuracy = {}".format(accuracy))
    if num_inputs == 2:
        utilities.plot2DDataWithDecisionBoundary(data, target, model)
    else:
        utilities.plot2DData(data, target)
