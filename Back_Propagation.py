import numpy as np
import matplotlib.pyplot as plt
from utils import loadMNISTLabels, loadMNISTImages
from ff import FF

DIGITS_NUM = 10


def print_num(sample, title, plot):
    c = plt.imshow(sample.reshape((28, 28)))
    plot.colorbar(c)
    plot.title(title, fontweight="bold")
    plot.show()


def get_all_digit_samples(x_set, y_set, digit):
    y_set = y_set.T
    mask = y_set[:, digit] == 1
    all_relevant = x_set[:, mask]
    return all_relevant


def choose_n_digit_samples(x_set, y_set, digit, samples_num):
    all_relevant = get_all_digit_samples(x_set, y_set, digit)
    random_indices = np.random.choice(all_relevant.shape[0], size=samples_num, replace=False)
    random_rows = all_relevant[:, random_indices]
    return random_rows


def predict(random_rows, samples_num, n_net):
    all_prediction = []
    for i in range(samples_num):
        # print_num(random_rows[:, i])
        prediction = n_net.predict(random_rows[:, i])
        res = np.where(prediction == np.amax(prediction))
        all_prediction.append(res[0][0])
    return all_prediction


def prepare(x_set, y_set, samples_num, n_net):
    all_prediction = []
    all_sample = None
    for i in range(DIGITS_NUM):
        random_rows = choose_n_digit_samples(x_set, y_set, i, samples_num)
        if all_sample is None:
            all_sample = random_rows
        else:
            all_sample = np.hstack((all_sample, random_rows))

        prediction = predict(random_rows, samples_num, n_net)
        all_prediction.extend(prediction)

    return all_sample, all_prediction


def display(all_sample, all_prediction):
    columns = rows = 10
    figure = plt.figure(figsize=(17, 17))

    for i in range(columns * rows):
        pic = figure.add_subplot(rows, columns, i+1)
        plt.imshow(all_sample[:, i].reshape((28, 28)))
        pic.title.set_text(str(all_prediction[i]))
        pic.axes.get_xaxis().set_visible(False)
        pic.axes.get_yaxis().set_visible(False)
    plt.show()


# Loading the dataset
y_test = loadMNISTLabels('../MNIST_data/t10k-labels-idx1-ubyte')
y_train = loadMNISTLabels('../MNIST_data/train-labels-idx1-ubyte')

X_test = loadMNISTImages('../MNIST_data/t10k-images-idx3-ubyte')
X_train = loadMNISTImages('../MNIST_data/train-images-idx3-ubyte')


# random permutation of the input
# uncomment this to use a fixed random permutation of the images
# perm = np.random.permutation(784)
# X_test = X_test[perm, :]
# X_train = X_train[perm, :]

# Parameters
layers_sizes = [784, 30, 10]
epochs = 10
eta = 0.1
batch_size = 20

# Training
net = FF(layers_sizes)
steps, test_acc = net.sgd(X_train, y_train, epochs, eta, batch_size, X_test, y_test)

all_samples, all_predictions = prepare(X_train, y_train, 10, net)
display(all_samples, all_predictions)

# plotting learning curve and visualizing some examples from test set
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xlabel("#Step")
plt.ylabel("Test Accuracy")
plt.plot(steps, test_acc, 'm')
plt.title("Test Accuracy As Function Of #Steps")
plt.show()

