from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import torchvision
import random
import numpy
import torch
import tqdm
import os

from sklearn.model_selection import train_test_split
from loss_functions import LSRLoss, SparseLSRLoss

# Use the GPU/CUDA when available, else use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensuring PyTorch gives deterministic output.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():

    # Loading the CIFAR-10 dataset into memory.
    training, validation, testing = cifar10()

    # Creating the different loss functions.
    ce = torch.nn.CrossEntropyLoss()
    lsr = LSRLoss(smoothing=0.25)
    sparse_lsr = SparseLSRLoss(smoothing=0.25)

    # Running a training session using different loss functions.
    for loss_function, file_name in zip([ce, lsr, sparse_lsr],
                ["Cross Entropy", "Label Smoothing", "Sparse Label Smoothing"]):

        # Setting the reproducibility seed in PyTorch.
        torch.cuda.manual_seed(0)
        torch.manual_seed(0)
        numpy.random.seed(0)
        random.seed(0)

        # Creating the AlexNet model.
        model = AlexNet().to(device)

        # Training the model using a basic training loop.
        train(model, training, loss_function, gradient_steps=150000)

        # Calculating the training and testing accuracy.
        training_accuracy = evaluate(model, training)
        testing_accuracy = evaluate(model, testing)
        print("Training Accuracy:", round(training_accuracy, 4), "%")
        print("Testing Accuracy:", round(testing_accuracy, 4), "%")

        # Plotting the penultimate layer representation using TSNE.
        plot_penultimate_representation(model, testing, file_name, testing_accuracy)


def cifar10():

    """
    Loading the CIFAR-10 dataset into memory using TorchVision. Using
    the same partitioning strategy as what is used in the original
    papers figures.
    """

    # Finding the current directory of this file.
    directory = os.path.dirname(os.path.realpath(__file__)) + "/datasets/"

    # Loading the CIFAR-10 training and testing sets.
    training = torchvision.datasets.CIFAR10(
        train=True, root=directory, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    )
    testing = torchvision.datasets.CIFAR10(
        train=False, root=directory, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    )

    # Partitioning the data into training and validation sets.
    train_indices, val_indices, _, _ = train_test_split(
        range(len(training)),
        training.targets,
        stratify=training.targets,
        test_size=0.1,
    )

    validation = torch.utils.data.Subset(training, val_indices)
    training = torch.utils.data.Subset(training, train_indices)

    return training, validation, testing


class AlexNet(torch.nn.Module):

    def __init__(self):

        """
        Implementation of AlexNet from the paper "ImageNet Classification
        with Deep Convolutional Neural Networks" by Alex Krizhevsky, Ilya
        Sutskever and Geoffrey E. Hinton.
        """

        super(AlexNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 192, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 2 * 2, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, 10)
        )

        # Initializing the weights of the network.
        self.reset()

    def reset(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight)
                module.bias.data.zero_()

    def forward(self, x):
        return self.network(x)


def train(model, dataset, loss_function, gradient_steps):

    """
    A vanilla training loop which uses stochastic gradient descent to learn the
    parameters of the base network, using the given pytorch loss function.

    :param model: Base network used for the given task.
    :param dataset: PyTorch Dataset containing the training data.
    :param loss_function: Loss function to minimize.
    :param gradient_steps: Number of gradient steps.
    """

    # Creating an optimizer object for learning the model parameter's.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.0005)

    # Creating a multi-step learning rate scheduler to decay the learning rate.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000, 135000], gamma=0.1)

    # Creating a DataLoader to generate samples/batches for each tasks.
    generator = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    # Looping until the maximum number of gradient steps is reached.
    for step in (training_progress := tqdm.tqdm(range(gradient_steps))):

        # Clearing the gradient cache.
        optimizer.zero_grad()

        # Sampling a batch and sending it to the device.
        X, y = next(iter(generator))
        X, y = X.to(device), y.to(device)

        # Performing inference and computing the loss.
        y_pred = model(X)
        loss = loss_function(y_pred, y)

        # Performing the backward pass and gradient step/update.
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Recording the training accuracy.
        accuracy = (y_pred.argmax(dim=1) == y).float().mean().item() * 100
        training_progress.set_description("Accuracy: " + str(round(accuracy, 4)))


def evaluate(model, dataset, batch_size=100):

    """
    Performs inference on the provided model, and computes the
    performance using the provided performance metric.

    :param model: Base network used for the given task.
    :param dataset: PyTorch DataLoader used for evaluation.
    :param batch_size: Batch size used for inference.
    """

    # Creating a PyTorch dataloader object for generating batches.
    task = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pred_labels, true_labels = [], []

    model.eval()  # Switching network to inference mode.
    with torch.no_grad():  # Disabling gradient calculations.

        # Iterating over the whole dataset in batches.
        for instances, labels in task:
            yp = model(instances.to(device))
            pred_labels.append(yp)
            true_labels.append(labels.to(device))

    # Converting the list to a PyTorch tensor.
    pred_labels = torch.cat(pred_labels)
    true_labels = torch.cat(true_labels)

    model.train()  # Switching network back to training mode.

    # Returning the performance of the trained model.
    accuracy = (pred_labels.argmax(dim=1) == true_labels).float().mean().item()
    return accuracy * 100


def plot_penultimate_representation(model, dataset, filename, performance):

    """
    Function for visualizing the penultimate representation of a neural network
    using t-distributed Stochastic Neighbor Embedding (TSNE).

    :param model: Base network used for the given task.
    :param dataset: PyTorch DataLoader used for evaluation.
    :param filename: Name of the file to save the figure to.
    :param performance: The final inference performance of the model.
    """

    plt.clf()

    # Adding a title to the plot showing the method and performance.
    plt.title(filename) #+ " (" + str(round(performance, 4)) + "%" + ")")

    # Converting the datasets into a DataLoader.
    generator = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Generating a forward hook to obtain the penultimate layers output.
    model.network[-2].register_forward_hook(get_activation('encoder_penultimate_layer'))

    model.eval()  # Switching network to inference mode.
    with torch.no_grad():  # Disabling gradient calculations.

        pred_labels, true_labels = [], []  # List for recording the results.

        # Iterating over the whole dataset in batches.
        for instances, labels in generator:
            model(instances.to(device))
            pred_labels.append(activation['encoder_penultimate_layer'])
            true_labels.append(labels.to(device))

        # Converting the list to a PyTorch tensor.
        pred_labels = torch.cat(pred_labels)
        true_labels = torch.cat(true_labels)

    # Moving predictions to the cpu and converting into numpy arrays.
    pred_labels = pred_labels.cpu().detach().numpy()
    true_labels = true_labels.cpu().detach().numpy()

    # Plotting the penultimate layer representation using TSNE.
    embedding = TSNE(n_components=2).fit_transform(pred_labels)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=true_labels, s=2)
    plt.savefig(filename.lower().replace(" ", "-") + ".png")


if __name__ == '__main__':
    main()
