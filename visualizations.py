import matplotlib.pyplot as plt
import matplotlib
import numpy

# =================================================================
# Functions for visualizing the different loss functions. Note,
# the loss functions are not defined in their numerically stable
# form. For visualization purposes only.
# =================================================================


def cross_entropy_loss(y, fx, smoothing, C=100):
    return - y * numpy.log(fx)


def label_smoothing_loss(y, fx, smoothing, C=100):
    return - (y * (1 - smoothing) + smoothing/C) * numpy.log(fx)


def sparse_label_smoothing_loss(y, fx, smoothing, C=100):
    # return - y * (numpy.log(fx) + smoothing * numpy.log(1 - fx))
    return - y * ((1 - smoothing + smoothing / C) * numpy.log(fx) +
                  (smoothing * (C - 1) / C) * numpy.log((1 - fx) / (C - 1)))


def plot_loss_function(loss_function, n_samples=1000):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Settings used for visualizing the loss functions.
    fx = numpy.linspace(0.01, 0.99, n_samples)
    smoothing_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
    colors = matplotlib.cm.get_cmap("copper_r")(numpy.linspace(0, 1, len(smoothing_values)))

    # Plotting the target loss function, i.e., y = 1.
    y = numpy.ones(n_samples)
    for index, smoothing in enumerate(smoothing_values):
        axs[0].plot(fx, loss_function(y, fx, smoothing=smoothing), linewidth=4,
                   c=colors[index], label="$\\xi = $" + str(smoothing))

    axs[0].set_title("Target Loss ($y=1$)")
    axs[0].set_xlabel("Prediction")
    axs[0].set_ylabel("Loss")
    axs[0].grid()
    axs[0].legend()

    # Plotting the non-target loss function, i.e., y = 0.
    y = numpy.zeros(n_samples)
    for index, smoothing in enumerate(smoothing_values):
        axs[1].plot(fx, loss_function(y, fx, smoothing=smoothing), linewidth=4,
                    c=colors[index], label="$\\xi = $" + str(smoothing))

    axs[1].set_title("Non-Target Loss ($y=0$)")
    axs[1].set_xlabel("Prediction")
    axs[1].set_ylabel("Loss")
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_loss_function(cross_entropy_loss)
    plot_loss_function(label_smoothing_loss)
    plot_loss_function(sparse_label_smoothing_loss)
