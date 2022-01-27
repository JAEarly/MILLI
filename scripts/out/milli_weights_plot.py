import numpy as np
from matplotlib import pyplot as plt
from interpretability.instance_attribution.milli_instance_attribution import milli_func, milli_integral


plt.style.use(['science', 'bright', 'grid'])


def plot_weights():
    n_instances = 31

    alphas = [0.1, 0.3, 0.8]
    betas = [1.0, 0.1, 0.0, -0.1, -1.0]

    fig, axes = plt.subplots(nrows=1, ncols=len(alphas), figsize=(7, 2))
    for i, alpha in enumerate(alphas):
        axis = axes[i]
        for beta in betas:
            xs = np.linspace(0, n_instances-1, int(1e2))
            milli_weights = generate_weights(xs, n_instances, alpha, beta)
            integral = milli_integral(n_instances, alpha, beta)
            print('Mean:', milli_weights.mean())
            print('Integral:', integral)
            print('Expectation:', integral/(n_instances - 1))
            print('Diff:', milli_weights.mean() - integral/(n_instances - 1))
            axis.plot(xs, milli_weights, label=r'$\beta={:}$'.format(beta))

        axis.set_xlim(0, n_instances - 1)
        axis.set_ylim(0, 1)
        if i == 1:
            axis.set_xlabel(r'$r_i$')
        axis.set_title(r'$\alpha={:}$'.format(alpha))
        axis.set_xticklabels([])

    fig.supylabel(r'$\pi_R$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    plt.tight_layout(rect=[0, 0, 0.84, 1])
    fig.savefig("out/milli_weights.png", format='png', dpi=400)
    plt.show()


def plot_expected_coalition_size():
    n_instances = 31

    alphas = [0.1, 0.3, 0.5, 0.6, 0.8, 1.0]
    betas = np.linspace(-4, 4, int(1e3))

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(6, 2.2))
    for i, alpha in enumerate(alphas):
        expectations = []
        for beta in betas:
            integral = milli_integral(n_instances, alpha, beta)
            expectation = integral/(n_instances - 1)
            expectations.append(expectation)
        axis.plot(betas, expectations, label=r'$\alpha={:}$'.format(alpha))

    axis.set_xlim(betas[0], betas[-1])
    axis.set_xlabel(r'$\beta$')
    axis.set_ylim(0, 1)
    axis.set_ylabel(r'$\mathop{\mathbb{E}}[|z|]/k$')

    axis.legend(loc='best', ncol=2)
    plt.tight_layout()

    fig.savefig("out/milli_expected_coalition_sizes.png", format='png', dpi=400)
    plt.show()


def generate_weights(xs, n_instances, alpha, beta):
    weights = np.zeros(len(xs))
    for i, x in enumerate(xs):
        w = milli_func(x, n_instances, alpha, beta)
        weights[i] = w
    return weights


if __name__ == "__main__":
    # plot_weights()
    plot_expected_coalition_size()
