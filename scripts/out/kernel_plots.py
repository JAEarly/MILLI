import numpy as np
import interpretability.instance_attribution.lime_instance_attribution as lime
from interpretability.instance_attribution.shap_instance_attribution import shap_kernel
from functools import partial
from matplotlib import pyplot as plt
import math

# plt.style.use(['science', 'bright', 'grid'])


def run():
    n_instances = 30

    l2_kernel_width = lime.get_l2_kernel_width(n_instances)
    cosine_kernel_width = lime.get_cosine_kernel_width(n_instances)

    print('L2 kernel width: {:.2f}'.format(l2_kernel_width))
    print('Cosine kernel width: {:.2f}'.format(cosine_kernel_width))

    l2_sample_weights = generate_sample_weights(partial(lime.lime_kernel, dist_func=lime.l2_dist,
                                                        kernel_width=l2_kernel_width), n_instances)
    cosine_sample_weights = generate_sample_weights(partial(lime.lime_kernel, dist_func=lime.cosine_dist,
                                                            kernel_width=cosine_kernel_width), n_instances)
    shap_sample_weights = generate_sample_weights(partial(shap_kernel, n_instances), n_instances)

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    xs = np.asarray(range(1, n_instances))
    # axis.plot(xs, l2_sample_weights, label='LIME L2')
    # axis.plot(xs, cosine_sample_weights, label='LIME Cosine')
    axis.plot(xs, shap_sample_weights, label='SHAP')
    axis.set_xlim(1, n_instances - 1)
    axis.set_xlabel('|z|')
    axis.set_ylabel('Sample weight')
    axis.legend(loc='best')
    axis.set_yscale('log')

    # plot_lime_varying_kernel_width(axes[1][0], n_instances, l2_dist,
    #                                sorted([0.5, 0.75, 1.0, 2.5, 5.0, l2_kernel_width]))
    # axes[1][0].legend(loc='best')
    #
    # plot_lime_varying_kernel_width(axes[1][1], n_instances, cosine_dist,
    #                                sorted([0.1, 0.25, 0.5, 0.75, 1.0, 5.0, cosine_kernel_width]))
    # axes[1][1].legend(loc='best')

    plt.show()


def plot_lime_varying_kernel_width(axis, n_instances, dist_func, kernel_widths):
    for w in kernel_widths:
        sample_weights = generate_sample_weights(partial(lime_kernel, dist_func=dist_func, kernel_width=w), n_instances)
        xs = np.asarray(range(1, n_instances))
        axis.plot(xs, sample_weights, label='{:}'.format(w))


def generate_sample_weights(kernel_func, n_instances):
    sample_weights = np.zeros(n_instances - 1)
    for i in range(1, n_instances):
        coalition = np.zeros(n_instances)
        coalition[:i] = 1
        try:
            sample_weights[i - 1] = kernel_func(coalition)
        except TypeError:
            sample_weights[i - 1] = kernel_func((coalition == 1).sum())
    return sample_weights


if __name__ == "__main__":
    run()
