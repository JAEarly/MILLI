import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from data.mnist_bags import create_andmil_datasets
from interpretability.instance_attribution import milli_instance_attribution as milli
from interpretability.instance_attribution.base_instance_attribution import InherentAttribution
from interpretability.base_interpretability import Model
from interpretability.interpretability_util import get_pred
from model import mnist_models

plt.style.use(['science', 'bright', 'grid'])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
COLOUR_MAP = LinearSegmentedColormap.from_list("", [colors[1], "white", colors[2]])


def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_info = Model('MI-Attn', mnist_models.MnistAttentionNN,
                       'models/mnist/MnistAttentionNN/MnistAttentionNN_{:d}.pkl')

    model = load_model(device, model_info.clz, model_info.path_fmt.format(0))

    _, _, dataset = create_andmil_datasets(8, 0, random_state=5, num_test_bags=100)

    for idx in range(len(dataset)):
        bag, label, instance_targets = dataset.get_bag_verbose(idx)
        label = int(label)
        pred = get_pred(bag, model)

        if label != 1:
            continue

        attributions = get_instance_attributions(bag, model, pred)

        methods = ['Attention', 'MILLI 1', 'MILLI 0']

        heights = [2] + [1] * len(methods)

        fig = plt.figure(constrained_layout=False, figsize=(10, 2))

        gs = GridSpec(len(methods) + 1, len(bag)+2, figure=fig, height_ratios=heights)

        instance_axes = [fig.add_subplot(gs[0, i+1]) for i in range(len(bag))]
        method_axes = [fig.add_subplot(gs[i+1, 0]) for i in range(len(methods))]
        attribution_axis = fig.add_subplot(gs[1:, 1:-1])

        draw_instances(instance_axes, bag)
        draw_method_names(method_axes, methods)
        draw_instance_attributions(attribution_axis, attributions)

        path = 'out/interpretability_out/mnist/{:d}_mnist_out_clz_{:d}.png'.format(idx, label)
        fig.savefig(path, format='png', dpi=400)

        plt.show()


def load_model(device, model_clz, path):
    model = model_clz(device)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model


def get_instance_attributions(bag, model, original_pred):
    attention = InherentAttribution()
    shap = milli.Milli(n_samples=150, alpha=0.05, beta=0.01, method='single')
    attributions = [
        attention.get_instance_clz_attributions(bag, model, original_pred, 0),
        # shap.get_instance_clz_attributions(bag, model, original_pred, 3),
        # shap.get_instance_clz_attributions(bag, model, original_pred, 2),
        shap.get_instance_clz_attributions(bag, model, original_pred, 1),
        shap.get_instance_clz_attributions(bag, model, original_pred, 0),
    ]
    attributions = np.asarray(attributions)
    return attributions


def draw_instances(instance_axes, instances):
    for i, instance in enumerate(instances):
        instance_axes[i].imshow(instance.squeeze(), cmap='gray')
        configure_axis(instance_axes[i])


def draw_method_names(methods_axis, interpretability_names):
    for i, name in enumerate(interpretability_names):
        methods_axis[i].annotate(name, (1.0, 0.4), ha='right', va='center')
        configure_axis(methods_axis[i])


def draw_instance_attributions(score_axis, interpretability_scores):
    c = score_axis.pcolor(np.flip(interpretability_scores, axis=0), cmap=COLOUR_MAP,
                          vmin=-1, vmax=1, edgecolors='k', linewidths=1)
    configure_axis(score_axis)
    return c


def configure_axis(axis):
    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.set_yticks([])
    axis.set_yticklabels([])
    axis.axis('off')


if __name__ == "__main__":
    run()
