import csv
import math
import random

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from data.sival import sival_dataset
from interpretability.instance_attribution.milli_instance_attribution import Milli
from interpretability.base_interpretability import Model
from interpretability.interpretability_util import get_pred
from interpretability.sival_interpretability import SivalInterpretabilityStudy
from model import sival_models

n_classes = 25
input_size = 30


def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_info = Model('MI-Attn', sival_models.SivalAttentionNN,
                       'models/sival/SivalAttentionNN/SivalAttentionNN_{:d}.pkl')
    model = load_model(device, model_info.clz, model_info.path_fmt.format(0))
    dataset, _, _ = sival_dataset.create_datasets(random_state=868)
    milli = Milli(n_samples=200, alpha=0.05, beta=-0.01, method='single')
    # loop_dataset(dataset, model, weighted_shap)
    # output_specific(dataset, model, milli, [2, 45], 'wd40can')
    # output_specific(dataset, model, milli, [265, 533], 'cokecan')
    output_specific(dataset, model, milli, [447, 680], 'dataminingbook')
    # output_specific(dataset, model, milli, [514, 660], 'goldmedal')


def output_specific(dataset, model, weighted_shap, idxs, name):
    fig, axes = plt.subplots(nrows=len(idxs), ncols=3, figsize=(8, 4))
    for i in range(len(idxs)):
        render_output(dataset, model, weighted_shap, idxs[i], axes[i])
    plt.tight_layout()

    path = 'out/interpretability_out/sival/{:s}_sival_out.png'.format(name)
    fig.savefig(path, format='png')#, dpi=400)

    plt.show()


def loop_dataset(dataset, model, weighted_shap):
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    for idx in idxs:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        render_output(dataset, model, weighted_shap, idx, axes)
    plt.show()


def render_output(dataset, model, weighted_shap, idx, axes):
    bag_name = dataset.bag_names[idx]
    bag, label, instance_targets = dataset.get_bag_verbose(idx)
    label = int(label.item())
    pred = get_pred(bag, model)
    pred_clz = torch.argmax(pred).item()

    label_name = 'negative' if label == 0 else sival_dataset.positive_clzs[label - 1]
    print('Idx: {:d} Bag Name: {:s}'.format(idx, bag_name))
    print('  Label: {:s}'.format(label_name))
    print('   Pred: {:s}'.format('negative' if pred_clz == 0 else sival_dataset.positive_clzs[pred_clz - 1]))

    orig_img = dataset.get_img_from_name(bag_name)

    instance_locs = parse_instance_locs(dataset, bag_name)
    gt_clz_targets = SivalInterpretabilityStudy.get_clz_target_mask(instance_targets, label)
    num_positive = np.count_nonzero(gt_clz_targets)

    gt_mask = create_ground_truth_images(bag_name, dataset, instance_locs, gt_clz_targets)
    positive_mask = create_model_interpretation_image(weighted_shap, bag, model, pred, pred_clz,
                                                      instance_locs, num_positive)

    gt_img = mul_mask_with_orig(gt_mask, orig_img)
    positive_img = mul_mask_with_orig(positive_mask, orig_img)

    axes[0].imshow(orig_img)
    configure_axis(axes[0])
    axes[1].imshow(gt_img)
    configure_axis(axes[1])
    axes[2].imshow(positive_img)
    configure_axis(axes[2])


def load_model(device, model_clz, path):
    model = model_clz(device)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model


def create_ground_truth_images(bag_name, dataset, instance_locs, gt_clz_targets):
    _, label, instance_targets = dataset.get_bag_from_name(bag_name)
    mask_img = create_binary_image(instance_locs, gt_clz_targets)
    return mask_img


def create_model_interpretation_image(method, bag, model, pred, clz, instance_locs, num_positive):
    instance_attributions = method.get_instance_clz_attributions(bag, model, pred, clz)
    instance_binary_predictions = np.zeros_like(instance_attributions)
    top_n_idxs = np.argsort(instance_attributions)[::-1][:num_positive].copy()
    for i, idx in enumerate(top_n_idxs):
        w = 1/math.log2(i + 2)
        instance_binary_predictions[idx] = w
    mask_img = create_binary_image(instance_locs, instance_binary_predictions)
    return mask_img


def parse_instance_locs(dataset, bag_name):
    file_path = dataset.get_path_from_name("data/SIVAL/bags", bag_name, '.imbag')
    instance_locs = {}
    instance_idx = 0
    parsing = False
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=' ')
        for line in reader:
            if parsing:
                if len(line) == 0:
                    parsing = False
                else:
                    instance_locs[instance_idx] = []
                    for b in line:
                        try:
                            instance_locs[instance_idx].append(int(b))
                        except ValueError:
                            pass
                    instance_idx += 1
            if len(line) > 0 and line[0] == 'Blocks:':
                parsing = True
            if parsing and len(line) == 0:
                parsing = False
    return instance_locs


def create_binary_image(instance_locs, instance_binary_labels):
    image_arr = np.zeros((256, 192))
    for instance_idx, locs in instance_locs.items():
        segment_colour = instance_binary_labels[instance_idx]
        for loc in locs:
            loc = int(loc)
            x = loc % 256
            y = loc // 256
            image_arr[x, y] = segment_colour * 255
    image_arr = image_arr.swapaxes(0, 1)
    im = Image.fromarray(np.uint8(image_arr), 'L')
    return im


def mul_mask_with_orig(mask_img, orig_img, bg_weight=0):
    # Scale mask size up
    mask_scaled_img = mask_img.resize((orig_img.width, orig_img.height))
    # Reduce to 0-1 rather than 0-255
    mask_arr = np.asarray(mask_scaled_img) / 255
    mask_arr = mask_arr * (1.0 - bg_weight) + bg_weight
    # Create 3 channels so we can multiply
    mask_arr_3d = np.stack((mask_arr, mask_arr, mask_arr), axis=2)
    # Multiply and create image
    orig_arr = np.asarray(orig_img)
    final_arr = mask_arr_3d * orig_arr
    final_img = Image.fromarray(np.uint8(final_arr))
    return final_img


def configure_axis(axis):
    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.set_yticks([])
    axis.set_yticklabels([])
    axis.axis('off')



if __name__ == "__main__":
    run()
