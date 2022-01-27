import matplotlib.image as mpimg
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from interpretability.instance_attribution.milli_instance_attribution import Milli
from data.crc.crc_dataset import load_crc
from PIL import Image
from interpretability.metrics import get_top_k_mask, MetricError
from model import crc_models
from interpretability.base_interpretability import Model
from interpretability.crc_interpretability import CrcInterpretabilityStudy
from interpretability.interpretability_util import get_pred


plt.rcParams['text.usetex'] = True

COLOUR_MAP = LinearSegmentedColormap.from_list("", ["red", "white", "green"])


def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes = 2

    model_info = Model('MI-Attn', crc_models.CrcAttentionNN,
                       'models/crc/CrcAttentionNN/CrcAttentionNN_{:d}.pkl')
    model = load_model(device, model_info.clz, model_info.path_fmt.format(0))
    dataset, _, _ = load_crc(augment_train=False, random_state=868)
    milli = Milli(n_samples=1000, alpha=0.008, beta=-5, method='single')

    for idx, (bag, target) in enumerate(dataset):
        instance_paths = dataset.bags[idx]
        instance_targets = dataset.instance_targets[idx]
        bag_id = dataset.ids[idx]
        target = int(target)
        pred = get_pred(bag, model)
        pred_clz = torch.argmax(pred).item()

        if bag_id not in [1, 4, 9, 69]:
            continue

        print('Bag ID: img{:d}'.format(bag_id))
        print(pred)
        print('  Label: {:d}'.format(target))
        print('   Pred: {:d}'.format(pred_clz))

        orig_image = get_original_image(bag_id)

        stitched_image, clz_masks, idx_to_pos = create_stitched_image(instance_paths, instance_targets, num_classes)
        clz_img = create_highlighted_image(stitched_image, clz_masks[target])

        interpretability_img = create_interpretation_image(stitched_image, idx_to_pos, instance_targets, milli,
                                                           bag, model, pred, target)

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        axes[0].imshow(orig_image)
        hide_axis(axes[0])
        axes[1].imshow(stitched_image)
        hide_axis(axes[1])
        axes[2].imshow(clz_img)
        hide_axis(axes[2])
        axes[3].imshow(interpretability_img)
        hide_axis(axes[3])
        plt.tight_layout()

        path = 'out/interpretability_out/crc/img{:d}_crc_out.png'.format(bag_id)
        fig.savefig(path, format='png')#, dpi=1200)

        # del fig

        plt.show()


def get_original_image(bag_id):
    path = "data/CRC/orig/img{:d}/img{:d}.bmp".format(bag_id, bag_id)
    img = mpimg.imread(path)
    return img


def create_stitched_image(instance_paths, instance_targets, n_classes):
    """
    Stitch instances together to make one whole image.

    Also returns a binary mask of which instances exist.
    """
    stitched_image = np.ones((486, 486, 3), dtype=int) * 220
    clz_masks = []
    for i in range(n_classes):
        clz_masks.append(np.zeros((18, 18), dtype=int))

    idx_to_pos = {}
    for instance_idx, path in enumerate(instance_paths):
        file_name = path[path.rindex('/') + 1:]
        i1 = file_name.index('_')
        i2 = file_name.rindex('_')
        i3 = file_name.index('.')
        px = int(file_name[i1 + 1:i2])
        py = int(file_name[i2 + 1:i3])
        targets = instance_targets[instance_idx]

        with open(path, 'rb') as f:
            img = Image.open(f)
            instance = np.asarray(img.convert('RGB'), dtype=float)
            # if not targets:
            #     instance *= 0.8
            stitched_image[px * 27:(px + 1) * 27, py * 27:(py + 1) * 27, :] = instance

        for clz in targets:
            clz_masks[clz][px, py] = 1

        idx_to_pos[instance_idx] = [px, py]

    stitched_image = np.uint8(stitched_image)
    return stitched_image, clz_masks, idx_to_pos


def create_interpretation_image(stitched_image, idx_to_pos, instance_targets, method, bag, model, orig_pred, clz,
                                k=None):
    instance_attributions = method.get_instance_clz_attributions(bag, model, orig_pred, clz)
    clz_instance_targets = CrcInterpretabilityStudy.get_clz_target_mask(instance_targets, clz)
    mask = np.zeros((18, 18), dtype=int)
    try:
        top_k_idxs, _, _ = get_top_k_mask(instance_attributions, clz_instance_targets, k=k)
        for idx in top_k_idxs:
            x, y = idx_to_pos[idx]
            mask[x, y] = 1
    except MetricError:
        pass
    return create_highlighted_image(stitched_image, mask)


def create_highlighted_image(stitched_image, mask):
    highlighted_image = np.ones((486, 486, 3), dtype=int) * 220
    for r in range(18):
        for c in range(18):
            hv = mask[r, c]
            existing_patch = stitched_image[r * 27:(r + 1) * 27, c * 27:(c + 1) * 27]
            if hv == 1:
                highlighted_image[r * 27:(r + 1) * 27, c * 27:(c + 1) * 27] = existing_patch
            else:
                highlighted_image[r * 27:(r + 1) * 27, c * 27:(c + 1) * 27] = existing_patch * 0.5
    highlighted_image = highlighted_image.astype(int)
    return highlighted_image


def load_model(device, model_clz, path):
    model = model_clz(device, 2)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model


def hide_axis(axis):
    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.set_yticks([])
    axis.set_yticklabels([])
    axis.axis('off')


if __name__ == "__main__":
    run()
