import os
import sys
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from PIL import Image
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import rasterio
from models.deeplab import DeepLab
from models.unet import UNet

np.set_printoptions(threshold=sys.maxsize)

brightness_factor = 3
bands = []
S2_BANDS_SHOW = [2, 3, 4]
S2_BANDS_HR = [2, 3, 4, 8]
# S2_BANDS_HR = [2, 3, 4]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [1, 10, 11]

DFC2020_CLASSES = [
    0,  # class 0 unused in both schemes
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    3,  # --> will be masked if no_savanna == True
    3,  # --> will be masked if no_savanna == True
    4,
    5,
    9,  # 12 --> 6
    7,  # 13 --> 7
    9,  # 14 --> 6
    8,
    9,
    10
    ]

def classnames():
    return ["Forest", "Shrubland", "Savanna", "Grassland", "Wetlands",
            "Croplands", "Urban/Built-up", "Snow/Ice", "Barren", "Water"]

def showcmap():
    show_cmap = colors.ListedColormap(['#009900',
                                    '#c6b044',
                                    '#fbff13',
                                    '#b6ff05',
                                    '#27ff87',
                                    '#c24f44',
                                    '#a5a5a5',
                                    '#69fff8',
                                    '#f9ffa4',
                                    '#1c0dff',
                                    '#ffffff'])
    return show_cmap

def mycmap():
    cmap = colors.ListedColormap(['#009900',
                                  '#c6b044',
                                  '#fbff13',
                                  '#b6ff05',
                                  '#27ff87',
                                  '#1c0dff',
                                  '#a5a5a5',
                                  '#69fff8',
                                  '#f9ffa4',
                                  '#1c0dff',
                                  '#ffffff'])
    return cmap

def mypatches():
    patches = []
    for counter, name in enumerate(classnames()):
        patches.append(mpatches.Patch(color=showcmap().colors[counter],
                                      label=name))
    return patches

def get_parser():
    # define and parse arguments
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--config_file', type=str, default="args.pkl",
                        help='path to config file (default: ./args.conf)')
    parser.add_argument('--checkpoint_file', type=str, default="checkpoint.pth",
                        help='path to checkpoint file (default: ./checkpoint.pth)')

    # general
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for prediction (default: 32)')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers for dataloading (default: 4)')
    parser.add_argument('--score', action='store_true', default=False,
                        help='score prediction results using ground-truth data')

    # data
    parser.add_argument('--dataset', type=str, default="sen12ms_holdout",
                        choices=['sen12ms_holdout', 'dfc2020_val', 'dfc2020_test',
                                'tiff_dir'],
                        help='type of dataset (default: sen12ms_holdout)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='path to dataset')
    parser.add_argument('--out_dir', type=str, default="results",
                        help='path to output dir (default: ./results)')
    parser.add_argument('--preview_dir', type=str, default=None,
                        help='path to preview dir (default: no previews)')

    # validation
    parser.add_argument('--validation_folder', type=str, default=None,
                        help='path to validation')
    

    
    return parser.parse_args()

def get_image(path):
    with rasterio.open(path) as data:
        s2 = data.read(bands)
        s2_output = data.read(S2_BANDS_SHOW)
        s2 = np.clip(s2.astype(np.float32), 0, 10000)
        s2_output = np.clip(s2_output.astype(np.float32), 0, 10000)
        s2 /= 10000
        s2_output /= 10000
        s2 = s2[:,:256,:256]
        s2_output = s2_output[:,:256,:256]
    assert s2.size, s2_output.size
    return s2, s2_output

def get_val(path):
    with rasterio.open(path) as data:
        dfc = data.read(1)
        dfc = dfc - 1
    return dfc

def get_bands(s2hr, s2mr, s2lr):
    bands = []
    if s2hr:
        bands = bands + S2_BANDS_HR
    if s2mr:
        bands = bands + S2_BANDS_MR
    if s2lr:
        bands = bands + S2_BANDS_LR
    return sorted(bands)    

def main():
    args = get_parser()

    # load config
    train_args = pkl.load(open(args.config_file, "rb"))

    global bands
    bands = get_bands(train_args.use_s2hr, train_args.use_s2mr,train_args.use_s2lr)

    # set flags for GPU processing if available
    if torch.cuda.is_available():
        args.use_gpu = True
        if torch.cuda.device_count() > 1:
            raise NotImplementedError("multi-gpu prediction not implemented! "
                                    + "try to run script as: "
                                    + "CUDA_VISIBLE_DEVICES=0 predict.py")
    else:
        args.use_gpu = False

    # set up network
    n_classes = len(classnames())-1
    n_inputs = len(bands)
    if train_args.model == "deeplab":
        model = DeepLab(num_classes=n_classes,
                        backbone='resnet',
                        pretrained_backbone=False,
                        output_stride=train_args.out_stride,
                        sync_bn=False,
                        freeze_bn=False,
                        n_in=n_inputs)
    else:
        model = UNet(n_classes=n_classes,
                    n_channels=n_inputs)
    if args.use_gpu:
        model = model.cuda()

    # restore network weights
    state = torch.load(args.checkpoint_file)
    step = state["step"]
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print("loaded checkpoint from step", step)

    # Get image
    for filename in os.listdir(args.data_dir):
        filepath = os.path.join(args.data_dir, filename)
        img_raw,img_out = get_image(filepath)
        img = torch.from_numpy(img_raw)
        img = img.repeat(16,1,1,1)

        # GPU
        if args.use_gpu:
            img = img.cuda()

        # predict
        with torch.no_grad():
            prediction = model(img)

        # convert to 256x256 numpy arrays
        prediction = prediction.cpu().numpy()
        prediction = np.argmax(prediction, axis=1)
        copy = prediction.copy()
        prediction[copy>=2] += 1
        cmap = mycmap()
        prediction_out = cmap(prediction)[0]

        # validation
        validation = False
        if args.validation_folder:
            validation = True
            val_path = os.path.join(args.validation_folder, filename).replace("_s2_","_dfc_")
            val_img = get_val(val_path)
            val_img = cmap(val_img)

        # Plot
        img_out = np.rollaxis(img_out, 0, 3)[:,:,::-1]
        if not validation:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(np.clip(img_out* brightness_factor, 0, 1))
            ax2.imshow(prediction_out)
            lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0),
                            handles=mypatches(), ncol=1, title="DFC Classes")
            plt.tight_layout()
            ax1.axis('off')
            ax2.axis('off')
            plt.rcParams["figure.figsize"] = [16,9]
            plt.show()
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(np.clip(img_out* brightness_factor, 0, 1))
            ax2.imshow(prediction_out)
            ax3.imshow(val_img)
            lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0),
                            handles=mypatches(), ncol=1, title="DFC Classes")
            plt.tight_layout()
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            plt.rcParams["figure.figsize"] = [16,9]
            plt.show()

main()

