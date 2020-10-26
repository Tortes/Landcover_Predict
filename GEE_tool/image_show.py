import os
import argparse
import rasterio
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

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

cmap = colors.ListedColormap(['#009900',
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

ppisize = 256

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sentinel_path', type=str, default=None,
                        help='Path to Sentinel 2 data')
    parser.add_argument('--elevation_path', type=str, default=None,
                        help='Path to elevation data')
    parser.add_argument('--dfc_label_path', type=str, default=None,
                        help='Path to the dfc label')
    parser.add_argument('--mode', type=str, default='s2',
                        help='Select display mode.(s2, dfc, elevation, s2&elevation, s2&dfc)')
    return parser.parse_args()

def show_s2():
    band_true = [4,3,2]
    band_ultre = [4,3,2]
    brightness_factor = 1

    for filename in os.listdir(arg.sentinel_path):
        filepath = os.path.join(arg.sentinel_path, filename)
        with rasterio.open(filepath) as data:
            s2 = data.read(band_true)
            s2u = data.read(band_ultre)

        s2 = s2.astype(np.float32)
        s2 = np.clip(s2, 0, 10000)
        s2 /= 10000
        s2 = s2.astype(np.float32)
        s2 = s2[:,:256,:256]
        s2 = np.rollaxis(s2, 0, 3)

        s2u = s2u.astype(np.float32)
        s2u = np.clip(s2u, 0, 10000)
        s2u /= 10000
        s2u = s2u.astype(np.float32)
        s2u = np.rollaxis(s2u, 0, 3)


        plt.rcParams["figure.figsize"] = [16,9]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ttl = fig.suptitle(filename, y=1)
        ax1.imshow(np.clip(s2 * 3, 0, 1))
        ax2.imshow(np.clip(s2u * 3 , 0, 1))
        plt.show()

def show_dfc():
    for filename in os.listdir(arg.dfc_label_path):
        filepath = os.path.join(arg.dfc_label_path, filename)
        with rasterio.open(filepath) as data:
            dfc = data.read([1])

            dfc = cmap(dfc[0])
        
        plt.rcParams["figure.figsize"] = [16,9]
        fig, (ax1) = plt.subplots(1, 1)
        ttl = fig.suptitle(filename, y=1)
        ax1.imshow(dfc)
        plt.show()

def show_elevation():
    for filename in os.listdir(arg.elevation_path):
        heightpath = os.path.join(arg.elevation_path, filename)
        with rasterio.open(heightpath) as data:
            height = data.read([1])
            # height = height[:,:ppisize*3,:ppisize*3]
            height = np.rollaxis(height, 0, 3)

        plt.rcParams["figure.figsize"] = [16,9]
        fig, (ax1) = plt.subplots(1,1)
        ttl = fig.suptitle(filename, y=1)
        ax1.imshow(height, cmap='gray')
        plt.show()

def show_both():
    band_true = [4,3,2]
    brightness_factor = 3

    for filename in os.listdir(arg.sentinel_path):
        filepath = os.path.join(arg.sentinel_path, filename)
        heightpath = os.path.join(arg.elevation_path, filename)
        with rasterio.open(filepath) as data:
            s2 = data.read(band_true)
        with rasterio.open(heightpath) as data:
            height = data.read([1])
            height = np.rollaxis(height, 0, 3)
        
        s2 = s2.astype(np.float32)
        s2 = np.clip(s2, 0, 10000)
        s2 /= 10000
        s2 = s2.astype(np.float32)
        s2 = np.rollaxis(s2, 0, 3)


        plt.rcParams["figure.figsize"] = [16,9]
        fig, (ax1, ax2) = plt.subplots(1,2)
        ttl = fig.suptitle(filename, y=1)
        ax1.imshow(np.clip(s2*brightness_factor, 0, 1))
        ax1.axis('off')
        ax2.imshow(height, cmap='gray')
        ax2.axis('off')
        ttl = fig.suptitle(filename, y=1)
        plt.show()

def show_dfc_val():
    band_true = [4,3,2]
    brightness_factor = 3

    for filename in os.listdir(arg.sentinel_path):
        filename_dfc = filename.replace("_s2_","_dfc_")
        filepath = os.path.join(arg.sentinel_path, filename)
        heightpath = os.path.join(arg.dfc_label_path, filename_dfc)
        with rasterio.open(filepath) as data:
            s2 = data.read(band_true)
        with rasterio.open(heightpath) as data:
            height = data.read([1])
            height = height[0,:256,:256]
        
        assert s2.size, height.size
        s2 = s2.astype(np.float32)
        s2 = np.clip(s2, 0, 10000)
        s2 /= 10000
        s2 = s2[:,:256,:256]
        s2 = np.rollaxis(s2, 0, 3)

        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(np.clip(s2*brightness_factor, 0, 1))
        ax1.axis('off')
        ax2.imshow(cmap(height-1))
        ax2.axis('off')
        plt.rcParams["figure.figsize"] = [16,9]
        plt.show()
        
        # delete_key = input("wait for key, y to delete\n")
        # if delete_key=='y':
        #     if os.path.exists(filepath):
        #         os.remove(filepath)
        #         print("Delete file")
        #     if os.path.exists(heightpath):
        #         os.remove(heightpath)
        #         print("Delete height map")
        
        # plt.close('all')

def delete_blue():
    band_true = [4,3,2]
    brightness_factor = 3

    plt.ion()

    for filename in os.listdir(folder_path):
        filename_dfc = filename.replace("_s2_","_dfc_")
        filepath = os.path.join(folder_path, filename)
        heightpath = os.path.join(dfc_label_path, filename_dfc)
        with rasterio.open(heightpath) as data:
            height = data.read([1])
            height = height[0,:256,:256]
        
        assert height.size
        
        if len(height[height==10]) < 40000:
            if os.path.exists(filepath):
                os.remove(filepath)
                print("Delete file")
            if os.path.exists(heightpath):
                os.remove(heightpath)
                print("Delete height map")

if __name__ == "__main__":
    global arg
    arg = get_parser()
    if arg.mode=="s2" and arg.sentinel_path:
        show_s2()
    elif arg.mode=="dfc" and arg.dfc_label_path:
        show_dfc()
    elif arg.mode=="s2&elevation" and arg.sentinel_path and arg.elevation_path:
        show_both()
    elif arg.mode=="s2&dfc" and arg.sentinel_path and arg.dfc_label_path:
        show_dfc_val()
    else:
        raise Exception("Mode Error")
