import os
import cv2
import rasterio
import numpy as np
from itertools import product
import pickle as pkl
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from models.deeplab import DeepLab
from models.unet import UNet

import config

class Terrain:
    def __init__(self, image_path, elevation_path, DEM_path):
        self.image_path = image_path
        self.elevation_path = elevation_path
        self.DEM_path = DEM_path
        self.image = self.get_image()
        self.elevation = self.get_elevation()
        self.height, self.width, _ = self.elevation.shape

    def get_image(self):
        if not os.path.exists(self.image_path):
            raise Exception("No image path")

        with rasterio.open(self.image_path) as data:
            s2 = data.read(config.band_true)
            s2_mx, s2_my = map(round, [s2.shape[1]/2, s2.shape[2]/2])

        s2 = s2.astype(np.float32)
        s2 = np.clip(s2, 0, 10000)
        s2 /= 10000
        s2 = s2[:, s2_mx - config.ppi_size : s2_mx + config.ppi_size,
                   s2_my - config.ppi_size : s2_my + config.ppi_size]
        s2 = np.rollaxis(s2, 0, 3)

        return s2 * config.brightness_factor


    def get_predict_image(self):
        if not os.path.exists(self.image_path):
            raise Exception("No image path")

        with rasterio.open(self.image_path) as data:
            s2 = data.read(config.band_choose)
            s2_mx, s2_my = map(round, [s2.shape[1]/2, s2.shape[2]/2])

        s2 = s2.astype(np.float32)
        s2 = np.clip(s2, 0, 10000)
        s2 /= 10000
        s2 = s2[:, s2_mx - config.ppi_size : s2_mx + config.ppi_size,
                   s2_my - config.ppi_size : s2_my + config.ppi_size]
        return s2


    def get_elevation(self):
        if not os.path.exists(self.elevation_path):
            raise Exception("No elevation path")

        with rasterio.open(self.elevation_path) as data:
            elevation = data.read([1])
            ele_mx, ele_my = map(round, [elevation.shape[1]/2, elevation.shape[2]/2])
        
        elevation = elevation[:, ele_mx - config.ppi_size : ele_mx + config.ppi_size,
                                 ele_my - config.ppi_size : ele_my + config.ppi_size]
        elevation = np.rollaxis(elevation, 0, 3)

        return elevation


    def get_DEM(self):
        if not os.path.exists(self.DEM_path):
            raise Exception("No elevation path")

        with rasterio.open(self.DEM_path) as data:
            elevation = data.read([1])
            ele_mx, ele_my = map(round, [elevation.shape[1]/2, elevation.shape[2]/2])
        
        elevation = elevation[:, ele_mx - config.ppi_size : ele_mx + config.ppi_size,
                                 ele_my - config.ppi_size : ele_my + config.ppi_size]

        return elevation[0]


    def steepness_filter(self):
        height, width, _ = self.elevation.shape
        bmap = self.elevation.copy()
        bmap[self.elevation > config.slope_threshold] = 255
        bmap[self.elevation <= config.slope_threshold] = 0

        floodfill_map = bmap.copy()
        mask = np.zeros((height+2, width+2), np.uint8)
        cv2.floodFill(floodfill_map, mask, (0,0), 70)

        floodfill_map[floodfill_map != 70] = 255
        floodfill_map[floodfill_map == 70] = 0

        # imgplot = plt.imshow(floodfill_map)
        # plt.show()

        return floodfill_map


    def landcover_filter(self):
        train_args = pkl.load(open(config.pkl_path, "rb"))

        # Use GPU
        use_gpu = False
        if torch.cuda.is_available():
            use_gpu = True
            if torch.cuda.device_count() > 1:
                raise NotImplementedError("multi GPU Error")
        
        # Setup network
        n_classes = len(config.classnames) - 1
        n_inputs = len(config.band_choose)
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
        if use_gpu:
            model = model.cuda()
        
        # Restore weights from file
        state = torch.load(config.model_path)
        step = state["step"]
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        print("loaded checkpoint from step", step)

        # Load image
        img_raw = self.get_predict_image()
        img_input = torch.from_numpy(img_raw)
        img_input = img_input.repeat(16,1,1,1)

        if use_gpu:
            img_input = img_input.cuda()
        
        # Predict
        with torch.no_grad():
            prediction = model(img_input)
        
        prediction = prediction.cpu().numpy()
        prediction = np.argmax(prediction, axis=1)

        # Savanna data
        pre_copy = prediction.copy()
        prediction[pre_copy>2] += 1

        return prediction[0]


    def mix_filter(self):
        steep = self.steepness_filter()
        landcover = self.landcover_filter()
        height, width, _ = steep.shape
        mix_result = np.ones((height,width), dtype=np.uint8)*255
        for i,j in product(range(height), range(width)):
            if steep[i][j] == 0 and landcover[i][j] == 8:
                mix_result[i][j] = 0
        return steep, landcover, mix_result



def main():
    cnt = 11
    for filename in os.listdir(config.image_path):
        file_image = os.path.join(config.image_path, filename)
        file_elevation = os.path.join(config.elevation_path, filename)
        file_DEM = os.path.join(config.DEM_path, filename)

        # Create folder
        folder_path = os.path.join(config.output_path,str(cnt))
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        

        T = Terrain(file_image, file_elevation, file_DEM)
        E, L, R = T.mix_filter()
        DEM = T.get_DEM()
        print(filename)

        fig, ax = plt.subplots(1,1)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        fig.set_size_inches(T.height/100.0, T.width/100.0)

        ax.imshow(np.clip(T.image, 0, 1), aspect='equal')
        fig.savefig(os.path.join(folder_path, 'a.png'))

        ax.imshow(config.cmap(L), aspect='equal')
        fig.savefig(os.path.join(folder_path, 'b.png'))

        ax.imshow(T.elevation, cmap="gray", aspect='equal')
        fig.savefig(os.path.join(folder_path, 'c.png'))

        ax.imshow(E, cmap="gray", aspect='equal')
        fig.savefig(os.path.join(folder_path, 'd.png'))

        ax.imshow(R, cmap="gray", aspect='equal')
        fig.savefig(os.path.join(folder_path, 'e.png'))

        ax.imshow(DEM, cmap="gray", aspect='equal')
        fig.savefig(os.path.join(folder_path, 'dem.png'))

        MASK = np.asarray(L)
        DEM_array = np.asarray(DEM)
        DEM_64array = np.zeros((64,64), dtype=np.float32)
        for i,j in product(range(64), repeat=2):
            DEM_64array[i][j] = DEM_array[8*i][8*j]

        DEM_256 = np.zeros((256,256), dtype=np.float32)
        for i,j in product(range(256), repeat=2):
            DEM_256[i][j] = DEM_array[2*i][2*j]
        MASK_256 = np.zeros((256,256), dtype=np.float32)
        for i,j in product(range(256), repeat=2):
            MASK_256[i][j] = MASK[2*i][2*j]
        
        np.savetxt(os.path.join(folder_path, 'DEM512x512.csv'), DEM_array, delimiter=',')
        np.savetxt(os.path.join(folder_path, 'DEM64x64.csv'), DEM_64array, delimiter=',')
        np.savetxt(os.path.join(folder_path, 'MASK512x512.csv'), MASK, delimiter=',')
        np.savetxt(os.path.join(folder_path, 'DEM256x256.csv'), DEM_256, delimiter=',')
        np.savetxt(os.path.join(folder_path, 'MASK256x256.csv'), MASK_256, delimiter=',')


        plt.close(fig)

        cnt += 1


def main_show():
    for filename in os.listdir(config.image_path):
        file_image = os.path.join(config.image_path, filename)
        file_elevation = os.path.join(config.elevation_path, filename)
        file_DEM = os.path.join(config.DEM_path, filename)

        T = Terrain(file_image, file_elevation, file_DEM)
        E, L, R = T.mix_filter()
        print(filename)

        plt.rcParams["figure.figsize"] = [16,9]
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
        ax1.imshow(np.clip(T.image, 0, 1))
        ax2.imshow(config.cmap(L))
        # ax2.imshow(np.clip(T.image, 0, 1))
        ax3.imshow(T.elevation, cmap="gray")
        ax4.imshow(E, cmap="gray")
        ax5.imshow(R, cmap="gray")
        plt.show()



main()
# main_show() 