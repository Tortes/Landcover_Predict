import os
import rasterio
import numpy as np
from rasterio.transform import Affine

'''
Function to generate target dummy geotiff dfc label.
Input:  dfc_img: Path to save the target geotiff.
                Example: "/home/tortes/Desktop/new.tif"
        dfc_size: Size of target geotiff.(image will be size*size)
                Example: 256
        dfc_class: Target class of the whole dfc image.
                Example: 9
'''
def generate_barren_label(dfc_img, dfc_size, dfc_class):
    barren_array = np.ones((dfc_size,dfc_size), dtype=np.uint8) * dfc_class

    with rasterio.open(dfc_img, 'w', 
                        driver='GTiff', 
                        height=dfc_size, width=dfc_size,
                        count=1,
                        dtype=np.uint8) as dst:
        dst.write(barren_array, 1)

'''
Function to clip s2 data.
Input:  data_path: path to origin s2 data.
        output_path: path to save the slipped s2 data.

'''
def clip_s2():
    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)
        outputpath = os.path.join(output_path, filename)

        # Generate img
        transform = Affine.translation(1, 1)
        with rasterio.open(filepath, 'r') as data:
            img = data.read()
            with rasterio.open(outputpath, 'w',
                                driver='GTiff',
                                height=dfc_size, width=dfc_size,
                                count=23,
                                dtype=np.float32,
                                transform=transform) as dst:
                dst.write(img)
