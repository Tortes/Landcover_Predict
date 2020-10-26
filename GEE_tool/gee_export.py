import ee
ee.Initialize()
import geetools
import yaml
import numpy as np
from datetime import date
from math import pi, cos, acos

EARTH_RADIUS = 6378137

def get_longlati(start_point, dist):
    x1, y1 = map(lambda x: x*pi/180,start_point)
    y2 = y1 - dist/EARTH_RADIUS
    A = cos(-dist/EARTH_RADIUS)
    B = (1-A)/cos(y2)**2
    C = acos(1-B)
    x2 = x1 - C
    return x2*180/pi,y2*180/pi

def export_image(start_point):
    # ## Define an ImageCollection
    mask_s2_cloud = geetools.cloud_mask.sentinel2(['cloud'])
    mask_s2_cirrus = geetools.cloud_mask.sentinel2(['cirrus'])
    dist = 5120
    x1, y1 = start_point
    x2, y2 = get_longlati(start_point, dist)
    target_rectangle = ee.Geometry.Rectangle([x2,y2,x1,y1])
    collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                    .filterDate('2019-06-01', '2019-06-30') \
                    .median()
                    

    # Set parameters
    scale = 10
    name_pattern = str(x1)+"-"+str(y1)
    date_pattern = 'ddMMMy'
    folder = 'MYFOLDER_image'
    region = target_rectangle

    tasks = geetools.batch.Export.imagecollection.toDrive(
                collection=ee.ImageCollection(collection),
                folder=folder,
                region=region,
                namePattern=name_pattern,
                scale=scale,
                datePattern=date_pattern,
                verbose=True,
                maxPixels=int(1e13)
            )

def export_DEM(start_point):
    dist = 5120
    x1, y1 = start_point
    x2, y2 = get_longlati(start_point, dist)
    target_rectangle = ee.Geometry.Rectangle([x2,y2,x1,y1])
    image = ee.Image('CGIAR/SRTM90_V4').select('elevation')
    slope = ee.Terrain.slope(image)
    collection = ee.ImageCollection(slope)

    scale = 10 
    name_pattern = str(x1)+"-"+str(y1)
    date_pattern = 'ddMMMy'
    folder = 'MYFOLDER_elevation'
    region = target_rectangle

    tasks = geetools.batch.Export.imagecollection.toDrive(
                collection=collection,
                folder=folder,
                region=region,
                namePattern=name_pattern,
                scale=scale,
                datePattern=date_pattern,
                verbose=True,
                maxPixels=int(1e13)
            )

def export_DEM_(start_point):
    dist = 5120
    x1, y1 = start_point
    x2, y2 = get_longlati(start_point, dist)
    target_rectangle = ee.Geometry.Rectangle([x2,y2,x1,y1])
    image = ee.Image('CGIAR/SRTM90_V4').select('elevation')
    # slope = ee.Terrain.slope(image)
    collection = ee.ImageCollection(image)

    scale = 10 
    name_pattern = str(x1)+"-"+str(y1)
    date_pattern = 'ddMMMy'
    folder = 'MYFOLDER_elevation'
    region = target_rectangle

    tasks = geetools.batch.Export.imagecollection.toDrive(
                collection=collection,
                folder=folder,
                region=region,
                namePattern=name_pattern,
                scale=scale,
                datePattern=date_pattern,
                verbose=True,
                maxPixels=int(1e13)
            )

if __name__=='__main__':
    with open('pointlist.yaml', 'r') as f:
        yf = yaml.load(f)
        yf_arr = list(yf.values())
        for point in yf_arr:
            result = export_image(point)
            elevation = export_DEM(point)
