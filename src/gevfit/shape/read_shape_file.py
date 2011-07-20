__author__="huziy"
__date__ ="$12 nov. 2010 19:19:48$"


from osgeo import ogr
from osgeo import osr


import numpy as np

from matplotlib.patches import Polygon
from shapely.wkt import loads
#import mapscript

import application_properties
application_properties.set_current_directory()

def get_features_from_shape(basemap, path = 'data/shape/contour_bv_MRCC/Bassins_MRCC_utm18.shp', linewidth = 2):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataStore = driver.Open(path, 0)
    layer = dataStore.GetLayer(0)
    latlong = osr.SpatialReference()
    latlong.ImportFromProj4("+proj=latlong")
    result = []

    feature = layer.GetNextFeature()
    while feature:
        geom = feature.GetGeometryRef()
        geom.TransformTo(latlong)

        polygon = loads(geom.ExportToWkt())
        boundary = polygon.exterior
        coords = np.zeros(( len(boundary.coords), 2))
        for i, the_coord in enumerate(boundary.coords):
            coords[i, 0], coords[i, 1] = basemap( the_coord[0], the_coord[1] )

        result.append(Polygon(coords, facecolor = 'none', linewidth = linewidth))
        feature = layer.GetNextFeature()


    dataStore.Destroy()
    return result


    

if __name__ == "__main__":
    print "Hello World"
