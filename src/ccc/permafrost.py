from matplotlib.ticker import LinearLocator

__author__ = 'huziy'


from ccc import champ_ccc

from plot2D.map_parameters import polar_stereographic
import matplotlib.pyplot as plt

import application_properties
import numpy as np
import os
from util import plot_utils
#from matplotlib.patches import Polygon
from shapely.geometry import Polygon
from shapely.geometry import Point
from osgeo import ogr
from osgeo import gdal
import matplotlib as mpl



name_to_abbr = {
    "alberta" : "AB",
    "saskatchewan" : "SK",
    "manitoba" : "MB",
    "newfoundland  & labrador" : "NL",
    "prince edward island": "PE",
    "nova scotia" : "NS",
    "northwest territories":"NT" ,
    "nunavut" : "NU",
    "ontario": "ON",
    "new brunswick" : "NB",
    "yukon territory" : "YT",
    "british columbia": "BC",
    "quebec" : "QC"
}

def plot_stations(the_basemap):
    folder_path = "data/permafrost"
    file_names = [ "stationNumbersCont.txt" , "stationNumbersDisc.txt", "stationNumbersSpor.txt" ]
    marks = ["o", "+", 'd']
    for fName, the_mark in zip(file_names, marks):
        path = os.path.join(folder_path, fName)
        lines = open(path).readlines()
        lons = []
        lats = []
        for line in lines:
            line = line.strip()
            if line == "": continue
            fields = line.split()
            lons.append(float(fields[1]))
            lats.append(float(fields[2]))


        lons = -np.array(lons)
        lons, lats = the_basemap(lons, lats)
        the_basemap.scatter(lons, lats, marker = the_mark, c = "none", s = 20, linewidth = 1)


def inside_region(lon, lat, geometries):
    """
    Check whether the point is inside the region of interest
    """
    point = ogr.CreateGeometryFromWkt(Point(lon, lat).wkt)

    for g in geometries:
        if g.Intersect(point):
            return True

    return False
    pass


def main():
    data_path = "data/permafrost/p1perma"
    cccObj = champ_ccc(data_path)
    pData = cccObj.charge_champs()[0]["field"]


    plot_utils.apply_plot_params()
    basemap = polar_stereographic.basemap


    x, y = basemap(polar_stereographic.lons, polar_stereographic.lats)
    pData =   pData[10:190,10:182]

    pData = np.ma.masked_where(pData < 0, pData)


    x1d = x[~pData.mask]
    y1d = y[~pData.mask]
    xmin = np.min(x1d)
    xmax = np.max(x1d)
    ymin, ymax = np.min(y1d), np.max(y1d)


    #basemap.contourf(x,y, pData)

#    basemap.drawcoastlines()
#    basemap.drawcountries()
#    basemap.drawstates()
#    plt.colorbar()


    path_to_shapefile = "data/shape/canada_provinces_shape/PROVINCE"
    basemap.readshapefile(path_to_shapefile, name = "provinces")
    gdal.UseExceptions()
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataStore = driver.Open(path_to_shapefile + ".SHP", 0)

    layer = dataStore.GetLayer(0)

    print dataStore.GetLayerCount()



    geometries = []
    for feature in layer:
        name = feature.GetFieldAsString('NAME').lower()

        geom = feature.GetGeometryRef()
        geometries.append(geom.ExportToWkt())
        centroid = geom.Centroid()

        lon, lat = centroid.GetX(), centroid.GetY()

        abbr = name_to_abbr[name]

        if abbr == "YT":
            lon -= 1
        if abbr == "PE":
            lon -= 1.5

        if abbr == "NB":
            lat -= 0.5
            lon -= 1.5

        if abbr == "NS":
            lat -= 0.5
            lon += 1



        if abbr == "NU":
            lat -= 8
            lon -= 10

        if abbr == "NT":
            lat -= 3

        if abbr == "NL":
            lon, lat = basemap(9.18639e6, 3.36468e6, True)

        x1, y1 = basemap(lon, lat)
        plt.annotate(name_to_abbr[name], xy = (x1,y1))
        print name


    geometries = map(ogr.CreateGeometryFromWkt, geometries)
    lons = polar_stereographic.lons
    lats = polar_stereographic.lats
    nx, ny = lons.shape
    for i in xrange(nx):
        for j in xrange(ny):
            if pData.mask[i, j]:
                continue
            else:
                pData.mask[i, j] = not inside_region(lons[i,j], lats[i,j], geometries)


    basemap.pcolormesh(x,y, pData, cmap = mpl.cm.get_cmap(name = "gist_yarg",  lut = 5),
                       alpha = 0.8, linewidth = 0, shading='flat', vmin = 0, vmax = 5)

#    basemap.contourf(x, y, pData, levels = [0.0, 1.0,2.0,3.0,4.0, 5.0],
#                     cmap = mpl.cm.get_cmap(name = "gist_yarg",  lut = 5), alpha = 0.6)
    plt.colorbar(ticks = LinearLocator(numticks = 6), format = "%.1f")
    plot_stations(basemap)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()

    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()

