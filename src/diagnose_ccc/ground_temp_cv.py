import os
import pickle
from matplotlib import gridspec, axes
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from data import members
from util import plot_utils
import matplotlib.pyplot as plt
from plot2D import map_parameters
import matplotlib as mpl

__author__ = 'huziy'

import numpy as np
import application_properties
import ccc_util
from ccc.ccc import champ_ccc
from matplotlib_helpers import my_colormaps

from multiprocessing import Pool


def _func_for_each_process(file_path):
    """
    Takes temporal mean of the file_path
    """
    ccc_obj = champ_ccc(fichier = file_path)
    records = ccc_obj.charge_champs()
    current_fields = []
    for the_record in records:
        current_fields.append(the_record["field"])
    return np.array(current_fields).mean(axis = 0)


class CVPlotter():
    def __init__(self, member_list = None,
                       var_name = "gt",
                       data_root = "/skynet3_exec1/huziy/crcm4_data"):
        self.data_root = data_root
        self.var_name = var_name
        self.member_list = member_list
        pass



    def read_and_calculate_means(self, months = range(1,13)):
        self.mean_fields = []
        data_foder_format = os.path.join(self.data_root, "%s_p1%s")
        process_pool = Pool(processes=15)
        for the_id in self.member_list:
            folder_path = data_foder_format % ( the_id, self.var_name)
            file_paths = []
            for file_name in os.listdir(folder_path):
                #select data only for the specified months
                the_month = ccc_util.get_month_from_name(file_name=file_name)
                if the_month not in months:
                    continue
                file_paths.append(os.path.join(folder_path, file_name))
            #take means for each file corresponding to the member
            member_mean_fields = process_pool.map(_func_for_each_process, file_paths)
            self.mean_fields.append(np.array( member_mean_fields ).mean( axis = 0 ) )
        self.mean_fields = np.array( self.mean_fields )


    def get_cv_field(self):
        means = np.mean( self.mean_fields, axis = 0 )
        stds = np.std( self.mean_fields, axis = 0 )
        return stds / means


def plot_all(cv_current, cv_future):
    plot_utils.apply_plot_params(width_pt=None, font_size=9)


    fig = plt.figure()
    ps = map_parameters.polar_stereographic
    x = ps.lons
    y = ps.lats

    basemap = Basemap(projection="npstere", boundinglat = 20, lon_0=-115)
    [x, y] = basemap(x, y)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    all_axes = []
    all_images = []

    gs = gridspec.GridSpec(2,2)
    ax1 = fig.add_subplot(gs[0,0])
    all_axes.append(ax1)
    ax1.set_title("SST: CV current")
    image = basemap.pcolormesh(x, y, cv_current, ax = ax1)
    all_images.append(image)

    ax2 = fig.add_subplot(gs[0, 1])
    all_axes.append(ax2)
    ax2.set_title("SST: CV future")
    image = basemap.pcolormesh(x, y, cv_current, ax = ax2)
    all_images.append(image)

    ax3 = fig.add_subplot(gs[1, :])
    all_axes.append(ax3)
    ax3.set_title("SST: CV future - CV current")
    cMap = my_colormaps.get_red_blue_colormap(ncolors=10)
    image = basemap.pcolormesh(x, y, cv_future - cv_current, ax = ax3,
            cmap = cMap, vmin = -0.004, vmax = 0.004 )
    all_images.append(image)

    for the_ax, image  in zip( all_axes, all_images):
        divider = make_axes_locatable(the_ax)
        the_ax.set_xlim(x_min, x_max)
        the_ax.set_ylim(y_min, y_max)
        cax = divider.append_axes("right", "8%", pad="3%")
        cb = fig.colorbar(image, cax = cax, ax = the_ax)
        cb.outline.set_visible(False)
        basemap.drawcoastlines(ax = the_ax)

    fig.tight_layout()
    fig.savefig("crcm4_sst_cv.png")



def main():

    cache_file = "amno_sst_cv.bin"
    months = [3,4,5,6,7]
    cache_file = "_".join(map(str, months) + [cache_file])

    members.all_current.remove("aev")
    members.all_future.remove("aew")

    if os.path.isfile(cache_file):
        [pc, pf] = pickle.load(open(cache_file))
    else:
        pc = CVPlotter(member_list=members.all_current)
        pc.read_and_calculate_means(months = months)

        pf = CVPlotter(member_list=members.all_future)
        pf.read_and_calculate_means(months=months)
        pickle.dump([pc, pf], open( cache_file, mode="w" ))

    plot_all(pc.get_cv_field(), pf.get_cv_field())
    pass

import time
if __name__ == "__main__":
    t0 = time.clock()
    application_properties.set_current_directory()
    main()
    print "elapsed time: %f seconds" % (time.clock() - t0)
