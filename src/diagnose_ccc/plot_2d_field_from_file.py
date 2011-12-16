__author__ = 'huziy'

import application_properties
from ccc import ccc
import matplotlib.pyplot as plt
from plot2D.map_parameters import polar_stereographic

def plot_fields_in_file(path = "data/ccc_data/aex/aex_p1gt/aex_p1gt_198001.ccc"):
    cccObj = ccc.champ_ccc( fichier = path)
    fields = cccObj.charge_champs()
    basemap = polar_stereographic.basemap
    x = polar_stereographic.xs
    y = polar_stereographic.ys

    for the_record in fields:
        plt.figure()
        basemap.pcolormesh(x, y, the_record["field"])
        plt.colorbar()
        plt.show(block = True)

    pass


def main():
    plot_fields_in_file()
    pass


if __name__ == "__main__":
    application_properties.set_current_directory()
    main()