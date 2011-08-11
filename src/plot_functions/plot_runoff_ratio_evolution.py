__author__="huziy"
__date__ ="$Aug 9, 2011 12:09:31 PM$"

import data.data_select as data_select
import matplotlib.pyplot as plt
import application_properties
import numpy as np

import matplotlib as mpl
import diagnose_ccc.compare_swe as swe

def get_mean_for_day_of_year(stamp_dates, values):
    surfDict = {}
    for stamp_date, value in zip(stamp_dates, values):
        if surfDict.has_key(stamp_date):
            surfDict[stamp_date].append(value)
        else:
            surfDict[stamp_date] = [value]

    sortedDates = sorted(surfDict.keys())
    return sortedDates, [np.mean(surfDict[d]) for d in sortedDates]


def plot_ratio(path = 'data/streamflows/hydrosheds_euler10_spinup100yrs/aex_discharge_1970_01_01_00_00.nc'):
    res = data_select.get_data_from_file(path = path, field_name = 'surface_runoff')
    surf_runoff, times, x_indices, y_indices = res
    total_runoff = data_select.get_field_from_file(path, field_name = 'total_runoff')

    mean_surf_runoff = np.mean(surf_runoff, axis = 1) #mean in space
    mean_total_runoff = np.mean(total_runoff, axis = 1)


    stamp_dates = map(lambda d: swe.toStampYear(d, stamp_year = 2000), times)

    t1, v1 = get_mean_for_day_of_year(stamp_dates, mean_surf_runoff)
    plt.plot(t1, v1, label = 'surface runoff', linewidth = 3)

    t2, v2 = get_mean_for_day_of_year(stamp_dates, mean_total_runoff)
    plt.plot(t2, v2, label = 'total runoff', linewidth = 3)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%b'))
    plt.show()
    







if __name__ == "__main__":
    application_properties.set_current_directory()
    plot_ratio()
    print "Hello World"
