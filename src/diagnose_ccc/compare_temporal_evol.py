

__author__="huziy"
__date__ ="$Oct 31, 2011 10:53:07 AM$"

import application_properties
from data import members
from diagnose_ccc import compare_swe

from ccc.ccc import champ_ccc
from datetime import datetime
from datetime import timedelta
import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl



#data_folder_format = "RUNOFF/%s/%s_p1rof"
data_folder_format = "data/ccc_data/%s/%s_p1st"

file_format = "%s_p1rof_%s.ccc"
start_date = datetime(1980, 1, 1)
end_date = datetime(1990, 12, 31)


def get_date_from_name(name = "", format = "%Y%m"):
    """
    get datetime object from the file name (represents start date for the
    data in the file)
    """
    f1 = name.split("_")[-1]
    f1 = f1.split(".")[0]
    return datetime.strptime(f1, format)


def get_mean_over_domain(the_id, stamp_year = 2001, mask = None):
    dt = members.id_to_step[the_id]
    timeToSpatialMean = {}
    folderPath = data_folder_format % (the_id, the_id)

    #get spatial mean
    for fileName in os.listdir(folderPath):
        d = get_date_from_name(name = fileName)
        #skip files from the outside of interest dates
        if d < start_date: continue
        if d > end_date: continue

        ccc_obj = champ_ccc(os.path.join(folderPath, fileName))
        fields = ccc_obj.charge_champs()

        for record in fields:
            the_field = record["field"]
            if mask is None:
                timeToSpatialMean[d] = np.mean(the_field)
            else:
                timeToSpatialMean[d] = np.mean(the_field[mask == 1])
            d += dt


    #get mean for each day of a year
    one_day = timedelta(days = 1)
    stamp_start_date = datetime(stamp_year, 1, 1)
    year_days = [stamp_start_date + i * one_day for i in xrange(365)]

    day_to_datalist = {}
    for yd in year_days:
        day_to_datalist[yd] = []


    for t, v in timeToSpatialMean.iteritems():
        #skip leap day
        if t.month == 2 and t.day == 29: continue

        t1 = datetime(stamp_year, t.month, t.day)
        day_to_datalist[t1].append(v)

    for yd in year_days:
        x = day_to_datalist[yd]
        day_to_datalist[yd] = np.mean(x)

    return year_days, [day_to_datalist[yd] for yd in year_days]


def main():

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in xrange(3):
        members.all_current.pop()

    domain_mask = compare_swe.get_domain_mask()

    #plot runoff evol for members
    for the_id in members.all_current:
        times, data = get_mean_over_domain(the_id, mask = domain_mask)
        ax.plot(times, data, "--", label = the_id)
        print "finished: %s " % the_id

    times, data = get_mean_over_domain(members.control_id, mask = domain_mask)
    ax.plot(times, data, label = members.control_id)


    ax.xaxis.set_major_locator(
            mpl.dates.MonthLocator(bymonth = range(2,13,2))
    )


    ax.xaxis.set_major_formatter(
            mpl.dates.DateFormatter('%b')
    )


    plt.legend(ncol = 3)
    plt.savefig("runoff_evol.pdf")

    pass


if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello World"

