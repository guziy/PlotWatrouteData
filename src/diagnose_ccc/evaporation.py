import os
import application_properties

__author__ = 'huziy'


from ccc.ccc import champ_ccc
import numpy as np
from datetime import timedelta
from datetime import datetime

import matplotlib.pyplot as plt
from data import quebec_route_domain_crcm4


def _get_month_from_filename(file_name):
    s = file_name.split("_")[-1].split(".")[0]
    return datetime.strptime(s, "%Y%m").month

def calculate_mean_evap_integrated_over(mask = None, data_folder = "data/ccc_data/aex/aex_p1qfs"):

    dt = timedelta( hours = 6 )  #data time step

    data_for_months = [[] for i in xrange(12) ]

    for the_file in os.listdir(data_folder):
        ccc_obj = champ_ccc( fichier = os.path.join(data_folder, the_file) )
        records = ccc_obj.charge_champs()

        the_month = _get_month_from_filename(the_file)
        #each record is a step in time
        domain_mean_series = []
        for the_record in records:
            datai = np.mean(the_record["field"][mask == 1]) #get mean for domain
            domain_mean_series.append(datai)

        data_for_months[the_month - 1].append(np.sum(domain_mean_series))

    for i in xrange(12):
        data_for_months[i] = np.mean(data_for_months[i])

    print np.sum(data_for_months)

    plt.figure()
    plt.plot(xrange(1,13), data_for_months)
    plt.savefig("evap_qc_crcm4.png")

    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    calculate_mean_evap_integrated_over(mask=quebec_route_domain_crcm4.get_domain_mask())