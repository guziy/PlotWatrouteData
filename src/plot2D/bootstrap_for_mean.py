__author__ = 'huziy'

import application_properties
from data import members
import os
from data import data_select
import pickle
import numpy as np
import matplotlib.pyplot as plt

class MeansAndDeviation():
    def __init__(self, sim_id = "", means_for_domain = None, standard_devs_for_domain = None):
        self.sim_id = sim_id
        self.means_for_domain = means_for_domain
        self.standard_devs_for_domain = standard_devs_for_domain
        pass


def _get_cache_file_path(sim_id = "", months = range(1,13)):
    """
    get file path with standard deviations calculated using bootstrap, and saved
    """
    the_list = [sim_id, "std"]
    the_list.extend( map( str, months ) )
    fPath = os.path.join("tmp", "_".join(the_list) + ".bin")
    return fPath

def do_bootstrap_for_simulation_mean(sim_id = "aet", folder_path = "data/streamflows/hydrosheds_euler9",
                                     months = range(1, 13), n_samples = 1000):

    """
    returns the object containing means for the domain and standard deviations from bootstrap
    """
    cache_file = _get_cache_file_path(sim_id=sim_id, months=months)
    if os.path.isfile(cache_file):
       return pickle.load(open(cache_file))


    #determine path to the file with data
    filePath = None
    for f in os.listdir(folder_path):
        if f.startswith(sim_id):
            filePath = os.path.join(folder_path, f)
            break

    streamflow, times, i_indices, j_indices = data_select.get_data_from_file(filePath)

    #for each year and for each gridcell get mean value for the period
    means_dict = data_select.get_means_over_months_for_each_year(times, streamflow, months = months)

    means_sorted_in_time = map( lambda x : x[1], sorted(means_dict.items(), key=lambda x: x[0]) )
    data_matrix = np.array(means_sorted_in_time)
    print "data_matrix.shape = ", data_matrix.shape

    #generate indices
    index_matrix = np.random.rand(n_samples, data_matrix.shape[0])
    index_matrix *= (data_matrix.shape[0] - 1)
    index_matrix =  index_matrix.round().astype(int)

    means_matrix = np.zeros((n_samples, streamflow.shape[1])) #n_samples x n_points
    for i in xrange(n_samples):
        means_matrix[i,:] = np.mean(data_matrix[index_matrix[i,:],:], axis = 0)


    m_holder = MeansAndDeviation(sim_id=sim_id,
                                 means_for_domain=np.mean(data_matrix, axis = 0),
                                 standard_devs_for_domain=np.std(means_matrix, axis = 0))

    pickle.dump(m_holder, open(cache_file, mode="w"))
    return m_holder


def get_significance_for_change_in_mean_over_months(months = range(1, 13)):
    """
    returns boolean vector of the size n_grid_cells where True means
    significant change and False not significant
    and
    the percentage of changes with respect to the current mean
    """

    current_means = []
    current_stds = []

    future_means = []
    future_stds = []

    for the_id in members.all_current:
        current_obj = do_bootstrap_for_simulation_mean(sim_id=the_id, months=months)
        future_obj = do_bootstrap_for_simulation_mean(sim_id=members.current2future[the_id], months=months)

        current_means.append(current_obj.means_for_domain)
        future_means.append(future_obj.means_for_domain)

        current_stds.append(current_obj.standard_devs_for_domain)
        future_stds.append(future_obj.standard_devs_for_domain)

    current_means = np.array( current_means )
    current_stds = np.array( current_stds )

    future_means = np.array( future_means )
    future_stds = np.array( future_stds )


    current_means = np.mean(current_means, axis=0)
    future_means = np.mean(future_means, axis=0)

    current_half_interval = 1.96 * np.mean(current_stds, axis = 0)
    future_half_interval = 1.96 * np.mean(future_stds, axis = 0)


    is_sign = np.absolute(future_means - current_means) > current_half_interval + future_half_interval

    print "number of significant points = ", sum( map(int, is_sign) )
    return is_sign, (future_means - current_means) / current_means * 100.0



    pass

def main():
    get_significance_for_change_in_mean_over_months()
    #do_bootstrap_for_simulation_mean(n_samples=10)
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()


