

__author__ = 'huziy'

import application_properties
from data import members
import os
from data import data_select
import pickle
import numpy as np
import bootstrap_for_mean
import matplotlib.pyplot as plt



def generate_indices_restrict_data_to_member(nvalues, nvalues_per_member):
    result = []
    min_index = 0
    max_index = nvalues_per_member - 1
    n_members = nvalues / nvalues_per_member
    assert nvalues % nvalues_per_member == 0
    for i in xrange(n_members):
        result.extend(np.random.randint(min_index, high=max_index + 1, size=nvalues_per_member))
        min_index += nvalues_per_member
        max_index += nvalues_per_member
    return result



def _get_cache_file_path(months = None, sim_ids = None):
    months_str = map( str, months )
    the_list = ["merged", "std"]
    the_list.extend(months_str)
    the_list.extend(sim_ids)
    return os.path.join("tmp", "_".join(the_list))
    pass


def save_means_to_csv(data_matrix, sim_id = ""):
    fName = "%s_means.csv" % sim_id
    f = open(fName, mode="w")
    for pos in xrange(data_matrix.shape[1]):
        means_str = map(str, data_matrix[:, pos])
        f.write(",".join(means_str) + "\n")
    f.close()

    pass

def get_std_and_mean_using_bootstrap_for_merged_means(sim_ids = None, folder_path = "data/streamflows/hydrosheds_euler9",
                                     months = range(1, 13), n_samples = 1000):

    """
    returns the object containing means for the domain and standard deviations from bootstrap
    """
    cache_file = _get_cache_file_path(months=months, sim_ids = sim_ids)
    if os.path.isfile(cache_file):
       return pickle.load(open(cache_file))


    #determine path to the file with data
    filePaths = []
    for f in os.listdir(folder_path):
        if f.split("_")[0] in sim_ids:
            filePath = os.path.join(folder_path, f)
            filePaths.append(filePath)



    boot_means = []
    real_means = []
    index_matrix = None

    all_means = []
    members_boot_means = []
    for file_path in filePaths:
        streamflow, times, i_indices, j_indices = data_select.get_data_from_file(file_path)

        #for each year and for each gridcell get mean value for the period
        means_dict = data_select.get_means_over_months_for_each_year(times, streamflow, months = months)

        means_sorted_in_time = map( lambda x : x[1], sorted(means_dict.items(), key=lambda x: x[0]) )
        data_matrix = np.array(means_sorted_in_time)


        real_means.append(data_matrix) #save modelled means, in order to calculate mean of the merged data
        #print "data_matrix.shape = ", data_matrix.shape
        boot_means = []
        for i in xrange(n_samples):
            #generate indices
            index_vector = np.random.randint(0, data_matrix.shape[0], data_matrix.shape[0])

            #average 30 bootstrapped annual means
            boot_means.append( np.mean(data_matrix[index_vector,:], axis = 0) )
    
        members_boot_means.append( boot_means )
    
    #take average over members
    print np.array(members_boot_means).shape
    boot_means = np.array(members_boot_means).mean(axis = 0) #nsamples x npoints

    print boot_means[:, 499]
    print boot_means[:, 19]
    assert boot_means.shape[0] == n_samples, boot_means.shape

    print "boot_means.shape = ", boot_means.shape
    std_result = np.std(boot_means, axis = 0)
    mean_result = np.array(real_means).mean(axis = 0).mean(axis = 0)
    pickle.dump([std_result, mean_result], open(cache_file, mode="w"))
    return std_result, mean_result


def get_significance_for_change_in_mean_of_merged_over_months(months = range(1, 13)):
    """
    returns boolean vector of the size n_grid_cells where True means
    significant change and False not significant
    and
    the percentage of changes with respect to the current mean
    """

    current_stds, current_means = get_std_and_mean_using_bootstrap_for_merged_means(sim_ids = members.all_current,
                                                                                    months = months, n_samples=1000)

    future_stds, future_means = get_std_and_mean_using_bootstrap_for_merged_means(sim_ids = members.all_future,
                                                                                  months = months, n_samples=1000)




    current_half_interval = 1.96 * current_stds
    future_half_interval = 1.96 * future_stds

    is_sign = np.absolute(future_means - current_means) > current_half_interval + future_half_interval

    print "number of significant points = ", sum( map(int, is_sign) )
    return is_sign, (future_means - current_means) / current_means * 100.0



    pass

def main():
    get_significance_for_change_in_mean_of_merged_over_months()
    #do_bootstrap_for_simulation_mean(n_samples=10)
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
