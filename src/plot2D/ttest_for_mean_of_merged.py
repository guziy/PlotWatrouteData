import os
import application_properties

__author__ = 'huziy'
#merges data from all simulation members and calculates significance
#of changes using ttest approach
from data import members
import numpy as np
from data import data_select
from scipy import stats

def get_significance_and_changes_for_months(months = range(1, 13),
                                            folder_path = "data/streamflows/hydrosheds_euler9"
                                            ):
    """
    returns boolean vector of the size n_grid_cells where True means
    significant change and False not significant
    and
    the percentage of changes with respect to the current mean
    """

    current_means = []
    future_means = []


    id_to_file_path = {}
    for the_id in members.all_members:
        for file_name in os.listdir(folder_path):
            if file_name.startswith( the_id ):
                id_to_file_path[the_id] = os.path.join(folder_path, file_name)




    for the_id in members.all_current:
        current_file = id_to_file_path[the_id]
        streamflows, times, i_indices, j_indices = data_select.get_data_from_file(current_file)
        current_mean_dict = data_select.get_means_over_months_for_each_year(times, streamflows, months=months)
        current_means.extend(current_mean_dict.values())

        future_file = id_to_file_path[members.current2future[the_id]]
        streamflows, times, i_indices, j_indices = data_select.get_data_from_file(future_file)
        future_mean_dict = data_select.get_means_over_months_for_each_year(times, streamflows, months=months)
        future_means.extend(future_mean_dict.values())




    current_means = np.array( current_means )
    future_means = np.array( future_means )



    print future_means.shape

    t, p = stats.ttest_ind(current_means, future_means, axis = 0)

    is_sign = p < 0.05 #significance to the 5% confidence level

    current_means = np.mean(current_means, axis=0)
    future_means = np.mean(future_means, axis=0)

    print future_means.shape
    print "number of significant points = ", sum( map(int, is_sign) )
    return is_sign, (future_means - current_means) / current_means * 100.0

    pass

def main():
    get_significance_and_changes_for_months()

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()