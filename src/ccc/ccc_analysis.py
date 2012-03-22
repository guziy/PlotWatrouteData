import os

__author__ = 'huziy'

import numpy as np
from diagnose_ccc import ccc_util
from ccc import champ_ccc


def get_seasonal_means_for_year_range(data_folder, year_range, months = None):
    """
    Using the fact that the data for each month are in a separate file

    return data (n_years, nx, ny) - numpy array
    """
    path_map = ccc_util.get_yearmonth_to_path_map(data_folder)
    result = []
    for the_year in year_range:
        print("year = {0}".format(the_year))
        data = []
        for the_month in months:
            key = (the_year, the_month)
            ccc_obj = champ_ccc(path_map[key])
            records = ccc_obj.charge_champs()
            data.extend(map(lambda x: x["field"],records))
        result.append(np.mean(data, axis=0))
    return np.array(result)


def get_seasonal_mean(data_folder_path, start_date = None, end_date = None, months = None):
    """
    returns seasonal mean
    """
    result = 0.0
    record_count = 0.0
    for file_name in os.listdir(data_folder_path):
        date_month = ccc_util.get_month_date_from_name(file_name)


        #take only months of interest
        if date_month.month not in months:
            continue

        #focus only on the interval of interest
        if start_date is not None:
            if date_month < start_date:
                continue

        if end_date is not None:
            if date_month > end_date:
                continue


        data = champ_ccc(os.path.join(data_folder_path, file_name)).charge_champs()
        for record in data:
            field = record["field"]
            result += field
            record_count += 1.0

    return result / record_count



def get_monthly_normals(data_folder_path, start_date = None, end_date = None):

    """
    returns numpy array of the monthly normals of the shape (12, nx, ny)
    """
    result = [0.0 for i in xrange(12)]
    record_counts = [0.0 for i in xrange(12)]
    for file_name in os.listdir(data_folder_path):
        date_month = ccc_util.get_month_date_from_name(file_name)

        #focus only on the interval of interest
        if start_date is not None:
            if date_month < start_date:
                continue

        if end_date is not None:
            if date_month > end_date:
                continue


        data = champ_ccc(os.path.join(data_folder_path, file_name)).charge_champs()
        m = date_month.month - 1
        for record in data:
            field = record["field"]
            result[m] += field
            record_counts[m] += 1.0

    result = np.array(result)
    record_counts = [ count * np.ones(result.shape[1:]) for count in record_counts ]
    record_counts = np.array(record_counts)
    result /= np.array(record_counts)
    return result


    pass


def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  