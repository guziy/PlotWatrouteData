import os
from vcs.slabapi import mask
from diagnose_ccc import compare_swe

__author__ = 'huziy'

from data import data_select
import matplotlib.pyplot as plt
import numpy as np
import application_properties
import itertools


class FirstYearData:
    def __init__(self, path = "", spinup_years = None):
        self.spinup_years = spinup_years
        dataCollection = data_select.get_data_from_file(path = path)
        self.data = dataCollection[0]
        self.times = dataCollection[1]
        self.x_indices = dataCollection[2]
        self.y_indices = dataCollection[3]
        self.label = "%d years" % spinup_years
        self._select_first_year_data()

    def _select_first_year_data(self):
        """
        selects data and times corresponding to the
        first simulation year
        """
        the_year = self.times[0].year
        all_years = map(lambda x: x.year, self.times)
        all_years = np.array(all_years)

        to_select = (all_years == the_year)

        self.data = self.data[ to_select, : ]
        self.times = list(itertools.ifilter(lambda x : x.year == the_year,  self.times ))
        self.first_year = the_year
        pass


    def get_spatial_mean(self, mask = None):
        if mask is None:
            return np.mean(self.data, axis = 1)
        else:
            mask_1d = np.zeros((self.data.shape[1],))
            mask_1d[:] = mask[self.x_indices, self.y_indices]
            return np.mean(self.data[:, mask_1d == 1], axis = 1)
            pass


class Plotter:

    def __init__(self, first_year_data_list = None):
        self.first_year_data_list = first_year_data_list

    def plot_all_on_one(self, domain_mask = None):
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)
        ax.set_title("first simulation year, using different spinup times \n mean over simulation domain ")
        for fyd in self.first_year_data_list:
            ax.plot(fyd.times, fyd.get_spatial_mean(mask = domain_mask), lw = 3, label = fyd.label)
        ax.legend()
        plt.show()
        pass


    @classmethod
    def test_class_method(cls):
        print "Hi from the class method"

    @classmethod
    def main(cls, path_to_spinups_dir = "data/spinup_testing"):
        data_list = []

        domain_mask = compare_swe.get_domain_mask()

        for fileName in os.listdir(path_to_spinups_dir):
            spinup = fileName.split("_")[-1].split("yrs")[0]
            spinup = int(spinup)
            filePath = os.path.join(path_to_spinups_dir, fileName)
            data_list += [FirstYearData(path=filePath, spinup_years=spinup)]
        data_list.sort(key = lambda x: x.spinup_years)

        plotter = Plotter(first_year_data_list=data_list)
        plotter.plot_all_on_one(domain_mask=domain_mask)





if __name__ == "__main__":
    application_properties.set_current_directory()
    Plotter.main()
    Plotter.test_class_method()