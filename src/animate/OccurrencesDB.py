import os.path
__author__="huziy"
__date__ ="$19-May-2011 11:33:24 PM$"



import data.data_select as data_select

import data.members as members

import application_properties
application_properties.set_current_directory()
import numpy as np
from datetime import timedelta
import os
from itertools import ifilter

class YearPositionObject():
    def __init__(self, year, pos):
        self.dates = []
        self.keys = (year, pos)

    def get_position_index(self):
        return self.keys[1]

    def get_year(self):
        return self.keys[0]

  #override hashing methods to use in dictionary
    def __hash__(self):
        return self.keys.__hash__()

    def __eq__(self, other):
        if other is None:
            return False

        for the_key, other_key in zip(self.keys, other.keys):
            if the_key != other_key:
                return False
        return True


def measure_of_time_to_event(current_time, event_time):
    dt = current_time - event_time
    return 100.0 / (1.0 + dt.days ** 2)

def date_time_to_day_of_year(d):
    return d.timetuple().tm_yday

class OccurrencesDB():
    def __init__(self, path_to_streamflow = 'data/streamflows/hydrosheds_euler9'):

        self.x_indices_name = 'x-index'
        self.y_indices_name = 'y-index'
        self.data_name = 'water_discharge'

        self.current_model_data = []
        self.future_model_data = []

        current_file_name_pattern = '%s_discharge_1970_01_01_00_00.nc'
        future_file_name_pattern = '%s_discharge_2041_01_01_00_00.nc'


        #share time and spatial dimensions
        self.c_times = None
        self.f_times = None

        self.i_indices = None
        self.j_indices = None

        for c_id, f_id in zip(members.current_ids, members.future_ids):
            c = ModelDataObject(member_id = c_id)
            f = ModelDataObject(member_id = f_id)

            self.current_model_data.append(c)
            self.future_model_data.append(f)

            c_file_path = os.path.join(path_to_streamflow, current_file_name_pattern % c_id)
            f_file_path = os.path.join(path_to_streamflow, future_file_name_pattern % f_id)

            if self.c_times is None:
                c.init_from_path(c_file_path)
                f.init_from_path(f_file_path)

                self.c_times = c.times
                self.f_times = f.times
                self.i_indices = c.i_indices
                self.j_indices = c.j_indices
            else:
                c.set_data_from_path(path = c_file_path)
                f.set_data_from_path(path = f_file_path)

                c.times = self.c_times
                f.times = self.f_times

                c.i_indices = self.i_indices
                c.j_indices = self.j_indices

                f.i_indices = self.i_indices
                f.j_indices = self.j_indices


    def get_measure_of_distance_to_high_event(self, year_position, the_date, current = True):
        if current:
            the_list = self.current_model_data
        else:
            the_list = self.future_model_data

        measure_list = []
        for m in the_list:
            high_date = m.select_high_date_for_year_and_position(year_position)
            measure_list.append(measure_of_time_to_event(the_date, high_date))
        return np.mean(measure_list)


    #day of year of high flow event averaged over
    #members and time
    def get_mean_dates_of_maximum_annual_flow(self, current = True):
        if current:
            data_list = self.current_model_data
        else:
            data_list = self.future_model_data

        positions = self.get_all_positions()
        result = np.zeros(len(positions))
        count = np.zeros(len(positions))
        positions = self.get_all_positions()
        for pos in positions:
            for m in data_list:
                # @type m ModelDataObject
                for year, d in m.get_dates_of_annual_maxima(pos).iteritems():
                    result[pos] += date_time_to_day_of_year(d)
                    count[pos] += 1
            
        
        result = np.round(np.array(result) / count)
        return result
        pass

    def get_all_positions(self):
        return self.current_model_data[0].get_all_positions()

   

#corresponds to data in one file
class ModelDataObject():
    def __init__(self, member_id = '', low_duration = timedelta(days = 15),
                       high_duration = timedelta(days = 1)):
        self.i_indices = None
        self.j_indices = None
        self.times = None
        self.low_dates = {}
        self.high_dates = {}
        self.member_id = member_id

        self.low_duration = low_duration
        self.high_duration = high_duration

        self._data = None

        self.data_time_step = None

        self.nc_file = None
        self._year_to_dates_cache = {}
        pass

    def init_from_path(self, path = ''):
        self._data, self.times, \
        self.i_indices, self.j_indices = data_select.get_data_from_file(path)

        pass

    def set_data_from_path(self, path = ''):
        self._data = data_select.get_field_from_file(path = path, field_name = 'water_discharge')

    def get_all_positions(self):
        return range(self._data.shape[1])

    ##get dates of occurences of the maxima
    #{year: date}
    def get_dates_of_annual_maxima(self, position_index):
        """

        """
        return data_select.get_period_maxima_dates(self._data[:, position_index], self.times,
                start_month = 1, end_month = 12, event_duration = timedelta(days = 1)
        )

    def select_high_date_for_year_and_position(self, year_position):
        if self.high_dates.has_key(year_position):
            return self.high_dates[year_position]

        year = year_position.get_year()
        pos = year_position.get_position_index()
        
        if self._year_to_dates_cache.has_key(year):
            selected_dates = self._year_to_dates_cache[year]
        else:
            selected_dates = [d for d in ifilter(lambda d: d.year == year, self.times)]
            self._year_to_dates_cache[year] = selected_dates

        selected_values = []
 
        for d, v in ifilter(lambda t: t[0].year == year, zip(self.times, self._data[:, pos])):
            selected_values.append(v)

        assert len(selected_dates) == len(selected_values)

  
        #return None if the data for the year is not complete
        if len(selected_values) < 365:
            return None
        #data time step is assumed to be constant
        #and less than the event duration
        if self.data_time_step is None:
            dt1 = selected_dates[1] - selected_dates[0]
            self.data_time_step = dt1
        else:
            dt1 = self.data_time_step

        #determine the number of points for each event
        #from the relation event_duration = (N-1) * dt
        averaging_length = 2
        while (averaging_length - 1) * dt1 < self.high_duration:
            averaging_length += 1


        the_max_value = None
        the_date_of_max = None
        n_dates = len(selected_dates)
        for i, d in enumerate(selected_dates):
            if i + averaging_length > n_dates:
                break
            else:
                the_value = np.mean(selected_values[i:i + averaging_length])
                if the_max_value is None or the_max_value < the_value:
                    the_max_value = the_value
                    the_date_of_max = d

        self.high_dates[year_position] = the_date_of_max
        return the_date_of_max



if __name__ == "__main__":
    print YearPositionObject(1,2) == YearPositionObject(1,2)
    print YearPositionObject(1,2) == YearPositionObject(1,2)
    print "Hello World"
