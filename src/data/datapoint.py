__author__="huziy"
__date__ ="$27 mai 2010 09:46:22$"

from datetime import *
import itertools

class DataPoint():
    '''
    Represents a point with model data
    '''
    
    def __init__(self, dates = None, values = None):
        if dates == None or values == None:
            self.timeseries = {}
        else:
            self.timeseries = dict(zip(dates, values))
        self.sorted_dates = None

    def remove_record_for_date(self, the_date):
        if self.timeseries.has_key(the_date):
            del self.timeseries[the_date]

    def delete_data_for_year(self, year):
        to_remove = []
        for d in self.sorted_dates:
            if d.year == year:
                to_remove.append(d)
            if d.year > year:
                break

        for d in to_remove:
            self.sorted_dates.remove(d)
            if self.timeseries.has_key(d):
                del self.timeseries[d]

    def delete_data_before_year(self, year):
        if self.sorted_dates[0].year >= year:
            return

        start_year = self.sorted_dates[0].year
        for the_year in xrange(start_year, year):
            self.delete_data_for_year(the_year)


    def delete_data_after_year(self, year):
        if self.sorted_dates[-1].year <= year:
            return

        end_year = self.sorted_dates[-1].year
        for the_year in xrange(year + 1, end_year + 1):
            self.delete_data_for_year(the_year)




    def get_sorted_dates(self):
        if self.sorted_dates == None:
            self.sorted_dates = sorted(self.timeseries.keys())
        return self.sorted_dates


    def add_timeseries_data(self, date, value):
        self.timeseries[date] = value

    def get_values_sorted_by_date(self):
        '''
        returns values sorted corresponding to dates
        '''
        result = []
        if len(self.timeseries) == 0:
            return result
        dates = self.timeseries.keys()
        dates.sort()
        for d in dates:
            result.append(self.timeseries[d])
        return result


 
    def get_monthly_dates_sorted(self):
        result = []
        dates = self.get_sorted_dates()
        month_date = None
        for date in dates:
            if month_date == None or month_date.month != date.month:
               month_date = datetime(year = date.year, month = date.month, day = 1)
               result.append(month_date)
        return result

    def get_monthly_means_sorted_by_date(self):
        '''
        monthly mean values sorted corresponding to date
        '''
        result = []
        dates = self.get_dates_sorted()
        month_date = None
        for date in dates:
            if month_date == None or month_date.month != date.month:
               month_date = datetime(year = date.year, month = date.month, day = 1)
               value = 0.0
               number = 0.0
               result.append(value)
            number += 1.0
            value += self.timeseries[date]
            result[len(result) - 1] = value / number

        assert len(result) == len(self.get_monthly_dates_sorted())
        return result

    def get_values_for_dates(self, dates):
        result = []
        for date in dates:
            the_value = None
            if date in self.timeseries.keys():
                the_value = self.timeseries[date]
            result.append(the_value)
        return result

    def get_timeseries_length(self):
        return len(self.timeseries)


if __name__ == "__main__":
    print "Hello World"
