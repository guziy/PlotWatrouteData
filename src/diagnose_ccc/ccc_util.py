import os

__author__ = 'huziy'

from datetime import datetime


def get_month_from_name(file_name = "", date_format = "%Y%m"):
    """
    Gets the month from the given file name
    """
    return get_month_date_from_name(file_name=file_name, date_format = date_format).month

def get_month_date_from_name(file_name = "", date_format = "%Y%m"):
    """
    gets the first day of the month at 00:00:00
    :rtype : datetime.datetime
    """
    s = file_name.split("_")[-1].split(".")[0]
    return datetime.strptime(s, date_format)



def get_yearmonth_to_path_map(data_folder):
    """
    returns a map {(year, month) -> path to data file}
    :rtype: dict
    """
    result = {}
    for the_file in os.listdir(data_folder):
        month_date = get_month_date_from_name(the_file)
        year, month = month_date.year, month_date.month
        result[(year, month)] = os.path.join(data_folder, the_file)

    return result



def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  