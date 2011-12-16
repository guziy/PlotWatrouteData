__author__ = 'huziy'

from datetime import datetime


def get_month_from_name(file_name = "", date_format = "%Y%m"):
    """
    Gets the month from the given file name
    """
    s = file_name.split("_")[-1].split(".")[0]
    return datetime.strptime(s, date_format).month


def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  