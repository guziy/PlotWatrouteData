__author__="huziy"
__date__ ="$Jul 31, 2011 11:47:12 PM$"

from netCDF4 import Dataset
import application_properties
import plot2D.calculate_performance_errors as pe

def check(path = 'data/hydrosheds/directions_for_streamflow9.nc'):
    cells = pe.get_connected_cells(directions_path = path)

if __name__ == "__main__":
    application_properties.set_current_directory()
    check()
    print "Hello World"
