from netCDF4 import Dataset

__author__ = 'huziy'

#This domain is a subset of the AMNO domain, containing 547 grid cells

def get_domain_mask(path = 'data/infocell/amno180x172_basins.nc'):
    """
    returns the matrix of dimensions 180x172, with 1 for the points of interest and 0
    for the others
    """
    ds = Dataset(path)
    result = None
    for v in ds.variables.values():
        if result is None:
            result = v[:]
        else:
            result += v[:]
    ds.close()
    return result

  