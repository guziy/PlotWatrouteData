__author__="huziy"
__date__ ="$7 juil. 2010 12:40:06$"

import Nio as nio
import os
import numpy as np

import application_properties
application_properties.set_current_directory()


X_DIM = 'x'
Y_DIM = 'y'
def test():

    file_path = 'test1.nc'
    if os.path.isfile(file_path):
        os.remove(file_path)

    file = nio.open_file(file_path, 'c')
    nx = 10
    ny = 10
    file.create_dimension('x', nx)
    file.create_dimension('y', ny)

    the_var = file.create_variable('int_var', 'i', (X_DIM, Y_DIM))
    data = np.zeros((nx, ny))
    the_var[:] = data.astype('i')

    file.close()



if __name__ == "__main__":
    test()
