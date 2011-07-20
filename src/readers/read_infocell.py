__author__="huziy"
__date__ ="$23 mai 2010 13:59:05$"

from mpl_toolkits.basemap import Basemap
import application_properties
from shape.read_shape_file import get_features_from_shape
from plot2D.calculate_mean_map import *


from util.convert import amno_convert_list
from util.convert import *

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.basemap import NetCDFFile
from math import *
import pylab
from util.geo.ps_and_latlon import *
from math import *
from plot2D.plot_utils import draw_meridians_and_parallels
from matplotlib.patches import Rectangle


from util.geo.lat_lon import get_distance_in_meters
from matplotlib.lines import Line2D

from plot2D.calculate_mean_map import *





import wrf_static_data.read_variable as wrf

inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (sqrt(5) - 1.0) / 2.0       # Aesthetic ratio
fig_width = 1800 * inches_per_pt          # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = [fig_width, fig_height]

params = {
        'axes.labelsize': 14,
        'font.size':18,
        'text.fontsize': 18,
        'legend.fontsize': 14,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'figure.figsize': fig_size
        }

pylab.rcParams.update(params)


#set current directory to the root directory of the project
application_properties.set_current_directory()

from plot2D.map_parameters import polar_stereographic

n_cols = polar_stereographic.n_cols
n_rows = polar_stereographic.n_rows
xs = polar_stereographic.xs
ys = polar_stereographic.ys

lons = polar_stereographic.lons
lats = polar_stereographic.lats

m = polar_stereographic.basemap


#Flow to the right is 2, down is 4, left is 6, and up is 8, the odd numbers are
#the diagonal flows: 1 is up and right, 3 is down and right, 5 is down and left,
#and 7 is up to the left.

def get_indices_of_next_cell(value, i, j):
    i1 = i
    j1 = j
    
    if value not in range(1,9):
        return -1, -1

    if value in [1,2,3]:
        i1 = i + 1
    if value in [5,6,7]:
        i1 = i - 1

    if value in [1,8,7]:
        j1 = j + 1
    if value in [3,4,5]:
        j1 = j - 1

    return i1, j1

### 8   1   2
### 7   9   3
### 6   5   4
def get_indices_of_next_cell_for_trip(value, i, j):
    i1 = i
    j1 = j

    if value not in range(1,9):
        return -1, -1

    if value in [2,3,4]:
        i1 = i + 1
    if value in [6,7,8]:
        i1 = i - 1

    if value in [1,2,8]:
        j1 = j + 1
    if value in [4,5,6]:
        j1 = j - 1

    return i1, j1


def get_indices_of_next_cell_esri(value, i, j):

    if value not in [1,2,4,8,16,32,64,128]:
        return -1, -1

    inext = i
    jnext = j
    if value in [1,2,128]:
        inext += 1
    if value in [8, 16, 32]:
        inext -= 1

    if value in [32, 64, 128]:
        jnext += 1
    if value in [2,4,8]:
        jnext -= 1
    return inext, jnext


class Cell():
    '''
    Class representing a grid cell
    '''
    def __init__(self, id = None):
        self.id = id
        self.next = None
        self.previous = []
        self.area = -1
        self.clay = -1
        self.sand = -1
        self.number_of_upstream_cells = -1  #number of cells that flow into this cell
        self.drainage = -1 #number of cells that flow into this cell
        self.drainage_area = -1 #drainage area in km**2
        self.ibn = 3
        self.chslp = -1
        self.x = -1
        self.y = -1
        self.basin = None
        self.is_end_cell = False
        self.direction_value = -1
        self.basins = []
        self.topo = None
        #rid  - is the indication river or lake, 0 - river
        self.rid = 0
        self.rout = 0
        self.channel_length = -1


    def set_next(self, next_cell):
        if self.next != None and self in self.next.previous:
            self.next.previous.remove(self)

        self.next = next_cell
        if next_cell != None:
            next_cell.add_previous(self)

        #sanity checks
        if self.next != None:
            assert self in self.next.previous
            self.next.basin != None
        assert self not in self.previous



    def coords(self):
        return self.x, self.y


    def get_cells_upflow(self, basin):
        result = []
        for prev in self.previous:
            if prev in basin.cells:
                result.extend(prev.get_cells_upflow(basin))
                result.append(prev)
        return result

    def set_coords(self, i, j):
        self.x = i
        self.y = j

    def add_previous(self, prev):
        if prev not in self.previous:
            self.previous.append(prev)
        assert(len(self.previous) <= 8)
        assert(prev != self)

    def is_connected_to(self, other_cell):
        '''
        returns True if the cell is connected to  the other cell, False otherwize
        '''
        current = self
        i = 0
        path = []
        while current != None:

            if other_cell in path:
                print 'Here'
                return True

            #closed loop
            if current in path:
                return False
            path.append(current)

            
            if current == other_cell:
                return True


            if current.basin != other_cell.basin or current.basin == None:
                return False

            if current.next != None:
                if current.basin != current.next.basin:
                    return False

            current = current.next

            i += 1
            assert i < 10000
        return False


    def calculate_drainage_area(self):
        '''
        calculate drainage area for the cell
        '''
        if self.drainage_area >= 0:
            return

        self.drainage_area = self.area
        for prev in self.previous:
            if prev.basin == None:
                continue
            prev.calculate_drainage_area()
            self.drainage_area += prev.drainage_area
#            self.drainage_area += prev.area

    def calculate_number_of_upstream_cells(self):
        '''
        returns the number of upstream cells which
        flow into the current cell
        '''
        if self.number_of_upstream_cells >= 0:
            return self.number_of_upstream_cells
        else:
            result = 0
            for the_previous in self.previous:
                result += the_previous.calculate_number_of_upstream_cells()
            result += len(self.previous)
        return result
        pass






class Basin():
    '''
    Class representing a basin
    '''
    def __init__(self, id = -1, name = "unknown"):
        self.id = id
        self.name = name
        self.cells = []
        self.exit_cells = []
        self.description = ''

    def add_cell(self, the_cell):
        self.cells.append(the_cell)
        the_cell.basins.append(self)
        the_cell.basin = self
 

    def _drainage_for_cell(self, the_cell):
        if the_cell.drainage >= 0:
            return
        drainage = 0
        for prev in the_cell.previous:
            if prev not in self.cells:
                continue
            self._drainage_for_cell(prev)
            drainage += prev.drainage + 1
        the_cell.drainage = drainage
        

    def _calculate_internal_drainage(self):
        #initialize drainages
        for the_cell in self.cells:
            the_cell.drainage = -1
        for the_cell in self.cells:
            self._drainage_for_cell(the_cell)
        pass


    def _try_to_determine_exits(self):
        self._calculate_internal_drainage()
        drainage = -1
        result_cell = None
        for the_cell in self.cells:
            if drainage < the_cell.drainage and the_cell.next not in self.cells:
                result_cell = the_cell
                drainage = the_cell.drainage
        result_cell.next = None
        self.exit_cells.append(result_cell)

    def set_exit_cells(self, i_list = None, j_list = None):
        '''
        Sets exit cells for the basin using cell indices
        '''
        if i_list == None or j_list == None:
            self._try_to_determine_exits()
        else:
            for i, j in zip(i_list,  j_list):
                print i, j
                assert cells[i][j] in self.cells, "outlet not in mask of the basin %s" % self.name

                self.exit_cells.append(cells[i][j])
                cells[i][j].basin = self

                #if the outflow points to the cell that does not belong to any basin
                if cells[i][j].next != None and cells[i][j].next.basin == None:
                    cells[i][j].set_next( None )

                #if the outflow points to the cell in the same basin
                if cells[i][j].next != None and cells[i][j].next.basin == cells[i][j].basin:
                    cells[i][j].set_next(None)



 

    def get_max_i(self):
        '''
            returns maximum horizontal cell index in the basin
        '''
        i_max = -1
        for the_cell in self.cells:
            if i_max < the_cell.x:
                i_max = the_cell.x
        return i_max


    def get_min_i(self):
        '''
            returns minimum horizontal cell index in the basin
        '''
        i_min = -1
        for the_cell in self.cells:
            if i_min == -1:
                i_min = the_cell.x

            if i_min > the_cell.x:
                i_min = the_cell.x
        return i_min


    def get_max_j(self):
        '''
            returns maximum vertical cell index in the basin
        '''
        j_max = -1
        for the_cell in self.cells:
            if j_max < the_cell.y:
                j_max = the_cell.y
        return j_max

    def get_min_j(self):
        '''
            returns minimum vertical cell index in the basin
        '''
        j_min = -1
        for the_cell in self.cells:
            if j_min == -1:
                j_min = the_cell.y

            if j_min > the_cell.y:
                j_min = the_cell.y
        return j_min


    def get_approxim_middle_indices(self):
        _i_list = []
        _j_list = []
        for cell in self.cells:
            _i_list.append(cell.x)
            _j_list.append(cell.y)
        return np.mean(_i_list), np.mean(_j_list)


###Create data structures
### 2D array of cells and list of basins
#create dummy cell objects
cells = []
for i in range(n_cols):
    cells.append([])
    for j in range(n_rows):
        the_cell = Cell()
        the_cell.set_coords(i, j)
        cells[i].append( the_cell )

basins = []


####

def get_additional_cells(path, start_col = 90, start_row = 143):
    f = open(path)
    i0 = start_col - 1
    j0 = start_row - 1
    i = i0
    j = j0
    for line in f:
        if line.strip() == "":
            continue
        fields = line.split(",")
        for the_field in fields:
            i_next, j_next = get_indices_of_next_cell(int(the_field.strip()), i, j)

            #if the cell was assigned from other source
            if cells[i][j].next != None:
                i = i + 1
                continue
            
            if i_next < 0 or j_next < 0:
                cells[i][j].next = None
                i = i + 1
                continue


            cells[i][j].direction_value = int(the_field.strip())
            cells[i][j].id = 10
            cells[i][j].next = cells[i_next][j_next]
            cells[i_next][j_next].add_previous(cells[i][j])
            
            i = i + 1
        j = j - 1
        i = i0



def read_clay(file = 'data/infocell/amno_180x172_clay.nc'):
    f = NetCDFFile(file)
    data = f.variables['cell_clay']
    for i in range(n_cols):
        for j in range(n_rows):
            cells[i][j].clay = data[i, j]
    pass

def read_sand(file = 'data/infocell/amno_180x172_sand.nc'):
    f = NetCDFFile(file)
    data = f.variables['cell_sand']
    for i in range(n_cols):
        for j in range(n_rows):
            cells[i][j].sand = data[i, j]
    pass

def get_cells_from_infocell(path):
    f = open(path)
    temp_cells = []
    nexts = []


    lines = f.readlines()

    del lines[0] #delete header
    del lines[-1] #delete last line

    f.close()

    for line in lines:
        if line.strip() == "":
            continue
        fields = line.split()
        i = int(fields[1]) - 1
        j = int(fields[2]) - 1
        the_cell = Cell(id = 100)
        the_cell.set_coords(i,j)
        temp_cells.append(the_cell)
        nexts.append( int(fields[6]) - 1 )

    none_next = len(temp_cells)
    print 'none next is: ', none_next


    for cell, next_index in zip(temp_cells, nexts):
        #there is a dummy cell which signals that the current cell does not have next
        if next_index == none_next:
            cell.next = None
            continue
        cell.next = temp_cells[next_index]


    for cell in temp_cells:
        if cell.next != None:
            next_cell = cell.next

            #if the direction was already assigned from another source
            if cells[cell.x][cell.y].next != None:
                continue
            cells[cell.x][cell.y].set_next( cells[next_cell.x][next_cell.y] )
    pass



def get_index_distance(cell1, cell2):
    '''
    get distance between 2 cells in index space
    '''
    return ((cell1.x - cell2.x) ** 2 + (cell1.y - cell2.y) ** 2 ) ** 0.5


def get_distance_along_flow(cell1, cell2):
    '''
    get distance in index space between cell1 and cell2, along the flow
    '''
    x = 0.0
    current = cell1
    while current != cell2:
        x += get_index_distance(current, current.next)
        current = current.next
    return x



def read_elevations(path = 'data/infocell/amno_180x172_topo.nc'):
    '''
    Read elevations for amno grid for amno grid 180x172
    '''
    ncfile = NetCDFFile(path)
    data = ncfile.variables['topography'].data

    for i in range(n_cols):
        for j in range(n_rows):
            if i ==0 and j == 0:
                min_elev = data[i,j]
                max_elev = data[i,j]
            else:
                min_elev = min(min_elev, data[i, j])
                max_elev = max(max_elev, data[i, j])
            cells[i][j].topo = data[i, j]
    
    print 'Elevations from %f (m) to %f (m)' % ( min_elev, max_elev )



def read_cell_area(path = 'data/infocell/amno_180x172_area.nc'):
    ncfile = NetCDFFile(path)
    data = ncfile.variables['cell_area'].data
    for i in range(n_cols):
        for j in range(n_rows):
            cells[i][j].area = data[i, j]
    pass


def calculate_slopes(min_slope = 1.0e-3):
    '''
    Calculates channel slopes taking into account flow directions
    (use after the directions have been corrected)
    '''


    METERS_PER_KM = 1000.0

    number_of_negative_slopes = 0
    number_of_slopes = 0

    for i in range(n_cols):
        for j in range(n_rows):
            slope = min_slope
            current = cells[i][j]
            next = current.next
            current_lon, current_lat = lons[current.x, current.y], lats[current.x, current.y]
            if next != None:
                number_of_slopes += 1
                lon2, lat2 = lons[next.x, next.y], lats[next.x, next.y]
                if current.area <= 0:
                    dist = get_distance_in_meters(current_lon, current_lat, lon2, lat2)
                else:
                    dist = ( current.area ) ** 0.5 * METERS_PER_KM

                slope = ( - next.topo + current.topo ) / dist
                if slope <= 0:
                    slope = min_slope
                    number_of_negative_slopes += 1
            else:
                #check whether there is a neighbor cell, lower than the current if not,
                #assign minimum slope
                dys = []
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0: continue
                        i1 = i + di
                        j1 = j + dj
                        if i1 < 0 or j1 < 0 or i1 == n_cols or j1 == n_rows:
                            continue

                        dy = current.topo - cells[i1][j1].topo
                        if dy > 0:
                            lon2, lat2 = lons[i1, j1], lats[i1, j1]
                            if current.area <= 0:
                                dist = get_distance_in_meters(current_lon, current_lat, lon2, lat2)
                            else:
                                dist = (current.area) ** 0.5 * METERS_PER_KM
                            dys.append(dy / dist)
                if len(dys) > 0:
                    slope = np.min(dys)
            cells[i][j].chslp = slope

    
    print '%d corrected slopes of %d' % (number_of_negative_slopes, number_of_slopes)
    pass





def write_cell(cell, file, format, final_cell):
    '''
    write cell to file
    '''
    next = final_cell if (cell.next == None or cell.next.basin == None) else cell.next


    i, j = cell.x, cell.y
    
    #in order to force the model to calculate channel lengths inside the watroute model
    #cell.channel_length = -1

    if i >= 0 and j >= 0:
        lon = lons[i,j]
        lat = lats[i,j]
    else:
        lon = -1
        lat = -1

    #me
    line = format % (cell.id, cell.x, cell.y, cell.drainage_area, cell.chslp, cell.ibn,
                     next.id, cell.area, cell.rid, cell.rout,
                     lon, lat, cell.channel_length)

    #Vincent
#    line = format % (cell.id, cell.x + 1, cell.y + 1, cell.drainage_area, cell.chslp, cell.ibn,
#                     next.id, cell.area, cell.clay, cell.sand, cell.rid, cell.rout)
                     
    if cell != final_cell:
        file.write(line + '\n')
    else:
        file.write(line)


def put_to_list(cell, the_list):
    '''
    puts cell in the order to the list, first the
    cells upflow then downflow
    '''
    
    if cell in the_list:
        return
    for prev in cell.previous:
#        assert prev.basin != None
        if prev.basin != None:
            put_to_list(prev, the_list)
    the_list.append(cell)
    cell.id = len(the_list)




def write_new_infocell(path = "infocell_QC2.txt"):
    f = open(path, 'w')
    cell_list = []

    #me
    format = '%15s' * 13
    f.write(format % ('N','XX','YY','DA','CHSLP','IBN','NEXT','AREA', 'RID','ROUT','LON', 'LAT', 'CHLEN') + '\n')
    format = '%15d' * 3 + '%15.1f' + '%15.10f' + '%15d' * 2 + '%15.1f' + '%15d' * 2 + '%15.1f' * 3

#    format = '%15s' * 12
#    f.write(format % ('N','XX','YY','DA','CHSLP','IBN','NEXT','AREA', 'CLAY', 'SAND' ,'RID','ROUT') + '\n')
#    format = '%15d' * 3 + '%15.1f' + '%15.10f' + '%15d' * 2 + '%15.1f'*3 + '%15d' * 2

    for basin in basins:
        for cell in basin.cells:
            put_to_list(cell, cell_list)


    final_cell = Cell()
    final_cell.id = len(cell_list) + 1
    final_cell.next = final_cell
    final_cell.area = 0
    final_cell.drainage_area = 0
    final_cell.set_coords(-1, -1)



    for cell in cell_list:
        if cell.next != None:
            if cell.next.basin == None:
                print cell.next.coords()
                

        write_cell(cell, f, format, final_cell)

    write_cell(final_cell, f, format, final_cell)


    print 'cell list: ', len(cell_list)

    for basin in basins:
        for the_cell in basin.cells:
            assert the_cell in cell_list

    assert len(set(cell_list)) == len(cell_list)


    f.close()


#calculate drainage in number of cells, if no cell inflows
#to the crrent cell, then drainaige = 0
def calculate_drainage(the_cell):
    if the_cell.drainage >= 0:
        return

    drainage = 0
    for prev in the_cell.previous:
        calculate_drainage(prev)
        drainage += prev.drainage
        
    drainage += len(the_cell.previous)
    the_cell.drainage = drainage





def calculate_drainage_areas():
    '''
    calculates drainage areas in km**2
    '''
    for basin in basins:
        for cell in basin.cells:
            cell.calculate_drainage_area()


def calculate_drainage_for_all(cells):
    '''
    Calculate drainage, number of cells that 
    inflow into the current cell
    '''
    check_for_loops(cells)
    for i in range(n_cols):
        for j in range(n_rows):
            calculate_drainage(cells[i][j])

#interpolates slope data from the wrf grid
def get_slopes_from_wrf_data(min_slope = 1.0e-3):
    data = wrf.get_slopes_interpolated_to_amno_grid(polar_stereographic)
    for i in range(n_cols):
        for j in range(n_rows):
            slp = data[i, j]
            if slp <= 0:
                slp = min_slope
            cells[i][j].chslp = slp
    pass

def get_flow_directions(cells):
    u = np.zeros((n_cols, n_rows))
    v = np.zeros((n_cols, n_rows))

    u[:,:] = None
    v[:,:] = None
    for i in range(n_cols):
        for j in range(n_rows):
            if cells[i][j].next != None:
                next_cell = cells[i][j].next
                u[i, j] = float(next_cell.x - i)
                v[i, j] = float(next_cell.y - j)
    return u, v


def plot_directions(cells):
    '''
        cells - 2D array of cells
        basins_mask - 1 where basins, 0 elsewhere
    '''
    u_plot = np.zeros((n_cols, n_rows))
    v_plot = np.zeros((n_cols, n_rows))

    u_plot[:, :] = None
    v_plot[:, :] = None

    u, v = get_flow_directions(cells)

    for i in range(n_cols):
        for j in range(n_rows):
            if cells[i][j].next != None and len(cells[i][j].basins) > 0:
                u_plot[i, j] = u[i, j]
                v_plot[i, j] = v[i, j]


    m.quiver(xs, ys, u_plot, v_plot, scale = 6.5, width = 0.02 , units='inches')
    
    m.drawcoastlines(linewidth=0.5)
    draw_meridians_and_parallels(m, 20)
    plt.savefig("flows_and_masks.png", bbox_inches='tight')
    



def check_cell_for_loop(cell):
    current = cell
    path = [cell]
    while current.next != None:
        current = current.next
        if current in path:
            print 'closed path:'
            for the_cell in path:
                print the_cell.coords(), the_cell.direction_value
            print current.coords()
            raise Exception('Closed loop for %d, %d' % cell.coords())
        path.append(current)


def check_for_loops():
    for basin in basins:
        for cell in basin.cells:
            check_cell_for_loop(cell)

def get_cell_lats_lons(path):
    index_pairs = []
    f = open(path)
    #skip header
    f.readline()
    for line in f:
        fields = line.split()
        new_pair = [int( fields[1] ) , int( fields[2] ) ]

        if new_pair[0] == 1 and new_pair[1] == 1:
            continue
        index_pairs.append(new_pair)
    return amno_convert_list(index_pairs)



def plot_basins_separately(path, cells):
    ncfile = NetCDFFile(path)
    bas_names = ncfile.variables.keys()
    vars = ncfile.variables

    to_plot = np.zeros((n_cols, n_rows))

    u, v = get_flow_directions(cells)

    u_plot = np.zeros((n_cols, n_rows))
    v_plot = np.zeros((n_cols, n_rows))


    descr_map = get_basin_descriptions('data/infocell/basin_info.txt')
    for name in bas_names:
        to_plot[:,:] = None
        u_plot[:,:] = None
        v_plot[:,:] = None

        the_mask = np.transpose(vars[name].data)
        for i in range(n_cols):
            for j in range(n_rows):
                if the_mask[i, j] == 1:
                    u_plot[i,j] = u[i, j]
                    v_plot[i,j] = v[i, j]
                    to_plot[i, j] = the_mask[i, j]
        plt.cla()
        m.drawcoastlines(linewidth = 0.5)
        m.scatter(xs, ys, c=to_plot, marker='s', s=100, linewidth = 0, alpha = 0.2)
        m.quiver(xs, ys, u_plot, v_plot, scale = 5, width = 0.025 , units='inches')


        override = {'fontsize': 14,
                  'verticalalignment': 'baseline',
                  'horizontalalignment': 'center'}

        plt.title(name +':' + descr_map[name], override)
        ymin, ymax = plt.ylim()
        plt.ylim(ymin + 0.12 * (ymax - ymin), ymax * 0.32)


        xmin, xmax = plt.xlim()
        plt.xlim(xmin + (xmax - xmin) * 0.65, 0.95*xmax)

        plt.savefig('basins_images/' + name + '.png')
        print name



def plot_basins():
    '''
        Plot amno basins as scatter plot 
    '''

    bas_names = []
    for basin in basins:
        print basin.name, len(basin.cells)
        assert basin.id != None
        bas_names.append(basin.name)

    
    to_plot = np.zeros((n_cols, n_rows))
    to_plot[:,:] = None

    for basin in basins:
        i, j = basin.get_approxim_middle_indices()
        
#        text = '{0}'.format(basin.name)
#        plt.annotate(text, xy = (xs[i, j], ys[i, j]), size = 20,
#                        ha = 'center', va = 'center', bbox = dict(facecolor = 'white', pad = 12))

        for cell in basin.cells:
            i, j = cell.x, cell.y
            to_plot[ i, j] = basin.id



    color_map = mpl.cm.get_cmap('jet', len(bas_names))
    m.scatter(xs, ys, c=to_plot, marker='s', s=200,
              cmap = color_map, linewidth = 0, alpha = 0.4)

    m.drawcoastlines(linewidth = 1)
    m.drawstates(linewidth = 0.5)
    m.drawcountries(linewidth = 0.5)
    m.drawrivers()
 #   plot_basin_legend(bas_names, color_map)
    
    ymin, ymax = plt.ylim()
    plt.ylim(ymin + 0.14*(ymax - ymin), ymax * 0.34)

    xmin, xmax = plt.xlim()
    plt.xlim(xmin + (xmax - xmin) * 0.65, 0.84*xmax)

    
 
#    plot_basin_boundaries()
    plot_basin_boundaries_from_shape(m, linewidth = 1)


    plt.savefig("amno_quebec_basins.png", bbox_inches='tight')




def plot_basin_legend(basin_names, color_map, scale = 0.5):
    rectangles = []
    for k, name in enumerate(basin_names):
        r = Rectangle((0,0),1,1, facecolor = color_map(k), alpha = 0.4)
        rectangles.append(r)
    plt.legend(rectangles, basin_names)



def get_basin_descriptions(path = 'data/infocell/basin_info.txt'):
    f = open(path)
    descr_map = {}
    for line in f:
        if ':' not in line:
            continue
        if 'p1mk_' not in line:
            continue
        fields = line.split(':')
        key = fields[0].replace('p1mk_','').strip()
        descr_map[key] = fields[1]
    return descr_map



def paint_all_points_with_directions(cells):
    to_plot = np.zeros((n_cols, n_rows))
    to_plot[:,:] = None

    ncfile = NetCDFFile('data/infocell/quebec_masks_amno180x172.nc')
    vars = ncfile.variables

    the_data = np.transpose( vars['RDO'].data )


    for i in range(n_cols):
        for j in range(n_rows):
            if cells[i][j].next != None : to_plot[i,j] = 1
            if cells[i][j].direction_value == 0: to_plot[i,j] = 3
            if the_data[i, j] == 1 and cells[i][j].direction_value == 0: to_plot[i, j] = 2
    color_map = mpl.cm.get_cmap('jet', 3)
    m.scatter(xs, ys, c = to_plot, cmap = color_map , marker='s', s=150,
              linewidth = 0, alpha = 0.4)

    m.drawcoastlines()
    ymin, ymax = plt.ylim()
    plt.ylim(ymin + 0.1*(ymax - ymin), ymax * 0.35)

    xmin, xmax = plt.xlim()
    plt.xlim(xmin + (xmax - xmin) * 0.5, 0.95*xmax)



IMAX = 'imax'; JMAX = 'jmax'; IMIN = 'imin'; JMIN = 'jmin'
def get_indices_from_line(line):
    '''
    Parses line of type <basin name>: i1, j1; i2, j2 ...
    returns basin and 2 lists: of is and js
    '''
    i_list = []
    j_list = []

    line = line.replace(' ', '').strip() #delete spaces

    fields = line.split(':')
    basin = get_basin_for_name(fields[0].strip())

    if basin == None:
        print 'basin %s is not known ...' % fields[0].strip()
        return basin, i_list, j_list

    fields = fields[1].split(';')
    for field in fields:
        if ',' not in field:
            continue

        if IMAX in field:
            i = basin.get_max_i()
        else:
            i = basin.get_min_i()

        if JMAX in field:
            j = basin.get_max_j()
        else:
            j = basin.get_min_j()


        field = field.replace(IMAX, '')
        field = field.replace(IMIN, '')
        field = field.replace(JMAX, '')
        field = field.replace(JMIN, '')

        values = field.split(',')
        print values
        i += int(values[0]) if values[0] != '' else 0
        j += int(values[1]) if values[1] != '' else 0
        i_list.append(i)
        j_list.append(j)

    return basin, i_list, j_list



def read_outlets(path='data/infocell/outlets.txt'):
    '''
    read outlets for the basins, used in read_basins()
    '''
    f = open(path)
    for line in f:
        if ':' not in line:
            continue
        basin, i_list, j_list = get_indices_from_line(line)
        if basin != None:
            basin.set_exit_cells(i_list, j_list)
    #assign next cells for outlets
    read_next_for_outlets()
    pass

def read_next_for_outlets(path = 'data/infocell/next_for_outlets.txt'):
    '''
    read next cells for outlets (used inside read_outlets)
    '''

    for basin in basins:
        for outlet in basin.exit_cells:
            outlet.set_next(None)

    f = open(path)
    for line in f:
        if ':' not in line:
            continue
        basin, i_list, j_list = get_indices_from_line(line)
        if basin != None:
            for i, j, outlet in zip(i_list, j_list, basin.exit_cells):
                outlet.set_next(cells[i][j])
    pass


def read_basins(path = 'data/infocell/amno180x172_basins.nc'):
    '''
    reads data from netcdf files and fills in the basins array
    '''

    descr_map = get_basin_descriptions()
    ncfile = NetCDFFile(path)
    id = 1



    for name, values in ncfile.variables.iteritems():
        if n_cols == values.shape[1]:#transpose if necessary
            the_values = np.transpose(values)
        else:
            the_values = values
            
        the_basin = Basin(id = id, name = name)
        if descr_map.has_key(name):
            the_basin.description = descr_map[name]

        basins.append(the_basin)
        for i in range(n_cols):
            for j in range(n_rows):
                if the_values[i,j] == 1:
                    the_basin.add_cell(cells[i][j])
 
        id += 1

    
    pass


def get_basin_for_name(name):
    '''
    returns basin object for the specified name
    '''
    for basin in basins:
        if basin.name == name:
            return basin





def check_basin_intersections():
    for i in range(len(basins)):
        for j in range(i+1, len(basins)):
            basin1 = basins[i]
            basin2 = basins[j]

            common = []
            count = 0
            for cell1 in basin1.cells:
                if cell1 in basin2.cells:
                    common.append(cell1)
                    count += 1
            if count > 0:
                print 'basins %s and %s have %d common cells' % (basin1.name, basin2.name, count)
                for the_cell in common:
                    print the_cell.coords()

def delete_basins(names):
    '''
    Deletes the basins with names from the list of basins
    '''
    to_delete = []
    for name in names:
        for basin in basins:
            if basin.name == name:
                to_delete.append(basin)
    for x in to_delete:
        for the_cell in x.cells:
            the_cell.basins.remove(x)
            if len(the_cell.basins) > 0:
                the_cell.basin = the_cell.basins[0]
            else:
                the_cell.basin = None
                for prev in the_cell.previous:
                    prev.set_next(None)
                    
        basins.remove(x)



def get_neighbors_of(cell):
    '''
        returns the list of neighbor cells of the current one,
        the neighbors belong to the same basin as a cell
    '''
    i0, j0 = cell.coords()
    result = []
    for i in range(-1,2):
        for j in range(-1,2):
            if i == 0 and j == 0: continue #skip the cell
            #skip cells that belong to different basin
            if cells[i0 + i][j0 + j] not in cell.basin.cells: continue
            result.append( cells[i0 + i][j0 + j] )
    return result




def distance(cell, cell_list):
    '''
    distance between the cell and the cell_list
    '''
    for i, the_cell in enumerate(cell_list):
        d = get_index_distance(cell, the_cell)
        if i == 0:
            result = d
        if result > d: result = d
    return result

def get_closest_correct(wrong, correct_list):
    '''
    returns cell closest to wrong from correct_list
    '''

    for i, the_cell in enumerate(correct_list):
        d = get_index_distance(wrong, the_cell)
        if i == 0:
            result = d
            correct_cell = the_cell
        if result > d:
            result = d
            correct_cell = the_cell
    return correct_cell



def is_infinite_loop(cell):
    current = cell
    path = []
    while current.next != None:
        if current not in path:
            path.append(current)
        else:
            return True

        current = current.next
    return False



def correct_cell(cell):
     neighbors = get_neighbors_of(cell)
     assert len(neighbors) > 0
     for neighbor in neighbors:
        if not is_infinite_loop(neighbor):
            cell.set_next(neighbor)
            return

     for neighbor in neighbors:
         if is_infinite_loop(neighbor):
             correct_cell(neighbor)

     cell.set_next(neighbors[0])

def correct_loops_only():
    
    for basin in basins:
        wrong = []
        for the_cell in basin.cells:
            path = []
            current = the_cell
            while current != None:
                if current in path:
                    wrong.append(current)
                    break
                else:
                    path.append(current)
                current = current.next

        print basin.name
        print 'Number of wrong cells %d from %d' % (len(wrong), len(basin.cells) )

    for w in wrong:
       correct_cell(w)


def correct_directions():
    '''
        corrects the directions of the cells of the basin to point to the correct outlet
    '''
    for basin in basins:
        correct = []
        wrong = []
        outlets = basin.exit_cells


        print basin.name, basin.exit_cells[0].next

        #correct neighbors of the outlets
#        for outlet in outlets:
#            assert outlet.next == None or outlet.next.basin != outlet.basin
#            neighbors = get_neighbors_of(outlet)
#            for neighbor in neighbors:
#                if neighbor.next != outlet:
#                    neighbor.set_next(outlet)
#            correct.extend(neighbors)
#            correct.append(outlet)



        for the_cell in basin.cells:
            assert len(outlets) > 0
            for outlet in outlets:
                assert outlet in basin.cells, 'Outlets should be assigned to the basin: %s' % basin.name
                if the_cell.is_connected_to(outlet) and the_cell not in correct:
                    correct.append(the_cell)

            if the_cell not in correct:
                wrong.append(the_cell)


        print basin.name
        assert len(correct) > 0
        print len(correct), len(wrong), len(outlets), len(basin.cells)
        assert len(correct) + len(wrong) == len(basin.cells)

        #sort that we begin from the cells that are closest to correct ones
        while len(wrong) > 0:
            wrong.sort(key = lambda x: distance(x, correct))
            w = wrong.pop(0)

            c = get_closest_correct(w, correct)
            neighbors = get_neighbors_of(w)
            if c in neighbors:
                if w.next != None:
                    print 'next cell was (%d, %d) before correcting' % w.next.coords()
                w.set_next(c)

                correct.append(w)

                upflow_cells = w.get_cells_upflow(basin)

 

                for up_cell in upflow_cells:
                    assert up_cell not in correct
                    wrong.remove(up_cell)

                correct.extend(upflow_cells)
                print 'corrected %d, %d ' % w.coords()
                print 'new next cell is (%d, %d)' % w.next.coords()
            else:
                print c == w
                print c.coords()
                print w.basin.name
                print w.coords()
                raise Exception('Something wrong with the algorithm')

    #delete previous cells that do not belong to any basin
    for basin in basins:
        for cell in basin.cells:
            to_del = []
            for prev in cell.previous:
                if prev.basin == None:
                    to_del.append(prev)


            for x in to_del:
                x.set_next(None)


def get_cells_without_basin():
    missing = []
    for i in range(n_cols):
        for j in range(n_rows):
            if cells[i][j].basin != None: continue
            neighbors = get_all_neighbors(cells[i][j])
            is_missing = 0
            for neighbor in neighbors:
                if neighbor.basin == None:
                    is_missing += 1
            if is_missing < 2:
                missing.append(cells[i][j])
    return missing

import trip.read_ddm as trip
def get_ddm_from_trip():
    data = trip.get_directions_for_amno45grid(lons, lats)
    for i in range(n_cols):
        for j in range(n_rows):
            the_cell = cells[i][j]
            if the_cell.basin == None:
                continue

            inext, jnext = get_indices_of_next_cell_for_trip(data[i, j], i, j)

            if inext < n_cols and inext >= 0 and jnext < n_rows and jnext >= 0:
                if the_cell not in the_cell.basin.exit_cells:
                    the_cell.set_next(cells[inext][jnext])
            else:
                cells[i][j].set_next(None)



##
##
def get_basins_with_cells_connected_using_hydrosheds_data():
    path = 'data/hydrosheds/directions.nc'
    ncFile = NetCDFFile(path)
    read_cell_area()


    inext_var = ncFile.variables['flow_direction_index0'].data
    jnext_var = ncFile.variables['flow_direction_index1'].data


    slopes = ncFile.variables['slope'].data
    channel_length = ncFile.variables['channel_length'].data

    min_slope = 1.0e-4

    read_basins()
    for basin in basins:
        for the_cell in basin.cells:
            i = the_cell.x
            j = the_cell.y

            inext = inext_var[i, j]
            jnext = jnext_var[i, j]

            print i,j, '->', inext, jnext

            the_cell.chslp = slopes[i, j] if slopes[i, j] > 1.0e-10 else min_slope

            if inext >= 0:
                the_values = (lons[i, j], lats[i,j], lons[inext, jnext], lats[inext, jnext])
                print "(%f, %f) -> (%f, %f)" % the_values
                the_cell.set_next(cells[inext][jnext])
            else:
                the_cell.set_next(None)


            if channel_length[i,j] > 0:
                the_cell.channel_length = channel_length[i, j]
            else:
                the_cell.channel_length = (the_cell.area) ** 0.5 * 1000.0
            


    calculate_drainage_areas()
    check_for_loops()

    read_clay()
    read_sand()
    return basins




def plot_directions_from_file(path = 'data/hydrosheds/directions_qc_dx0.1.nc'):
    ncFile = NetCDFFile(path)
    read_cell_area()


    inext_var = ncFile.variables['flow_direction_index0'].data
    jnext_var = ncFile.variables['flow_direction_index1'].data

    lons = ncFile.variables['longitude'].data
    lats = ncFile.variables['latitude'].data

    lons = np.array(lons)

    lons[lons >= 180] -= 360

    print np.min(lons), np.max(lons)

    basemap = Basemap(resolution = 'i')

    
    
    lons, lats = basemap(lons, lats)




    nx, ny = lons.shape
    u = np.ma.masked_all(lons.shape)
    v = np.ma.masked_all(lons.shape)

    local_cells = []
    for i in xrange(nx):
        local_cells.append([])
        for j in xrange(ny):
            if inext_var[i, j] >= 0:
                i1 = inext_var[i, j]
                j1 = jnext_var[i, j]
                u[i,j] = lons[i1,j1] - lons[i,j]
                v[i,j] = lats[i1,j1] - lats[i,j]

    print np.ma.min(u), np.ma.max(u)
    basemap.quiver(lons, lats, u, v, scale = 1, width = 0.006 , units='inches')
    basemap.drawcoastlines()

    ymin, ymax = plt.ylim()
    xmin, xmax = plt.xlim()
    
    plt.xlim(xmin * 0.5, xmin * 0.25)
    plt.ylim(ymax * 0.4, ymax * 0.9)
    plt.show()
    pass



def read_derived_from_hydrosheds():

   # get_cells_from_infocell('data/infocell/HQ2_infocell.txt')

    path = 'data/hydrosheds/directions.nc'
    ncFile = NetCDFFile(path)
    read_cell_area()


    inext_var = ncFile.variables['flow_direction_index0'].data
    jnext_var = ncFile.variables['flow_direction_index1'].data
    

    slopes = ncFile.variables['slope'].data
    channel_length = ncFile.variables['channel_length'].data
    

    min_slope = 1.0e-4

    read_basins()

    for basin in basins:
        for the_cell in basin.cells:
            i = the_cell.x
            j = the_cell.y

            inext = inext_var[i, j]
            jnext = jnext_var[i, j]
            
            the_cell.chslp = slopes[i, j] if slopes[i, j] > 1.0e-10 else min_slope


            
            if inext >= 0:
                print inext, jnext
                print "(%f, %f) -> (%f, %f)" % (lons[i, j], lats[i,j], lons[inext, jnext], lats[inext, jnext])
                the_cell.set_next(cells[inext][jnext])
            else:
                the_cell.set_next(None)

            the_cell.channel_length = channel_length[i, j] if channel_length[i, j] > 0 else (the_cell.area) ** 0.5 * 1000.0



    calculate_drainage_areas()
    check_for_loops()
    plot_basins()
    plot_directions(cells)
    
    read_clay()
    read_sand()

    write_new_infocell()

   
  


    pass


def read_flowdirections_correct_and_save():
#    get_additional_cells('data/infocell/AMNO_final_direction.csv')
    get_additional_cells('data/infocell/direction_ver2.csv')
    read_basins('data/infocell/amno180x172_basins.nc')
    read_outlets()


    plot_basins()

    slopes_from_wrf = False
    directions_from_trip = False

    if slopes_from_wrf:
        get_slopes_from_wrf_data()
    else:
        read_elevations()
        calculate_slopes()


    if directions_from_trip:
        get_ddm_from_trip()


   # correct_loops_only()
    print '============='
   # correct_loops_only()

    correct_directions()

    plot_directions(cells)
    check_for_loops() #sanity check


    read_cell_area()
    calculate_drainage_areas()


   
    check_basin_intersections()

    write_new_infocell()




def get_all_neighbors(cell):
    '''
        returns the list of neighbor cells of the current one
    '''
    i0, j0 = cell.coords()
    result = []
    for i in range(-1,2):
        for j in range(-1,2):
            if i == 0 and j == 0: continue #skip the cell
            i1 = i0 + i
            j1 = j0 + j
            if i1 not in range(n_cols) or j1 not in range(n_rows):
                continue
            result.append( cells[i1][j1] )
    return result



def plot_boundaries(cell):
    '''
    Plot boundaries of the cell
    '''
    neighbors = get_all_neighbors(cell)
    lines = []
    for neighbor in neighbors:
        if neighbor.basin != cell.basin:
            i = cell.x
            j = cell.y
            
            if cell.x == neighbor.x:
                x2 = 0.5 * (xs[i,j] + xs[i + 1, j])
                x1 = 0.5 * (xs[i - 1,j] + xs[i, j])
                y1 = 0.5 * (ys[i,j] + ys[neighbor.x, neighbor.y])
                y2 = y1

            elif cell.y == neighbor.y:
                y2 = 0.5 * (ys[i,j] + ys[i, j + 1])
                y1 = 0.5 * (ys[i,j] + ys[i, j - 1])
                x1 = 0.5 * (xs[i,j] + xs[neighbor.x, neighbor.y])
                x2 = x1

            else:
                continue

            lines.append(Line2D([x1, x2],[y1, y2], linewidth = 2, color = 'black'))
    return lines

def plot_basin_boundaries():
    plot_basin_boundaries_from_shape(m)
#    lines = []
#    for basin in basins:
#        for cell in basin.cells:
#            lines.extend( plot_boundaries(cell) )
#    ax = plt.gca()
#    print len(lines)
#    for line in lines:
#        ax.add_line(line)


from shape.read_shape_file import *
def plot_basin_boundaries_from_shape(basemap, linewidth = 2, edgecolor = 'k'):
    ax = plt.gca()
    for poly in get_features_from_shape(basemap, linewidth = linewidth, edgecolor = edgecolor):
        ax.add_patch(poly)
    pass


def test():


    print cells[5][5].is_connected_to(cells[5][5])
#    write_new_infocell()
#    read_basins()
    pass

    

def main():
    #read data
    get_cells_from_infocell('data/infocell/HQ2_infocell.txt')
    read_flowdirections_correct_and_save()

    folder = 'data/streamflows/VplusF_newmask1'
    pass

import time
if __name__ == "__main__":
    t0 = time.clock()
#    test()
#    main()
#    read_derived_from_hydrosheds()
    plot_directions_from_file()
    print 'Execution time %f seconds' % (time.clock() - t0)