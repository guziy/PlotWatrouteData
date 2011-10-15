import ds
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
#from mpl_toolkits.basemap import NetCDFFile
from netCDF4 import Dataset
from math import *
import pylab
from util.geo.ps_and_latlon import *
from math import *
from util.plot_utils import draw_meridians_and_parallels
from matplotlib.patches import Rectangle


from util.geo.lat_lon import get_distance_in_meters
from matplotlib.lines import Line2D

from plot2D.calculate_mean_map import *

from data.cell import Cell
from data.basin import Basin

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


from matplotlib.patches import RegularPolygon

import util.plot_utils as plot_utils
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
    f = Dataset(file)
    data = f.variables['cell_clay'][:]
    for i in range(n_cols):
        for j in range(n_rows):
            cells[i][j].clay = data[i, j]
    f.close()
    pass

def read_sand(file = 'data/infocell/amno_180x172_sand.nc'):
    f = Dataset(file)
    data = f.variables['cell_sand'][:]
    for i in range(n_cols):
        for j in range(n_rows):
            cells[i][j].sand = data[i, j]
    f.close()
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
    ncfile = Dataset(path)
    data = ncfile.variables['topography'][:]

    for i in range(n_cols):
        for j in range(n_rows):
            if i ==0 and j == 0:
                min_elev = data[i,j]
                max_elev = data[i,j]
            else:
                min_elev = min(min_elev, data[i, j])
                max_elev = max(max_elev, data[i, j])
            cells[i][j].topo = data[i, j]
    ncfile.close()
    print 'Elevations from %f (m) to %f (m)' % ( min_elev, max_elev )



def read_cell_area(path = 'data/infocell/amno_180x172_area.nc'):
    ncfile = Dataset(path)
    data = ncfile.variables['cell_area'][:]
    for i in range(n_cols):
        for j in range(n_rows):
            cells[i][j].area = data[i, j]
    ncfile.close()
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
        if cell.next is not None:
            if cell.next.basin is None:
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
    """
    calculates drainage areas in km**2
    """
    for basin in basins:
        for cell in basin.cells:
            cell.calculate_drainage_area()


def calculate_drainage_for_all(cells = None):
    """
    Calculate drainage, number of cells that
    inflow into the current cell
    """
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

def get_flow_directions(cells, basins = None):
    """
    Document me!
    """
    u = np.ma.masked_all((n_cols, n_rows))
    v = np.ma.masked_all((n_cols, n_rows))

    if basins is None:
        for i in range(n_cols):
            for j in range(n_rows):
                next_cell = cells[i][j].next
                if next_cell is not None:
                    if next_cell is not None:
                        iNext, jNext = next_cell.coords()
                        # @type cell Cell
                        i, j = cells[i][j].coords()
                        dx = float(xs[iNext,jNext] - xs[i,j])
                        dy = float(ys[iNext,jNext] - ys[i,j])
                        dr = np.sqrt(dx ** 2 + dy ** 2)
                        u[i, j] =  dx
                        v[i, j] = dy
    else:
        ##plot directions only for the cells inside basins
        for basin in basins:
            for cell in basin.cells:
                next_cell = cell.next
                if next_cell is not None:
                    iNext, jNext = next_cell.coords()
                    # @type cell Cell
                    i, j = cell.coords()
                    dx = float(xs[iNext,jNext] - xs[i,j])
                    dy = float(ys[iNext,jNext] - ys[i,j])
                    dr = np.sqrt(dx ** 2 + dy ** 2)
                    u[i, j] =  dx
                    v[i, j] = dy


    return u, v


def plot_directions(cells, basemap = None, domain_mask = None):
    """
        cells - 2D array of cells
        basins_mask - 1 where basins, 0 elsewhere
    """
    if basemap is None:
        the_basemap = polar_stereographic.basemap
    else:
        the_basemap = basemap


    u, v = get_flow_directions(cells)
    if domain_mask is not None:
        u = np.ma.masked_where(~(domain_mask == 1), u)
        v = np.ma.masked_where(~(domain_mask == 1), v)
    x = np.ma.mean( np.ma.sqrt(np.ma.power(u, 2) + np.ma.power(v, 2)))
    print "arrow width = ", 0.1 * x
    the_basemap.quiver(xs, ys, u, v, pivot = 'middle', scale = 1.2, width =  0.1 * x, units='xy')
    
 #   m.drawcoastlines(linewidth=0.5)
 #   draw_meridians_and_parallels(m, 20)
 #   plt.savefig("flows_and_masks.pdf", bbox_inches='tight')
    



def check_cell_for_loop(cell):
    current = cell
    path = [cell]
    while current.next is not None:
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
    ncfile = Dataset(path)
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

        the_mask = np.transpose(vars[name][:])
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
    ncfile.close()



def plot_basins(sign_basins = True, save_to_file = False, 
                draw_rivers = True, basemap = None):
    """
        Plot amno basins as scatter plot,
        if sign_basins == True, then signs the basin names on top

    """

    bas_names = []
    for basin in basins:
        print basin.name, len(basin.cells)
        assert basin.id is not None
        bas_names.append(basin.name)

    
    #to_plot = np.zeros((n_cols, n_rows))
    to_plot = np.ma.masked_all((n_cols, n_rows))

    for basin in basins:
        if sign_basins:
            i, j = basin.get_approxim_middle_indices()
            text = '{0}'.format(basin.name)
            plt.annotate(text, xy = (xs[i, j], ys[i, j]), size = 20,
                            ha = 'center', va = 'center', bbox = dict(facecolor = 'white', pad = 8))

        for cell in basin.cells:
            i, j = cell.x, cell.y
            to_plot[i, j] = basin.id



    color_map = mpl.cm.get_cmap('jet', len(bas_names))



    if basemap is None:
        b = m
    else:
        b = basemap

    
    x, y = b(lons, lats)


    x_left = x[0,:] - (x[1,:] - x[0,:])
    y_bottom = y[:,0] - (y[:,1] - y[:,0])

    x1 = np.zeros(x.shape)
    y1 = np.zeros(y.shape)

    x1[1:,:] = x[:-1,:]
    x1[0,:] = x_left
    x1 = 0.5 * (x1 + x)

    y1[:,1:] = y[:,:-1]
    y1[:, 0] = y_bottom
    y1 = 0.5 * (y + y1)
 


 #   b.scatter(x, y, c = to_plot, marker = 's', s = 200)
    b.pcolormesh(x1, y1, to_plot, #marker='s', s=200,
                  cmap = color_map, alpha = 0.4)
                  
#    b.pcolormesh(x1, y1, to_plot, #marker='s', s=200,
#                  cmap = color_map, alpha = 0.4)

    b.drawcoastlines(linewidth = 0.5)
    if draw_rivers:
        b.drawrivers()

    plot_basin_boundaries_from_shape(b, linewidth = 1)
    if save_to_file:
        plt.savefig("amno_quebec_basins.pdf", bbox_inches='tight')




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

    ncfile = Dataset('data/infocell/quebec_masks_amno180x172.nc')
    vars = ncfile.variables

    the_data = np.transpose( vars['RDO'][:] )


    for i in range(n_cols):
        for j in range(n_rows):
            if cells[i][j].next is not None: to_plot[i,j] = 1
            if not cells[i][j].direction_value: to_plot[i,j] = 3
            if the_data[i, j] == 1 and cells[i][j].direction_value == 0: to_plot[i, j] = 2
    color_map = mpl.cm.get_cmap('jet', 3)
    m.scatter(xs, ys, c = to_plot, cmap = color_map , marker='s', s=150,
              linewidth = 0, alpha = 0.4)

    m.drawcoastlines()
    ymin, ymax = plt.ylim()
    plt.ylim(ymin + 0.1*(ymax - ymin), ymax * 0.35)

    xmin, xmax = plt.xlim()
    plt.xlim(xmin + (xmax - xmin) * 0.5, 0.95*xmax)
    ncfile.close()



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
        if basin is not None:
            for i, j, outlet in zip(i_list, j_list, basin.exit_cells):
                outlet.set_next(cells[i][j])
    pass


def read_basins(path = 'data/infocell/amno180x172_basins.nc'):
    """
    reads data from netcdf files and fills in the basins array
    """

    descr_map = get_basin_descriptions()
    ncfile = Dataset(path)
    id = 1



    for name, values in ncfile.variables.iteritems():
        data = values[:]
        if n_cols == data.shape[1]:#transpose if necessary
            the_values = np.transpose(data)
        else:
            the_values = data
            
        the_basin = Basin(id = id, name = name)
        if descr_map.has_key(name):
            the_basin.description = descr_map[name]

        basins.append(the_basin)

        the_is, the_js = np.where(the_values == 1)
        #add cells to a basin
        for i, j in zip(the_is, the_js):
            the_basin.add_cell(cells[i][j])
 
        id += 1

    ncfile.close()
    return basins
    pass

def get_domain_mask(path = 'data/infocell/amno180x172_basins.nc'):
    ds = Dataset(path)
    result = None
    for v in ds.variables.values():
        if result == None:
            result = v[:]
        else:
            result += v[:]
    ds.close()
    return result
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
    """
        returns the list of neighbor cells of the current one,
        the neighbors belong to the same basin as a cell
    """
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
                if prev.basin is None:
                    to_del.append(prev)


            for x in to_del:
                x.set_next(None)


def get_cells_without_basin():
    missing = []
    for i in range(n_cols):
        for j in range(n_rows):
            if cells[i][j].basin is not None: continue
            neighbors = get_all_neighbors(cells[i][j])
            is_missing = 0
            for neighbor in neighbors:
                if neighbor.basin is None:
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
            if the_cell.basin is None:
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
    path = 'data/hydrosheds/directions_qc_amno.nc'
    ncFile = Dataset(path)
    read_cell_area()


    inext_var = ncFile.variables['flow_direction_index0'][:]
    jnext_var = ncFile.variables['flow_direction_index1'][:]


    slopes = ncFile.variables['slope'][:]
    channel_length = ncFile.variables['channel_length'][:]

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
   # check_for_loops()

   # read_clay()
   # read_sand()
    ncFile.close()
    return basins




def plot_directions_from_file(path = 'data/hydrosheds/directions_qc_dx0.1.nc', basemap = None,
                              create_figure = True
                              ):
    """
    TODO: document me !!!!
    """
    ncFile = Dataset(path)


    inext_var = ncFile.variables['flow_direction_index0'][:]
    jnext_var = ncFile.variables['flow_direction_index1'][:]

    lons = ncFile.variables['lon'][:]
    lats = ncFile.variables['lat'][:]

    lons = np.array(lons)


    #if new figure creation is requested
    if create_figure:
        plt.figure()

    print np.min(lons), np.max(lons)

    if basemap is None:
        basemap = Basemap(resolution = 'i', llcrnrlon = np.min(lons),
                                        llcrnrlat = np.min(lats),
                                        urcrnrlon = np.max(lons),
                                        urcrnrlat = np.max(lats)
                                        )



    lons[lons >= 180] -= 360
    
    lons, lats = basemap(lons, lats)



    print lons.shape
    nx, ny = lons.shape
    u = np.ma.masked_all(lons.shape)
    v = np.ma.masked_all(lons.shape)

    for i in xrange(nx):
        for j in xrange(ny):
            i1, j1 = inext_var[i, j], jnext_var[i, j]
            if i1 >= 0 and i1 < nx and j1 >= 0 and j1 < ny:
                u[i,j] = lons[i1,j1] - lons[i,j]
                v[i,j] = lats[i1,j1] - lats[i,j]
#                plt.annotate(str((i,j)), xy = (lons[i,j], lats[i,j]))

    print np.ma.min(u), np.ma.max(u)
    basemap.quiver(lons, lats, u, v, width = 0.06 , units='xy', pivot = 'tail')
   



#    ymin, ymax = plt.ylim()
#    xmin, xmax = plt.xlim()
#
#    plt.xlim(xmin * 0.5, xmin * 0.25)
#    plt.ylim(ymax * 0.4, ymax * 0.9)
    ncFile.close()
    pass



def read_derived_from_hydrosheds():

    """
    !!!!!!!!!!!!!!!!!!!!!
    Plotting with Zooming
    """

   # get_cells_from_infocell('data/infocell/HQ2_infocell.txt')

    path = 'data/hydrosheds/directions_qc_amno.nc'
    ncFile = Dataset(path)
  #  read_cell_area()


    inext_var = ncFile.variables['flow_direction_index0'][:]
    jnext_var = ncFile.variables['flow_direction_index1'][:]
    

    slopes = ncFile.variables['slope'][:]
    channel_length = ncFile.variables['channel_length'][:]
    

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
                the_cell.set_next(cells[inext][jnext])
            else:
                the_cell.set_next(None)

            the_cell.channel_length = channel_length[i, j] if channel_length[i, j] > 0 else (the_cell.area) ** 0.5 * 1000.0



#    calculate_drainage_areas()
#    check_for_loops()
    plot_basins()

    print 'plotted basins'
    plot_directions(cells)
    print 'plotted directions'
   # plot_utils.zoom_to_qc(plotter = plt)

    domain_mask = get_domain_mask()

    x_start = np.ma.min(xs[domain_mask == 1])
    x_end = np.ma.max(xs[domain_mask == 1]) 

    y_start = np.ma.min(ys[domain_mask == 1]) 
    y_end = np.ma.max(ys[domain_mask == 1])

    marginx = abs(x_start) * 1.0e-1
    marginy = abs(y_start) * 1.0e-1
    x_start -= marginx * 0.2
    x_end += marginx
    y_start -= marginy
    y_end += marginy


    plt.xlim(x_start, x_end)
    plt.ylim(y_start, y_end)

     #inset axes for zoomed plot
    ax = plt.gca()
    theBasemap = Basemap(projection = 'npstere',
                        lat_ts = 60, lat_0 = -10, lon_0 = -115,
                        boundinglat = 0, resolution='c')

    axins = zoomed_inset_axes(ax, 0.1, loc = 4)
    x, y = theBasemap(lons, lats)
    x1 = np.min(x)
    x2 = np.max(x)
    y1 = np.min(y)
    y2 = np.max(y)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    coords = [
        [x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]
    ]

    axins.add_patch(Polygon(coords, facecolor = 'none', linewidth = 5, edgecolor = 'r'))

    #domain_mask = get_domain_mask()
    domain_mask = np.ma.masked_where(domain_mask == 0, domain_mask)
    theBasemap.pcolormesh(x, y, domain_mask , vmax = 1, vmin = 0)
    theBasemap.drawcoastlines()

    axins.annotate('AMNO', (x1 + abs(x1) * 1.0e-1, y1 + abs(y1) * 1.0e-1),
                    bbox = dict(facecolor = 'white', pad = 4))

    #plot_basins(sign_basins = False, draw_rivers = False, basemap = theBasemap)
    #center zoom on amno
    plt.xlim(x1  , x2)
    plt.ylim(y1  , y2)

    
#    read_clay()
#    read_sand()

#    write_new_infocell()
    ncFile.close()

   
  


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
    """
    Plot boundaries of the cell
    """
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
    read_derived_from_hydrosheds()
    
#    plot_directions_from_file(path = 'data/hydrosheds/directions_qc_amno.nc')
#    plot_directions_from_file(path = 'data/hydrosheds/directions0.nc')
    plt.show()
    print 'Execution time %f seconds' % (time.clock() - t0)