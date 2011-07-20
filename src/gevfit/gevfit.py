from matplotlib.ticker import LinearLocator
import os.path
__author__="huziy"
__date__ ="$22 oct. 2010 12:00:55$"


from mpl_toolkits.basemap import NetCDFFile
import data_select
from math import isnan
from math import isinf
from plot_utils import draw_meridians_and_parallels

import application_properties
import numpy as np
import matplotlib.pyplot as plt
import os

#tick locators
from matplotlib.ticker import *

import shape.basin_boundaries as boundaries

from math import *
#from osgeo import gdal, ogr
import scipy.optimize as opt

from scipy.special import gamma

from datetime import datetime
import matplotlib as mpl
import matplotlib_helpers.my_colormaps as my_cm

from datetime import timedelta

import members
import pylab
inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (sqrt(5.0) - 1.0) / 2.0       # Aesthetic ratio
fig_width = 2000 * inches_per_pt          # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = [fig_width, fig_height]

font_size = 25

def zoom_to_qc():
    ymin, ymax = plt.ylim()
    plt.ylim(ymin + 0.05 * (ymax - ymin) , ymax * 0.25)

    xmin, xmax = plt.xlim()
    plt.xlim(xmin + (xmax - xmin) * 0.55, 0.72*xmax)



params = {
        'axes.labelsize': font_size,
        'font.size':font_size,
        'text.fontsize': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'figure.figsize': fig_size
        }

pylab.rcParams.update(params)


import pickle

#set current directory to the root directory of the project
application_properties.set_current_directory()

import lmoments

from plot2D.map_parameters import polar_stereographic

xs = polar_stereographic.xs
ys = polar_stereographic.ys
m = polar_stereographic.basemap



BIG_NUM = 1.0e6



#np.seterr(all='raise', under='ignore')
#sigma, mu, ksi, zero_fraction = pars
def get_high_ret_level_stationary(pars, return_period):
    if pars[0] == None:
        return -1
    sigma, mu, ksi, zero_fraction = pars

    y = np.log(float(return_period) / (return_period - 1.0) )
    if np.abs(ksi) < 1.0e-2:
        lev = -sigma * np.log(y) + mu
    else:
        lev = sigma / ksi * ( np.power( y , -ksi) - 1.0 ) + mu
    return lev

#sigma, mu, ksi, zero_fraction = pars
def get_low_ret_level_stationary(pars, return_period):
    return get_low_ret_level(params = pars[0:3], return_period = return_period, zero_fraction = pars[3])

#rlevel = sigma/ksi * (ln(T/(1-Tz))^(-ksi) - 1) + mu
def get_low_ret_level(params = [], return_period = 2, zero_fraction = 0.0):
    if 1.0 / return_period <= zero_fraction:
        return 0

    if params[0] == None:
        return -1
    sigma, mu, ksi = params

    if np.abs(ksi) < 1.0e-2:
        lev = mu - np.log(np.log(return_period)) * sigma
    else:
        y = np.log(return_period * (1.0 - zero_fraction)/(1.0 - return_period * zero_fraction))
        lev = sigma / ksi * ( np.power(y, -ksi) - 1.0) + mu
    return lev

#Martins E.S. (2000)
def ksi_pdf(ksi):
    if abs(ksi) >= 0.5:
        return 0
    else:
        return 1     #temporary disable prior distribution function

    p = 6.0
    q = 9.0
    b = gamma(p) * gamma(q) / gamma(p + q)
    return (-ksi + 0.5) ** (p - 1) * (0.5 + ksi) ** (q - 1) / b


#if i_list or j_list are None, then take indices from the indices_file
def plot_data(data = None, imagefile = 'image.png',
              units = 'm**3/s',
              minmax = (None, None),
              indices_file = 'data/streamflows/hydrosheds_euler9/aex_discharge_1970_01_01_00_00.nc',
              color_map = mpl.cm.get_cmap('RdBu', 20),
              ticks_locator = LinearLocator(),
              i_list = None, j_list = None, title = ''):

    if imagefile != None:
        plt.clf()

    if i_list == None or j_list == None:
        i_list, j_list = data_select.get_indices_from_file(indices_file)
    to_plot = np.ma.masked_all(xs.shape)
    for i, j, value in zip(i_list, j_list, data):
        to_plot[i, j] = value

    plt.title(title)
    plt.pcolormesh(xs,ys,to_plot.copy(),  cmap = color_map, edgecolors = 'None',
                    antialiased = True, vmin = minmax[0], vmax = minmax[1])
    m.drawcoastlines()
    boundaries.plot_basin_boundaries_from_shape(m, plotter = plt, linewidth = 0.5)
    draw_meridians_and_parallels(m, 10)

#    plot_directions(data_mask = to_plot)

    cb = plt.colorbar(ticks = ticks_locator, format = "%.1f")
    cb.ax.set_ylabel(units)

    ymin, ymax = plt.ylim()
    plt.ylim(ymin + 0.05 * (ymax - ymin) , ymax * 0.25)
#
    xmin, xmax = plt.xlim()
    plt.xlim(xmin + (xmax - xmin) * 0.55, 0.72*xmax)

    
    if imagefile != None:
        plt.savefig(imagefile, bbox_inches = 'tight')




def qfunc(x, sigma, mu, ksi):
    '''
    Helper function (1 + ksi*(x - mu) / sigma)^(-1/ksi)
    '''
    if sigma <= 1.0e-10: #sigma > 0
        return None

    if 1.0 + ksi * (x - mu) / sigma <= 0:
        return None

    if abs(ksi) <= 1.0e-2: #ksi != 0
        the_power = -(x - mu) / sigma
        return np.exp(the_power)

    the_base = 1.0 + ksi * (x - mu) / sigma
    result = the_base ** (-1.0 / ksi)


    if isinf(result) or result == 0:
        return None

    if result == 0:
        print x, mu, sigma
        print the_base
        print -1.0 / ksi


    message = 'in qfunc: result = {0}, x = {1}, sigma = {2}, mu = {3}, ksi = {4}, the_base = {5}'
    assert result > 0.0, message.format(result, x, sigma, mu, ksi, the_base)

    if isinf(result) or isnan(result):
        print 'Too big numbers: ' , the_base, result
        assert False, 'qfunc = {0}'.format(result)
        return None
    return result

#-ln(gevpdf * ksi_pdf)
def objective_function_stationary_high(pars, data):
    result = 0.0
    sigma, mu, ksi = pars

    ksi_probability = ksi_pdf(ksi)

    if ksi_probability == 0:
        return BIG_NUM

    for the_data in data:

        qi = qfunc(the_data, sigma, mu, ksi)
        if qi == None:
            return BIG_NUM
        assert qi > 0, 'qi = {0}'.format(qi)
       
        minus_ln_pdfi = log(sigma) - (ksi + 1.0) * log(qi) + qi - log(ksi_probability)
        if minus_ln_pdfi < 0:
            return BIG_NUM
        result += minus_ln_pdfi

    assert np.isfinite(result), 'result is nan, result = {0}'.format(result)
    return result

#-ln(gevpdf)
def objective_function_stationary_low(pars, data):
    result = 0.0
    sigma, mu, ksi = pars


    ksi_probability = ksi_pdf(ksi)

    if ksi_probability == 0:
        return BIG_NUM


    for the_data in data:
        qi = qfunc(the_data, sigma, mu, ksi)
        if qi == None:
            return BIG_NUM
        assert qi > 0, 'qi = {0}'.format(qi)
        minus_ln_pdfi = np.log(sigma) - (ksi + 1.0) * log(qi) + qi - log(ksi_probability)
        if minus_ln_pdfi < 0:
            return BIG_NUM

        result += minus_ln_pdfi

    assert np.isfinite(result), 'result is nan, result = {0}'.format(result)
    return result


#vals timeseries for a point
def get_initial_params(vals):
    assert len(vals) > 0, 'len(vals) = {0}'.format(len(vals))
    ksi0 = 0.1

    if len(vals) == 1:
        bias = 1
    else:
        bias = 0

    sigma0 = np.sqrt(6.0 * np.cov(vals, bias = bias)) / pi

    if sigma0 == 0:
        sigma0 = 0.2 * np.mean(vals)

    mu0 = np.mean(vals) - 0.57722 * sigma0

    assert np.isfinite(mu0), 'mu0 = {0}'.format(mu0)
    return [sigma0, mu0, ksi0]


#returns initial parameters using L-moments
def get_initial_params_using_lm(vals):
    sorted_vals = sorted(vals)
    the_moments = lmoments.samlmu(sorted_vals, 3)
    mu, sigma, ksi = lmoments.pelgev(the_moments[0:3])
    return [sigma, mu, -ksi] #-ksi because they are using -ksi convention



#optimize using maxima over certain period
#returns [sigma, mu, ksi, zero_fraction]
def optimize_stationary_for_period(extremes, high_flow = True, use_lmoments = False):

    extremes = 1.0e6 * extremes

    #if all values are 0, do not optimize, return None for the parametes values
    if np.min(extremes) < 1.0:
        return [None, None, None, 1.0]

    indices = np.where(extremes > 1)
    zero_fraction = 1.0 - extremes[indices].shape[0] / float(len(extremes))

    ##L-moments
    if use_lmoments:
        pars = get_initial_params_using_lm(extremes[indices])
        pars.append(zero_fraction)
        lev = get_high_ret_level_stationary(pars, 10.0)
        if isnan(lev):
            print pars
            print extremes[indices].tolist()
            assert False, 'lev = {0}'.format(lev)
        return pars


    if high_flow:
        objective_function = objective_function_stationary_high
    else:
        objective_function = objective_function_stationary_low

    pars0 = get_initial_params(extremes[indices])



#    pars0 = get_initial_params_using_lm(extremes[indices])
#    if objective_function(pars0, extremes[indices]) == BIG_NUM:
#        pars0 = get_initial_params(extremes[indices])

     #default simplex
    pars, z, niter, funcalls, warnflag, all_vecs = opt.fmin(objective_function, pars0,
                                                       args = (extremes[indices],),
                                                       maxfun = 10000,
                                                       full_output = True,
                                                       disp = False,
                                                       maxiter = 10000,
                                                       retall = True
                                                       )

    #powell method
#    pars, z, direc, niter, funcalls, warnflag, all_vecs = opt.fmin_powell(objective_function,
#                                                        pars0,
#                                                        args = (extremes[indices],),
#                                                        maxfun = 10000,
#                                                        full_output = True,
#                                                        disp = False,
#                                                        maxiter = 10000,
#                                                        retall = True
#                                                        )


    if warnflag != 0:
        print extremes
        print warnflag
        print pars


    #assert warnflag == 0, 'warnflag = {0}, z = {1}, \n extremes = {2}'.format(warnflag, z, str(extremes))
    assert z > 0, 'z <= 0'
    
    if z < 0:
        print 'converged to negative objective function'
        return [None, None, None, zero_fraction]



    if z == BIG_NUM:
        print 'high_flow = ', high_flow
        print extremes
        print extremes[indices].tolist()
        print pars
        print all_vecs
 #       assert False
        return [None, None, None, zero_fraction]

    assert z != BIG_NUM, 'z == BIG_NUM'
    assert z >= 0, 'z < 0'

    pars[0] = pars[0] / (1.0e6)
    pars[1] = pars[1] / (1.0e6)
    pars = np.append(pars, zero_fraction)
    return pars



def optimize_stationary_for_period_and_all_cells_using_data(
                data = None,
                high_flow = True):

    pars_set = []
    #for all grid cells
    for pos in range(data.shape[1]):
        pars = optimize_stationary_for_period(data[:,pos], high_flow = high_flow)
        pars_set.append(pars)
    return pars_set



def optimize_stationary_for_period_and_all_cells(
                data_file = 'data/streamflows/hydrosheds_euler9/aex_discharge_1970_01_01_00_00.nc',
                paramfile = 'gev_params_stationary',
                high_flow = True,
                start_month = 1, end_month = 12,
                start_date = datetime(1970,1,1,0,0),
                end_date = datetime(1999,12, 31,0,0),
                event_duration = timedelta(days = 1)):

    print paramfile

    #check whether optimization is required
    if os.path.isfile(paramfile):
        print 'already optimized, if you want to reoptimize delete %s' % paramfile
        pars_set = pickle.load(open(paramfile))
        return pars_set

    #get streamflow data
    streamflow, times, xs, ys = data_select.get_data_from_file(path = data_file)

    data = []
    for pos in range(streamflow.shape[1]):
        if high_flow:
            data1 = data_select.get_period_maxima(streamflow[:,pos], times,
                            start_date = start_date,
                            end_date = end_date,
                            start_month = start_month,
                            end_month = end_month,
                            event_duration = event_duration
                            )
        else:
            data1 = data_select.get_period_minima(streamflow[:, pos], times,
                            start_date = start_date,
                            end_date = end_date,
                            start_month = start_month,
                            end_month = end_month,
                            event_duration = event_duration
                            )
        data.append(data1.values())


    data = np.array(data).transpose()
    pars_set = optimize_stationary_for_period_and_all_cells_using_data(data = data,
                                                    high_flow = high_flow)
    f = open(paramfile ,'w')
    pickle.dump(pars_set, f)
    f.close()
    return pars_set



def plot_low_flows(period = 10,
                   imagefile = 'figure.png',
                   pars_set = None,
                   indices_file = 'data/streamflows/hydrosheds_euler9/aex_discharge_1970_01_01_00_00.nc'):
    plt.clf()
    

    levs = []
    i_list, j_list = data_select.get_indices_from_file(indices_file)

    #iterate through all grid points
    for pars in pars_set:
        levs.append( get_low_ret_level_stationary(pars, period) )

    to_plot = np.ma.masked_all(xs.shape)
    for lev, i, j in zip(levs, i_list, j_list):
        assert not isinf(lev) and not isnan(lev)
        #assert lev >= 0
        if lev >= 0:
            to_plot[i,j] = lev

   

    nticks = 15
    color_map = mpl.cm.get_cmap('RdBu', nticks)
    m.pcolormesh(xs, ys, to_plot.copy(), cmap = color_map)
    m.drawcoastlines()
    boundaries.plot_basin_boundaries_from_shape(m, plt,  linewidth = 0.5)
#    plot_directions(data_mask = to_plot)
    plt.title('low flow, return period is {0}'.format(period))
    int_ticker = LinearLocator(numticks = color_map.N + 1)
    plt.colorbar(ticks = int_ticker, format = '%.1f')

    zoom_to_qc()
    print 'saving %s' % imagefile
    
    plt.savefig(imagefile, bbox_inches = 'tight')
    


#return period in years
#start - start year
def plot_high_flows(period = 10, 
                    imagefile = 'figure.png',
                    pars_set = None,
                    indices_file = 'data/streamflows/hydrosheds_euler9/aex_discharge_1970_01_01_00_00.nc'):
    print 'generating %s ' % imagefile

    plt.clf()
    levs = []
    i_list, j_list = data_select.get_indices_from_file(indices_file)

    #iterate through all grid points
    for pars in pars_set:
        lev = get_high_ret_level_stationary(pars, period)
        if lev < 0:
            print 'period = ', period
            print 'pars = ', pars
            assert False, 'in plot_high_flows'
            
        levs.append( lev )
    
    to_plot = np.ma.masked_all(xs.shape)
    for lev, i, j in zip(levs, i_list, j_list):
        assert lev >= 0, pars
        assert np.isfinite(lev)
        if isinf(lev):
            print lev
        to_plot[i,j] = lev


    nticks = 15
    color_map = mpl.cm.get_cmap('RdBu',nticks)
    int_ticker = LinearLocator(numticks = color_map.N + 1)

    m.pcolormesh(xs, ys, to_plot.copy(), cmap = color_map)
    plt.title('high flow, return period is {0}'.format(period))

    boundaries.plot_basin_boundaries_from_shape(m, plt,  linewidth = 0.5)
    m.drawcoastlines()
#    plot_directions(data_mask = to_plot)
    plt.colorbar( ticks = int_ticker, format = "%.1f" )

    zoom_to_qc()

    plt.savefig(imagefile, bbox_inches = 'tight')



def main():
   pass




def get_gevd_params_for_id_and_type(id = '', high = True):
    prefix = 'gev_params_stationary'
    postfix = '_high' if high else '_low'
    file = prefix + '_' + id + postfix
    return pickle.load(open(file))


def get_levels_for_type_and_id(id, return_period = None, type = 'high'):
    file_name_prefix = 'gev_params_stationary'
    if type == 'high':
        return get_high_levels_for_id(id, file_name_prefix, postfix = '_' + type, return_period = return_period)
    else:
        return get_low_levels_for_id(id, file_name_prefix, postfix = '_' + type, return_period = return_period)



def get_high_levels_for_id(id, prefix = 'gev_params_stationary', postfix = '_high' , return_period = 10):
    file = prefix + '_' + id + postfix
    pars_set = pickle.load(open(file))

    field = np.zeros((len(pars_set),))
    for pos, pars in enumerate(pars_set):
        field[pos] = get_high_ret_level_stationary(pars, return_period)
    return field


def get_low_levels_for_id(id, prefix = 'gev_params_stationary', postfix = '_low' , return_period = 10):
    file = prefix + '_' + id + postfix
    pars_set = pickle.load(open(file))

    field = np.zeros((len(pars_set),))
    for pos, pars in enumerate(pars_set):
        field[pos] = get_low_ret_level_stationary(pars, return_period)
    return field



#class to use in dictionary
class TypePeriodKey():
    pass



###Main function for determining parameters of GEV distribution, return levels and
### changes between current and future climate
def stationary():

    data_folder = 'data/streamflows/hydrosheds_euler10_spinup100yrs'
    current_data_path_pattern = '%s_discharge_1970_01_01_00_00.nc'
    future_data_path_pattern = '%s_discharge_2041_01_01_00_00.nc'


    current_start_date = datetime(1970, 1, 1, 0, 0)
    current_end_date = datetime(1999, 12, 31,0, 0)

    future_start_date = datetime(2041, 1, 1, 0, 0)
    future_end_date = datetime(2070, 12, 31,0, 0)


    high_return_periods = [10, 30, 50]
    high_start_month = 3
    high_end_month = 7
    high_event_duration = timedelta(days = 1)


    low_return_periods = [2, 5, 10]
    low_start_month = 1
    low_end_month = 5
    low_event_duration = timedelta(days = 15)





    all_return_periods = []
    all_return_periods.extend(high_return_periods)
    all_return_periods.extend(low_return_periods)

    plot_return_levels = False


    #calculate parameters of the gev distriution for each member
    #calculate and plot return levels
    for current_id in members.current_ids:
        param_file = 'gev_params_stationary'
        param_file += '_' + current_id + '_low'
        data_file = current_data_path_pattern % current_id
        data_file = os.path.join(data_folder, data_file)
        pars_set = optimize_stationary_for_period_and_all_cells(data_file,
                paramfile = param_file , 
                high_flow = False,
                start_month = low_start_month,
                end_month = low_end_month,
                start_date = current_start_date,
                end_date = current_end_date,
                event_duration = low_event_duration)

        if plot_return_levels:
            for period in low_return_periods:
                plot_low_flows(period = period,
                       imagefile = '%drlevel_low_stationary_%s.png' % (period, current_id),
                       pars_set = pars_set)


        

        param_file = 'gev_params_stationary'
        param_file += '_' + current_id + '_high'
        pars_set = optimize_stationary_for_period_and_all_cells(data_file,
                paramfile = param_file ,
                high_flow = True,
                start_month = high_start_month,
                end_month = high_end_month,
                start_date = current_start_date,
                end_date = current_end_date,
                event_duration = high_event_duration)

        if plot_return_levels:
            for period in high_return_periods:
                plot_high_flows(period = period, imagefile = '%drlevel_high_stationary_%s.png' % (period, current_id),
                                pars_set = pars_set)

        

    print 'Finished optimizing for current climate'


    
    for future_id in members.future_ids:
        param_file = 'gev_params_stationary'
        param_file += '_' + future_id + '_low'
        data_file = future_data_path_pattern % future_id
        data_file = os.path.join(data_folder, data_file)
        pars_set = optimize_stationary_for_period_and_all_cells(data_file,
                paramfile = param_file ,
                high_flow = False,
                start_month = low_start_month,
                end_month = low_end_month,
                start_date = future_start_date,
                end_date = future_end_date,
                event_duration = low_event_duration)


        if plot_return_levels:
            for period in low_return_periods:
                plot_low_flows(period = period, imagefile = '%drlevel_low_stationary_%s.png' % (period, future_id),
                       pars_set = pars_set)
        

        param_file = 'gev_params_stationary'
        param_file += '_' + future_id + '_high'
        pars_set = optimize_stationary_for_period_and_all_cells(data_file,
                paramfile = param_file ,
                high_flow = True,
                start_month = high_start_month,
                end_month = high_end_month,
                start_date = future_start_date,
                end_date = future_end_date,
                event_duration = high_event_duration)

        if plot_return_levels:
            for period in high_return_periods:
                plot_high_flows(period = period, imagefile = '%drlevel_high_stationary_%s.png' % (period, future_id),
                        pars_set = pars_set)


    print 'Finished optimizing for future climate'
    print 'Finished calculating return levels !!!'

    plot_mean_changes = False
    if not plot_mean_changes:
        return

###############################current climate return levels
    print 'Calculating mean high flow return levels for current climate ...'
    current_rl_means = {}
    future_rl_means = {}
    the_type_to_periods = {'high': high_return_periods, 'low': low_return_periods}



    keys = []
    for the_type, periods in the_type_to_periods.iteritems():
        for period in periods:
            k = TypePeriodKey()
            k.type = the_type
            k.return_period = period
            keys.append(k)


    for key in keys:
        current_rl_means[key] = []
        future_rl_means[key] = []






    #collect return levels for corresponding type(high , low) and return period for each member and then take mean
    for current_id in members.current_ids:
        for key in keys:
            future_id = members.current2future[current_id]
            the_field_current = get_levels_for_type_and_id(current_id, return_period = key.return_period, type = key.type)
            the_field_future = get_levels_for_type_and_id(future_id, return_period = key.return_period, type = key.type)


            indices = np.where((the_field_current > 0) & (the_field_future >= 0) )
            to_plot = np.ma.masked_all(the_field_current.shape)
            to_plot[indices] = (the_field_future[indices] - the_field_current[indices]) / the_field_current[indices] * 100.0


            file_name = '{0}-{1}_{2}_{3}yr_change.png'.format(current_id, future_id, key.type, key.return_period)

            delta = np.max(np.abs(to_plot[indices]))
            delta = min(100.0, delta)
            plot_data(data = to_plot, imagefile = file_name,
                  units = '%', minmax = (-125, 125), color_map = my_cm.get_red_blue_colormap(ncolors = 16),
                  title = '{0}-{1}, change, {2}, return period: {3}'.format(current_id,
                  future_id, key.type, key.return_period)
            )


            future_rl_means[key].append(the_field_future)
            current_rl_means[key].append(the_field_current)



            
    for key in keys:

        current_rl_means[key] = np.array(current_rl_means[key])
        current_rl_means[key] = np.ma.masked_where(current_rl_means[key] < 0, current_rl_means[key])

        future_rl_means[key] = np.array(future_rl_means[key])
        future_rl_means[key] = np.ma.masked_where(future_rl_means[key] < 0, future_rl_means[key])


        current_rl_means[key] = np.ma.mean(  current_rl_means[key]  , axis = 0)
        future_rl_means[key] = np.ma.mean(  future_rl_means[key]  , axis = 0)


#        plt.figure()
#        plt.subplot(2,1,1)
#        plt.title('current mean')
#        plot_data(current_rl_means[key], imagefile = None)
#
#        plt.subplot(2,1,2)
#        plt.title('future mean')
#        plot_data(future_rl_means[key], imagefile = None)
#
#        plt.savefig('means_%s_%dyr_rl.png' % (key.type, key.return_period))


###################################################
####Calculate differences between future and current return levels for 10 and 30 year
#### return period.
####


    for key in keys:
        current = current_rl_means[key]
        future = future_rl_means[key]
        indices = np.where((current > 0) & (future >= 0))
        to_plot = np.ma.masked_all(current.shape)
        to_plot[indices] = (future[indices] - current[indices]) / current[indices] * 100.0

        delta = np.max(np.abs(to_plot[indices]))

        if delta > 100:
            delta = 125
        else:
            delta = 40

        plot_data(data = to_plot, imagefile = '%s_%dyr_mean_change.png' % (key.type, key.return_period),
              units = '%', minmax = (-delta, delta),
              color_map = my_cm.get_red_blue_colormap(ncolors = 16), #mpl.cm.get_cmap('RdYlBu',20),
              title = '{0},return period {1}, mean changes'.format(key.type, key.return_period),
              ticks_locator = LinearLocator(numticks = 9)
              )


def plot_directions(data_mask = None):
    '''
        cells - 2D array of cells
        basins_mask - 1 where basins, 0 elsewhere
    '''

    u_plot = np.ma.masked_all(xs.shape)
    v_plot = np.ma.masked_all(xs.shape)


    f = NetCDFFile('data/hydrosheds/directions.nc')
    inext = f.variables['flow_direction_index0'][:]
    jnext = f.variables['flow_direction_index1'][:]

    nx, ny = xs.shape
    for i in range(nx):
        for j in range(ny):
            i1, j1 = inext[i, j], jnext[i, j]

            u_plot[i, j] = xs[i1, j1] - xs[i, j]
            v_plot[i, j] = ys[i1, j1] - ys[i, j]
            mag = np.sqrt(u_plot[i, j] ** 2 + v_plot[i, j] ** 2)
            u_plot[i, j] /= mag
            v_plot[i, j] /= mag
    
    print 'plotting directions'


    
    print 'calculated magnitude'

    indices = ~(data_mask.mask)
    m.quiver(xs[indices], ys[indices], u_plot[indices], v_plot[indices], scale = 6.5, width = 0.01, units = 'inches')

#    plt.savefig("flows_and_masks.png", bbox_inches='tight')



def test():
    pars = [8.4116509126033642e-05, 0.00084966170834377408, 0.10000000000000001]
    vals = [0.0010247972095385194, 0.0010247972095385194, 0.0012944934424012899,
            0.0042147189378738403, 0.00098561809863895178, 0.00095898169092833996,
            0.002480002585798502, 0.00084966170834377408, 0.0034388666972517967,
            0.0016178090590983629, 0.0013241175329312682, 0.0020841944497078657,
            0.001562419580295682, 0.0022000106982886791, 0.005726152565330267,
            0.0010590874589979649, 0.0014877116773277521, 0.0010104207322001457,
            0.0019218671368435025, 0.0030378694646060467, 0.0014164787717163563,
            0.00090275343973189592, 0.001988076139241457, 0.0026944593992084265,
            0.0033022623974829912, 0.0021143041085451841, 0.001547978725284338,
            0.0013833490666002035, 0.0042443717829883099, 0.0024236994795501232]

    print objective_function_stationary(pars, vals) == BIG_NUM
    print BIG_NUM


def gev_fit_all_members(high_flow = True, member_ids = [], data_folder = '', file_name_pattern = '',
                        start_date = None, end_date = None, start_month = 1, end_month = 12, duration_days = timedelta(days = 1)):

    param_file = 'high' if high_flow else 'low'
    for id in member_ids:
        param_file += '_' + id
    if os.path.isfile(param_file):
        print 'delete {0}, to reoptimize'.format(param_file)
        return pickle.load(open(param_file))

    #select data
    path_pattern = os.path.join(data_folder, file_name_pattern)
    all_extremes = []
    for id in member_ids:
        print id
        the_path = path_pattern.format(id)
        streamflow, times, i_indices, j_indices = data_select.get_data_from_file(the_path)

        if len(all_extremes) == 0:
            for i in range(streamflow.shape[1]):
                all_extremes.append([])


        for pos in range(streamflow.shape[1]):
            if high_flow:
                data1 = data_select.get_period_maxima(streamflow[:, pos], times,
                                start_date = start_date,
                                end_date = end_date,
                                start_month = start_month,
                                end_month = end_month,
                                event_duration = duration_days
                                )
            else:
                data1 = data_select.get_period_minima(streamflow[:, pos], times,
                                start_date = start_date,
                                end_date = end_date,
                                start_month = start_month,
                                end_month = end_month,
                                event_duration = duration_days
                                )
            all_extremes[pos].extend(data1.values())


    all_extremes = np.array(all_extremes).transpose()



 
    if np.any(all_extremes == None):
        assert False, 'all_extremes = ' + str(all_extremes)

    #optimize
    print all_extremes.shape
    assert all_extremes.shape[1] == 547
    param_set = optimize_stationary_for_period_and_all_cells_using_data(data = all_extremes,
                                        high_flow = high_flow
                                        )
    pickle.dump(param_set, open(param_file , 'wb'))
    return param_set
    pass


def fit_merged_for_current_and_future():


    the_types = [True, False]

    high_start_month = 3
    high_end_month = 7
    high_duration = timedelta(days = 1)

    low_start_month = 1
    low_end_month = 5
    low_duration = timedelta(days = 15)

    type_to_period_start_month = {True : high_start_month, False : low_start_month}
    type_to_period_end_month = {True : high_end_month, False : low_end_month}

    type_to_event_duration = {True : high_duration, False : low_duration}


    low_ret_periods = [2, 5, 10]
    high_ret_periods = [10, 30, 50]

    data_folder = 'data/streamflows/to_compare_with_Vincent'

    current_start_date = datetime(1961,1,1,0,0)
    current_end_date = datetime(1990,12, 31,0,0)

    future_start_date = datetime(2041,1,1,0,0)
    future_end_date = datetime(2070,12, 31,0,0)


    for hig_flow in the_types:
        pars_current = gev_fit_all_members(high_flow = hig_flow,
                        member_ids = members.current_ids,
                        data_folder = data_folder,
                        file_name_pattern = '{0}_discharge_1961_01_01_00_00.nc',
                        start_date = current_start_date,
                        end_date = current_end_date,
                        start_month = type_to_period_start_month[hig_flow] ,
                        end_month = type_to_period_end_month[hig_flow],
                        duration_days = type_to_event_duration[hig_flow])

        pars_future = gev_fit_all_members(high_flow = hig_flow,
                        member_ids = members.future_ids,
                        data_folder = data_folder,
                        file_name_pattern = '{0}_discharge_2041_01_01_00_00.nc',
                        start_date = future_start_date,
                        end_date = future_end_date,
                        start_month = type_to_period_start_month[hig_flow] ,
                        end_month = type_to_period_end_month[hig_flow],
                        duration_days = type_to_event_duration[hig_flow])

        

        assert len(pars_current) == 547
        assert len(pars_future) == 547
        periods = high_ret_periods if hig_flow else low_ret_periods
        for the_period in periods:
            rl_current = []
            rl_future = []
            for par_current, par_future in zip(pars_current, pars_future):
                if hig_flow:
                    rl_current.append( get_high_ret_level_stationary(par_current, the_period) )
                    rl_future.append( get_high_ret_level_stationary(par_future, the_period) )
                else:
                    rl_current.append( get_low_ret_level_stationary(par_current, the_period) )
                    rl_future.append( get_low_ret_level_stationary(par_future, the_period) )


            
            rl_current = np.array(rl_current)
            rl_future = np.array(rl_future)


            assert np.max(rl_current) < 20000

            
            deltas = np.ma.masked_all(rl_current.shape)
            indices = np.where((rl_current > 0) & (rl_future >= 0))
            deltas[indices] = (rl_future[indices] - rl_current[indices]) / rl_current[indices] * 100.0

            type_str = 'high' if hig_flow else 'low'
            interval = np.ma.max(np.ma.abs(deltas))
            interval = min(100, interval)

            print 'current - {0}, period: {1}, minmax = {2}, {3}'.format(type_str, the_period, np.min(rl_current), np.max(rl_current))
            print 'future - {0}, period: {1}, minmax = {2}, {3}'.format(type_str, the_period, np.min(rl_future), np.max(rl_future))

            plot_data(deltas, imagefile = '{0}_{1}_change_fit_all.png'.format(type_str, the_period), units = '%', minmax = (-interval, interval),
                            color_map = my_cm.get_diff_colormap(ncolors = 8),
                            title = '{0}, {1} years, fit all members'.format(type_str, the_period))

    pass


def test_lm():
    x = [360.228515625, 513.506103515625, 273.85031127929688, 340.94839477539062,
         244.13925170898438, 283.414306640625, 394.42819213867188, 284.3604736328125,
         281.26956176757812, 241.46173095703125, 489.75482177734375, 236.31536865234375,
         407.55133056640625, 244.6295166015625, 432.40670776367188, 260.501953125,
         517.23052978515625, 317.6553955078125, 407.61935424804688, 275.0709228515625,
         330.369140625, 285.92086791992188, 247.9954833984375, 344.34811401367188,
         379.55596923828125, 330.80569458007812, 312.35330200195312, 251.79550170898438,
         372.66928100585938, 239.72474670410156]

    print get_initial_params_using_lm(x)
    print np.mean(x)
    pars = [ 128.28104749,  578.4927539 ,    0.62410911]
    data = [588.4747314453125, 693.6640625, 519.03155517578125, 716.58013916015625,
            686.29168701171875, 432.65786743164062, 682.72113037109375, 730.12603759765625,
            698.971923828125, 491.75332641601562, 597.258544921875, 487.13619995117188, 482.33123779296875,
            573.57861328125, 801.67169189453125, 616.41668701171875, 690.954833984375, 671.31646728515625,
            680.87554931640625, 534.18414306640625, 427.86019897460938, 236.22953796386719, 691.40972900390625,
            599.84637451171875,
            545.3563232421875, 553.059814453125, 549.1295166015625, 658.3983154296875, 719.122802734375,
            636.84906005859375]

    the_moments = lmoments.samlmu(sorted(data),5)
    pars = lmoments.pelgev(the_moments[0:3])
    print pars
    mu, sigma, xi = pars
    print objective_function_stationary_high([sigma, mu, -xi], data)



    

if __name__ == "__main__":
#    fit_merged_for_current_and_future()
    stationary()
#    test_lm()

    print "Hello World"
