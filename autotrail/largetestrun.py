import numpy as np
import matplotlib.pyplot as plt

import osmnx as ox
import gpxpy
import shapely

#from
from planit.autotrail.process_gpx_data import *

from planit.autotrail.trailmap import TrailMap
from planit.autotrail import maptrails
from planit.osm_data import osm_process
from planit.osm_data import osm_fetch

import osmnx
import osmnx.geocoder as geocoder
import seaborn as sns
import os, sys

from matplotlib import rc, cm
viridis = cm.get_cmap('viridis')
magma   = cm.get_cmap('magma')
plasma  = cm.get_cmap('plasma')

fsize = 17
rc('text', usetex=False)
rc('font', size=fsize)#, ftype=42)
line_width = 3.5
point_size = 30

try:
    import cPickle as pickle
except:
    import pickle

def make_trails(location, target_values, start_coords):
    """
    """

    lat, lng = geocoder.geocode(location)
    center_point = (lat,lng)
    print("Location: ", location, lat, lng)
    tmap = osm_process.osmnx_trailmap(center_point = center_point,
                                      dist = 40233.6)
    tmap.ensure_edge_attributes()


    _,start_node = tmap.nearest_node( start_coords[0], start_coords[1])
    start_node = start_node[0]
    end_node   = start_node

    tmap._weight_precision = 6
    tmap._dynamic_weighting = True
    tmap._neg_weight = False

    tmap._assign_weights(target_values)

    totals, possible_routes, scores = tmap.multi_find_route(start_node,
                                                            target_values,
                                                            n_routes = 100,
                                                            iterations = 100,
                                                            end_node = end_node,
                                                     reinitialize=True,subgraph_filter=True,
                                                     reset_used_counter=True)




    return totals, possible_routes, scores, tmap


locations = ['Mount Wilson California',
                 'Mount Wilson California',
                 'Boulder CO',
                 'Boulder CO',
                 'Issaquah Washington',
                 'Issaquah Washington',
                 'Issaquah Washington']


du = 1609.34 # mi to meters
eu = 0.3048  # ft to meters

target_dicts = [  {'distance' : 14*du, 'elevation_gain' : 5000.0*eu},
                      {'distance' : 24*du, 'elevation_gain' : 9000.0*eu},
                      {'distance' : 6*du, 'elevation_gain' : 3000.0*eu},
                      {'distance' : 4*du, 'elevation_gain' : 500.0*eu},
                      {'distance' : 12*du, 'elevation_gain' : 4000.0*eu},
                      {'distance' : 24*du, 'elevation_gain' : 10000.0*eu},
                      {'distance' : 8*du, 'elevation_gain' : 2000.0*eu}]

start_coords = [ (-118.147562742233, 34.2238723),  # (long, lat) # sunset / mt low
                     (-118.13075333833697, 34.20405622317053), # sam merril / echo
                     (-105.27743875980377, 39.99780230823386), # flatirons
                     (-105.2365565299988,39.88560274046906),   # boulder flats
                     (-122.01346278190614, 47.48889049475722), # tiger south
                     (-121.98706179857255,47.530705357315355), # tiger north
                     (-121.98706179857255,47.530705357315355)  # tiger north
                   ]


filenames = ['CA_1.pickle','CA_2.pickle','CO_1.pickle','CO_2.pickle',
             'WA_1.pickle','WA_2.pickle', 'WA_3.pickle']


def make_plots():

    max_route = -1

    all_totals = []
    all_routes = []
    all_scores = []
    all_cd = []
    all_eg = []

    xlabel = 'Distance Constraint'
    ylabel = 'Elevation Constraint'
    xlim = ylim = (0.0,2.25)


    i = 0
    for i in range(len(filenames)):
    #for i in range(len(['CA_1.pickle'])):
        inname = './large_test/' + filenames[i]
        dnorm = target_dicts[i]['distance']
        enorm = target_dicts[i]['elevation_gain']
        with open(inname,'rb') as infile:
            totals, possible_routes, scores, tmap = pickle.load(infile)

            all_totals.extend(totals)
            all_routes.extend(possible_routes)
            all_scores.extend(scores)

            local_cd = []
            local_eg = []
            for j,nodes in enumerate(possible_routes):
                edges = tmap.edges_from_nodes(nodes)
                dists = tmap.reduce_edge_data('distances',edges=edges,function=None) # * 0.000621371
                alts  = tmap.reduce_edge_data('elevations',edges=edges,function=None) # * 3.28084

                ec = ((alts[1:] - alts[:-1]))
                eg = 1.0 * ec
                el = 1.0 * ec
                eg[eg<0] = 0.0
                el[el>0] = 0.0
                eg = np.cumsum(eg)
                el = np.cumsum(el)
                cd = np.cumsum(dists)[1:]

                local_cd.append(cd / dnorm)
                local_eg.append(eg / enorm)

            all_cd.extend(local_cd)
            all_eg.extend(local_eg)
            max_route = np.max([np.max([len(x) for x in possible_routes]), max_route])



    for index in range(3083,3084,10):

        fig, ax = plt.subplots()
        fs = 6
        fig.set_size_inches(fs,fs)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.fill_between( ax.get_xlim(), [0.9,0.9],[1.1,1.1], color = 'orange', alpha=0.4)
        ax.fill_betweenx( ax.get_ylim(), [0.9,0.9],[1.1,1.1], color = 'orange', alpha=0.4)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        avg_cd = np.zeros(len(all_cd))
        avg_eg = np.zeros(len(all_cd))
        maxlen_route = -1
        for j in range(len(all_cd)):

            cd = all_cd[j]
            eg = all_eg[j]

            if len(cd) == 0:
                #print('error?')
                avg_cd[j] =  -1
                avg_eg[j] = -1
                continue

            cdindex = np.min([index, len(cd)-1])
            egindex = np.min([index, len(eg)-1])

            ax.plot(cd[:cdindex],eg[:egindex], lw =1, ls = '-',color = 'black', alpha=0.2)

            avg_cd[j] = cd[cdindex]
            avg_eg[j] = eg[egindex]

            maxlen_route = np.max([maxlen_route,len(cd)])

        print("Average Fractional error: ",np.average(avg_cd[avg_cd>0]),np.average(avg_eg[avg_eg>0]))
        print("Median Fractional error: ", np.median(avg_cd[avg_cd>0]), np.median(avg_eg[avg_eg>0]))
        print("Max length of route ", maxlen_route)



        plt.tight_layout()
        #fig.savefig('./large_test/validation.png')
        fig.savefig('./large_test/figs/evolution_%0004i.png'%(index))

        plt.close()

        #print(np.shape(avg_cd), np.shape(avg_eg), np.shape(avg_cd[avg_cd>0]), np.shape(avg_eg[avg_eg>0]))
        #print(avg_cd)
        #print(avg_eg)
        kde = sns.jointplot(x=avg_cd[avg_cd>0],
                                y=avg_eg[avg_cd>0],
                                kind='kde')

        kde.fig.axes[0].set_xlim(xlim)
        kde.fig.axes[0].set_ylim(ylim)
        kde.fig.set_size_inches(fs,fs)
        kde.fig.axes[0].fill_between( ax.get_xlim(), [0.9,0.9],[1.1,1.1], color = 'orange', alpha=0.4)
        kde.fig.axes[0].fill_betweenx( ax.get_ylim(), [0.9,0.9],[1.1,1.1], color = 'orange', alpha=0.4)


        kde.set_axis_labels(xlabel,ylabel)
        plt.tight_layout()


        kde.fig.savefig('./large_test/kdes/kde_%0004i.png'%(index))

        plt.close()

        #snfig = kdeplot.get_figure()
        #snfig.savefig('./large_test/kde_test.png')
    return fig, kde

def main():
    """
    """

    for i in range(len(locations)):
        print("locations")
        results = make_trails(locations[i], target_dicts[i], start_coords[i])

        with open('./large_test/' + filenames[i],'wb') as f:
            pickle.dump(results, f, protocol=4)


    return

if __name__=="__main__":

    if sys.argv[1] == 'run':
        main()
    elif sys.argv[1] == 'plot':
        make_plots()
    else:
        print("DOING NOTHING")
