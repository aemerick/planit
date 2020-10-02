"""
    Author  : Andrew Emerick
    e-mail  : aemerick11@gmail.com
    year    : 2020

    LICENSE :GPLv3


    Fetch OSM data using osmnx package.

    Auto-generated hiking and trail running routes anywhere where
    OSM has hiking data available. Optimized routes are
    based on given user-specified contraints.
"""

import numpy as np
import osmnx
import os

try:
    import cPickle as pickle
except:
    import pickle

#

def get_graph(center_point = None,
              ll = None,
              rr = None,
              query=None,
              dist=40233.6, # 100 km !
              save_to_file=True,
              allow_cache_load=True
              ):
    """
    A convenience wrapper around OSMNX's (already convenient) APIs to
    pre-set parameters that are better for this use case. Treats
    graph_from_point queries as bounding boxes.

    If center is passed, computes the bounding box of width of the diameter
    of the circle nad uses bounding box computation.

    This may take some time (5 - 60s) depending on the size of the region.

    Parameters:
    -----------
    center_point  : (optional) (lat,long) coordinate of center of map
    ll            : (optional) lower left corner (lat,long) of bounding box
    rr            : (optional) upper right corner (lat,long) of bounding box
    query         : (optional) search query for doing graph_from_place. This
    dist          : (optional) radius of circle to draw
    save_to_file  : (optional) Cache graph using lat long coordinats of bbox
                        by pickling the graph object.
    allow_cache_load : (optional) If bbox coordinate perfectly match a
                          saved pickle on disk, allow to load this instead

    Returns:
    -----------
    ox_graph      : OSMNX Graph object of the selected region

    """
    if (center_point is None) and (ll is None) and (rr is None):
        print("Must choose center coordinates or bounding box!")
        raise ValueError
    elif (ll is None) and (rr is None):
        # bad distance metric.. but this is approximate anyway
        earth_c = 40.075E6 # km circumference
        sep     = dist / earth_c * 360.0
        ll = (center_point[0] - 0.5*sep, center_point[1] - 0.5*sep)
        rr = (center_point[0] + 0.5*sep, center_point[1] + 0.5*sep)

    elif (center_point is None):
        center_point = (0.5 * (ll[0]+rr[0]), 0.5 * (ll[1]+rr[1]))


    north,south,east,west = rr[0],ll[0],rr[1],ll[1]

    call_api = True
    if allow_cache_load:
        inname = os.getcwd() + "/cache/%4.5f_%4.5f_%4.5f_%4.5f_osmnx_graph.pickle"%(north,south,east,west)
        print("Trying to find file: ", inname)
        if os.path.isfile(inname):
            with open(inname,'rb') as infile:
                ox_graph = pickle.load(infile)

            call_api = False
        else:
            print("cannot find file: ", inname)

    if call_api:
        ox_graph = osmnx.graph_from_bbox(north,south,east,west,
                              retain_all=True, truncate_by_edge=False,
                              clean_periphery=True)

        ox_graph.center_point = center_point
        ox_graph.ll = ll
        ox_graph.rr = rr
        ox_graph.query = query
        ox_graph.dist = dist

        if save_to_file:
        # pickle!!!!
            outname = os.getcwd() + "/cache/%4.5f_%4.5f_%4.5f_%4.5f_osmnx_graph.pickle"%(north,south,east,west)

            with open(outname,'wb') as outfile:
                pickle.dump(ox_graph, outfile, protocol = 4)



    return ox_graph


def test():
    """
    Test to make sure this works.
    """

    center_point = ( 34.2070, -118.13684)

    ox_graph = get_graph(center_point=center_point, dist = 1.0E4)

if __name__ == "__main__":

    test()
