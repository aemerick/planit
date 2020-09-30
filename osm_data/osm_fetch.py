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
import osmnx as ox


def get_graph(center_point = None,
              ll = None,
              rr = None,
              dist=100000.0 # 100 km !
              ):
    """
    A convenience wrapper around OSMNX's (already convenient) APIs to
    pre-set parameters that are better for this use case. Treats
    graph_from_point queries as bounding boxes.

    If center is passed, computes the bounding box of width of the diameter
    of the circle nad uses bounding box computation.

    Parameters:
    -----------
    center_point  : (optional) (lat,long) coordinate of center of map
    ll            : (optional) lower left corner (lat,long) of bounding box
    rr            : (optional) upper right corner (lat,long) of bounding box
    dist          : (optional) radius of circle to draw

    Returns:
    -----------
    ox_graph      : OSMNX Graph object of the selected region
    
    """
    if (center_point is None) and (ll is None) and (rr is None):
        print("Must choose center coordinates or bounding box!")
        raise ValueError
    elif (ll is None) and (rr is None):
        # bad distance metric.. but this is approximate anyway
        earth_c = 40.075E3 # km circumference
        sep     = dist / earth_c * 360.0
        ll = (center_point[0] - 0.5*sep, center_point[1] - 0.5*sep)
        rr = (center_point[1] + 0.5*sep, center_point[1] + 0.5*sep)

    elif (center_point is None):
        center_point = (0.5 * (ll[0]+rr[0]), 0.5 * (ll[1]+rr[1]))
    else:
        raise RuntimeError

    north,south,east,west = rr[1],ll[1],rr[0],ll[0]


    ox_graph = osmnx.graph_from_bbox(north,south,east,west,
                          retain_all=True, truncate_by_edge=False,
                          clean_periphery=True)


    return ox_graph
