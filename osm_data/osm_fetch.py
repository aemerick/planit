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

import math

# degrees to radians
def deg2rad(degrees):
    return math.pi*degrees/180.0
# radians to degrees
def rad2deg(radians):
    return 180.0*radians/math.pi

# Semi-axes of WGS-84 geoidal reference
WGS84_a = 6378137.0  # Major semiaxis [m]
WGS84_b = 6356752.3  # Minor semiaxis [m]


#
#
# ============================================================================
# Coordinate conversions below taken from stackoverflow
# https://stackoverflow.com/questions/238260/how-to-calculate-the-bounding-box-for-a-given-lat-lng-location
# =============================================================================
#
#
import math

# Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
def WGS84EarthRadius(lat):
    # http://en.wikipedia.org/wiki/Earth_radius
    An = WGS84_a*WGS84_a * math.cos(lat)
    Bn = WGS84_b*WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    return math.sqrt( (An*An + Bn*Bn)/(Ad*Ad + Bd*Bd) )

# Bounding box surrounding the point at given coordinates,
# assuming local approximation of Earth surface as a sphere
# of radius given by WGS84
def boundingBox(latitudeInDegrees, longitudeInDegrees, halfSideInKm):
    lat = deg2rad(latitudeInDegrees)
    lon = deg2rad(longitudeInDegrees)
    halfSide = 1000*halfSideInKm

    # Radius of Earth at given latitude
    radius = WGS84EarthRadius(lat)
    # Radius of the parallel at given latitude
    pradius = radius*math.cos(lat)

    latMin = lat - halfSide/radius
    latMax = lat + halfSide/radius
    lonMin = lon - halfSide/pradius
    lonMax = lon + halfSide/pradius

    return (rad2deg(latMin), rad2deg(lonMin), rad2deg(latMax), rad2deg(lonMax))

#
# ==============================================================================
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

        # bounding box coordinates with distance in km
        bbox = boundingBox(center_point[0], center_point[1], 0.5*dist/1000.0)
        ll   = (bbox[0],bbox[1])
        rr   = (bbox[2],bbox[3])

    elif (center_point is None):
        center_point = (0.5 * (ll[0]+rr[0]), 0.5 * (ll[1]+rr[1]))


    north,south,east,west = rr[0],ll[0],rr[1],ll[1]

    call_api = True
    if allow_cache_load:

        if not os.path.isdir(os.getcwd() + '/cache'):
            os.mkdir(os.getcwd() + '/cache')

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
                              clean_periphery=True, custom_filter='["highway"~"path|track"]')

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
