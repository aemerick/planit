"""

    Author  : Andrew Emerick
    e-mail  : aemerick11@gmail.com
    year    : 2020

    LICENSE :GPLv3

    Routines to make nice graphs and plots of TrailMap objects and
    routes.

    Auto-generated hiking and trail running routes in Boulder, CO
    based on known trail data and given user-specified contraints.
"""

import matplotlib.pyplot as plt
import geopandas as geopd
import gpxpy
import srtm
import shapely
import networkx as nx

from autotrail import TrailMap
