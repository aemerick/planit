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


def plot_trails(trailmap, edges = None,
                ll = None, rr = None):
    """
    Plot all trails in trailmap in the region selected.
    """

    fig,ax = plt.subplots()
    fig.set_size_inches(6,6)


    if not (edges is None):
        if (ll is None) and (rr is None):

            min_long, max_long = 9999, -10000
            min_lat, max_lat = 9999, -10000

            for (u,v) in edges:
                long = [c[0] for c in trailmap.edges[(u,v)]['geometry'].coords]
                lat  = [c[1] for c in trailmap.edges[(u,v)]['geometry'].coords]

                min_long = np.min([min_long,np.min(long)])
                max_long = np.max([max_long,np.max(long)])
                min_lat  = np.min([min_lat,np.min(lat)])
                max_lat  = np.max([max_lat,np.max(lat)])

            dlong = (max_long-min_long)
            dlat  = (max_lat-min_lat)

            if dlong < dlat:
                d = dlat - dlong
                min_long = min_long - 0.5*d
                max_long = max_long + 0.5*d
            else:
                d = dlong - dlat
                min_lat = min_lat - 0.5*d
                max_lat = max_lat + 0.5*d

            # now expand both by 25%
            dlong = (max_long-min_long)*0.25
            dlat  = (max_lat-min_lat)*0.25
            min_long = min_long - 0.5*dlong
            max_long = max_long + 0.5*dlong
            min_lat = min_lat - 0.5*dlat
            max_lat = max_lat + 0.5*dlat

            ll = (min_long,min_lat)
            rr = (max_long,max_lat)

    if ll is None:
        ll = (-np.inf,-np.inf)
        rr = (np.inf,np.inf)

    for (u,v,d) in trailmap.edges(data=True):

        long = [c[0] for c in d['geometry'].coords]
        lat  = [c[1] for c in d['geometry'].coords]

        if all(long<ll[0]) or all(long>rr[0]) or all(lat<ll[1]) or all(lat>rr[1]):
            continue

        ax.plot(long, lat, lw = 1, ls = '--', color = 'black')

    if not (edges is None):
        for (u,v) in edges:
            long = [c[0] for c in trailmap.edges[(u,v)]['geometry'].coords]
            lat  = [c[1] for c in trailmap.edges[(u,v)]['geometry'].coords]

            ax.plot(long,lat,lw=3,ls='-',color='C1')


    ax.set_xlim(ll[0],rr[0])
    ax.set_ylim(ll[1],rr[1])

    return fig, ax
