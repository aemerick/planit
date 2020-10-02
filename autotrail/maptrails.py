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
import numpy as np

from planit.autotrail import trailmap


from matplotlib import rc, cm
viridis = cm.get_cmap('viridis')
magma   = cm.get_cmap('magma')
plasma  = cm.get_cmap('plasma')
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm



def plot_trails(tmap,
                nodes = None, edges = None,
                ll = None, rr = None,
                fs = 6, linecolor='gradient',
                aspect=False, plot_nodes=False,
                save_each_iteration=False,
                show_profile = True, colormap = magma):
    """
    Plot all trails in trailmap in the region selected.
    """

    if (edges is None) and (not (nodes is None)):
        edges = tmap.edges_from_nodes(nodes)

    color = ['C%i'%(i) for i in range(9)]
    ci = 1

    if show_profile and (edges is None):
        show_profile = False

    if show_profile:
        fig, all_ax = plt.subplots(1,2)
        fig.set_size_inches(fs*2,fs)
    else:
        fig, all_ax = plt.subplots()
        fig.set_size_inches(fs,fs)



    if not (edges is None):
        if (ll is None) and (rr is None):

            min_long, max_long = 9999, -10000
            min_lat, max_lat = 9999, -10000

            for e in edges:
                u = e[0]
                v = e[1]
                long = [c[0] for c in tmap.edges[(u,v,trailmap._IDIR)]['geometry'].coords]
                lat  = [c[1] for c in tmap.edges[(u,v,trailmap._IDIR)]['geometry'].coords]

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

    if show_profile:
        ax = all_ax[0]
    else:
        ax = all_ax

    for (u,v,d) in tmap.edges(data=True):

        long = np.array([c[0] for c in d['geometry'].coords])
        lat  = np.array([c[1] for c in d['geometry'].coords])

        try:
            if all(long<ll[0]) or all(long>rr[0]) or all(lat<ll[1]) or all(lat>rr[1]):
                continue
        except:
            print(long)
            print(lat)
            print(ll, rr)

        ax.plot(long, lat, lw = 1, ls = '--', color = 'black')
    plt.tight_layout()
    ax.set_xlim(ll[0],rr[0])
    ax.set_ylim(ll[1],rr[1])

    if plot_nodes and (not (nodes is None)):
        node_lat = tmap.reduce_node_data('lat',function=None, nodes=nodes)
        node_long = tmap.reduce_node_data('long',function=None, nodes=nodes)
        ax.scatter(node_long, node_lat, s = 80, marker='o',color='black')

    if aspect:
        # keep aspect ratio by adjusting figure size. Normalize to longest side
        if (rr[1]-ll[1]) > (rr[0]-ll[0]):
            fsy = fs
            fsx = fs * (rr[0]-ll[0])/(rr[1]-ll[1])
        elif (rr[0]-ll[0]) > (rr[1]-ll[1]):
            fsx = fs
            fsy = fs * (rr[1]-ll[1])/(rr[0]-ll[0])
        else:
            fsx = fsy = fs

        fig.set_size_inches(fsx,fsy)

    icount = 0
    if not (edges is None):

        if linecolor == 'gradient':
            lencount = 0
            maxlen   = np.size(tmap.reduce_edge_data('distances',edges=edges,function=None))

        for e in edges:
            u,v = e[0], e[1]
            long = [c[0] for c in tmap.edges[(u,v,trailmap._IDIR)]['geometry'].coords]
            lat  = [c[1] for c in tmap.edges[(u,v,trailmap._IDIR)]['geometry'].coords]

            if u > v:
                long = long[::-1]
                lat = lat[::-1]

            if linecolor == 'gradient':
                points = np.array([long, lat]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                norm = plt.Normalize(0.0,maxlen)
                lc = LineCollection(segments, cmap='magma', norm=norm)
                # Set the values used for colormapping
                lc.set_array(np.arange(len(long)-1)+lencount)
                lc.set_linewidth(4)
                line = ax.add_collection(lc)
                lencount=lencount+len(long)-1

                if (save_each_iteration):
                    outdir = os.getcwd() + '/movie/'
                    if not (os.path.isdir(outdir)):
                        os.mkdir(outdir)

                    fig.savefig(outdir + 'image_%8i.png'%(icount))
                    icount = icount+1

                #fig.colorbar(line, ax=axs[0])
                #ax.plot(long,lat,lw=3,ls='-',color=colormap( np.arange(long) / 1.0*np.size(long)) )
            else:
                ax.plot(long,lat,lw=3,ls='-',color=color[ci])





    if show_profile:
        ax = all_ax[1]
        dists = tmap.reduce_edge_data('distances',edges=edges,function=None) * 0.000621371
        alts  = tmap.reduce_edge_data('elevations',edges=edges,function=None) * 3.28084

        if linecolor == 'gradient' and False:
            points = np.array([np.cumsum(dists),alts]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0.0,len(dists))
            lc = LineCollection(segments, cmap='magma', norm=norm)
            # Set the values used for colormapping
            lc.set_array(np.arange(len(dists))-1)
            lc.set_linewidth(4)
            line = ax.add_collection(lc)
        else:
            ax.plot(np.cumsum(dists), alts, color = color[ci], lw = 3, ls = '-')

        ax.set_xlabel('Distance (mi)')
        ax.set_ylabel('Elevation (ft)')




    return fig, all_ax
