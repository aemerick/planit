import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as geopd
import networkx as nx
import os

from matplotlib import rc, cm
viridis = cm.get_cmap('viridis')
magma   = cm.get_cmap('magma')
plasma  = cm.get_cmap('plasma')

from pyproj import Proj, transform

import autotrail as AT
import process_gpx_data as gpx_process

m_in_mi = 1609.34
m_in_ft = 0.3048

import contextily as ctx

def plot_first(df, center, width=5000.0):

    center = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), center[0], center[1])

    if np.size(width) <= 1:
        width = (width,width)

    gdf_crs = df.copy().to_crs(epsg=3857)
    pc = gdf_crs.plot()
    pc.figure.set_size_inches(8,8)
    ctx.add_basemap(pc, source = ctx.tile_providers.OSM_A, zoom=12)

    ax = pc.figure.axes[0]

    ax.scatter(center[0],center[1],
                         color='black',marker='*',s=100)

    ax.set_xlim(center[0]-width[0],center[0]+width[0])
    ax.set_ylim(center[1]-width[1],center[1]+width[1])


    pc.figure.show()

    return

#
# plot the resulting routes
#
def plot_route(graph,
         node_order=None,
         weight='distance',
         zoom =True,
         center = None,
         nshow = -1,
         region = None):

    #pos = nx.spring_layout(graph, weight=weight, seed = 12345)


    pos = {ni : np.array([n['long'],n['lat']]) for ni,n in graph.nodes(data=True)}

    pc = nx.draw_networkx_nodes(graph, pos, node_size=20)
    ax = pc.figure.axes[0]

    pc.figure.set_size_inches(8,8)

    #edges = [(u,v)]
    #etraveled = [(u,v) for (u,v,d) in graph.edges(data=True) if d['traversed_count'] > 0]
    travelled_edges = graph.edges_from_nodes(node_order)

    etravelled    = [(u,v) if (u,v) in list(graph.edges) else (v,u) for (u,v) in travelled_edges]

    enottravelled = [(u,v) for (u,v) in graph.edges if (not ((u,v) in etravelled))]

    #print(etravelled)
    #print(enottravelled)

    colors = 'black'
    if not (node_order is None):
        ecolor_int = [node_order.index(e[0]) for e in etravelled]
        #colors = magma((np.array(ecolor_int)+5) / ((1.0*len(node_order)+5)))
        colors = ["C%i"%i for i in ecolor_int]

    colors = [np.array([0.0,0.0,0.0,1.0])]*len(node_order)
    if (not (node_order is None)):
        for ei,e in enumerate(etravelled):
            if not ((e in travelled_edges[:nshow]) or ((e[1],e[0]) in travelled_edges[:nshow])):
                colors[ei] = np.array([0.0]*3 + [1.0])
            else:
                dn = 5
                num = [i for i,x in enumerate(travelled_edges[:nshow]) if x==e or ((x[1]==e[0]) and (x[0]==e[1]))]
                num = num[-1]

                colors[ei] = magma((num+dn)/(1.0*len(node_order)+dn))


    temp = nx.draw_networkx_edges(graph, pos, edgelist=etravelled, width=5, edge_color=colors)
    temp = nx.draw_networkx_edges(graph, pos, edgelist=enottravelled, width=2, style='dashed')



    travelled_edges_counter = {x : 0 for x in np.unique(travelled_edges)}

    edge_labels = {}
    count = 0
    for (u,v,d) in graph.edges(data=True):

        if not((u,v) in edge_labels.keys()):
            edge_labels[(u,v)] = ''

    count = 1
    for (u,v) in travelled_edges:
        try:
            edge_labels[(u,v)] = edge_labels[(u,v)] + '%i '%(count)
        except:
            edge_labels[(v,u)] = edge_labels[(v,u)] + '%i '%(count)

        count = count + 1

    #try:
    #    edge_labels = { (u,v) : "%i"%(d['ShapeLen']) for (u,v,d) in graph.edges(data=True)}
    #except:
    #    edge_labels = { (u,v) : "%i"%(d['SHAPESTLength']) for (u,v,d) in graph.edges(data=True)}
    #temp = nx.draw_networkx_edge_labels(graph, pos,
    #                                    edge_labels=edge_labels, label_pos=0.5)

    #temp = nx.draw_networkx_labels(graph, pos, labels={n:n for n,data in graph.nodes(data=True)}, font_size=17,
    #                        font_color = 'black')


    chataqua = (-105.2795, 39.9972)
    ax.scatter(chataqua[0], chataqua[1], color = 'black', marker = '*', s = 200)

    start_point = graph.nodes[node_order[0]]
    ax.scatter(start_point['long'], start_point['lat'], color = 'black', marker = '*', s = 200)


    sep = 0.025
    if center is None:
        ax.set_ylim( chataqua[1] - sep, chataqua[1]+sep)
        ax.set_xlim( chataqua[0] - sep, chataqua[0]+sep)
    else:
        ax.scatter(center[0], center[1], color = 'red', marker = 'd', s = 300)

        select = [(n,graph.nodes[n]) for n in node_order]
        lats  = graph.reduce_node_data('lat', nodes = select)
        longs = graph.reduce_node_data('long', nodes = select)

        lat_min, lat_max   = np.min(lats), np.max(lats)
        long_min, long_max = np.min(longs), np.max(longs)


        lat_min, lat_max = (39.97109252073643,40.00096696488513)
        long_min, long_max = (-105.32, -105.27665459438845)
        fudge = 0.5 * (long_max-long_min)
        ax.set_xlim( long_min-fudge, long_max+fudge)
        fudge = 0.50 * (lat_max-lat_min)
        ax.set_ylim( lat_min-fudge, lat_max+fudge)


        #fudge = 0.02
        #ax.set_xlim(center[0]-fudge,center[0]+fudge)
        #ax.set_ylim(center[1]-fudge,center[1]+fudge)
    #print(pos)
    return pc,ax
