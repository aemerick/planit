import numpy as np
import pandas as pd

#
# Packages for handling geospatial data
#
import geopandas as geopd
import gpxpy
import srtm
import shapely
import copy
import networkx as nx

# set up global lookup table
_elevation_data = srtm.get_data()


def process_geopandas():
    """
    Given a geopandas data frame process it
    """

    return


def load_gpx_data():
    """
    """


    return


def add_elevation(gpx):
    """
    Given a set of GPX data
    """

    _elevation_data.add_elevations(gpx)

    return


def traverse_merge_list(ni, merge_list, to_merge):
    """
    Recursive function traverse a list (`merge_list`) containing
    lists of indexes within the same list to ensure that ALL
    grouping is properly done. `ni` is the index of `merge_list`
    that we are checking now.

    E.g. if `ni = 2`, `to_merge` should be initialized to
    `to_merge = [2]` and if `merge_list` is:

    x = [[],[],[4,3],[2],[2,6],[],[4]]

    All of these nodes are interconnected, so this function will
    set `to_merge` to be:

    to_merge = [2,4,3,6]

    and then sort it, finalizing to [2,3,4,6]

    """
    for nj in merge_list[ni]:
        if not (nj in to_merge):
            to_merge.append(nj)
            traverse_merge_list(nj, to_merge, merge_list)
        else:
            continue

    to_merge = np.sort(to_merge)

    return

def process_data(input_gdf, threshold_distance = 1.0):
    """
    Given a gdf, process it!

    This is the VERY brute force way of doing this because

    1) I only have to ultimately do it once for the MVP of this project

    and

    2) the smart part of my brain is semi-fried and I'm just
    trying to get code down....
    which means

    3) way too many loops

    threshold_distance  :  distance between two nodes to merge (in m!)
    """
    _gdf = input_gdf.copy()


    small_dataset_type = False # assume we're on a larger dataset
    if 'OBJECTID' in _gdf.columns:
        small_dataset_type = True # for small dataset columns

    #
    # step one, break apart all multi-line segments into their own
    # rows. Append to the end and delete original.
    #
    numrows = len(_gdf)

    to_drop   = []
    to_append = []
    for i, row in _gdf.iterrows():
        if isinstance(row['geometry'], shapely.geometry.multilinestring.MultiLineString):

            for iseg, seg in enumerate(row['geometry']):
                dcopy               = copy.deepcopy(row)
                dcopy['geometry']   = seg

                # make sure unique ID and still refers back to original:
                if small_dataset_type:
                    dcopy['OBJECTID']   = row['OBJECTID'] + (0.1*iseg)
                else:
                    dcopy['GlobalID']   = row['GlobalID'] + '%i'%(iseg)
                    dcopy['BATTrailID'] = row['BATTrailID'] + '-%i'%(iseg)

                to_append.append(dcopy)

            # remove multiline instance
            to_drop.append(i)

    #print(to_drop, len(to_drop))
    _gdf = _gdf.drop(to_drop)
    _gdf = _gdf.append(to_append, ignore_index=True)

    #
    # Step 2: Get list of all nodes as all segment end points
    #

    # list of dictionaries containing node properties
    # this will need to be converted to list of [ (index, {}) ]
    numrows   = len(_gdf)
    nodes = [{'index':0 , 'lat':0,'long':0,'edges':[]} for _ in range(2*numrows)] # initially 2x number of edges
    edges = [() for _ in range(numrows)] # list of tail and head nodes (one for each row)

    ni = 0 # node index
    for i, row in _gdf.iterrows():
        # print(ni, i, row['geometry'].coords[0])
        coords = row['geometry'].coords

        nodes[ni]['long'], nodes[ni]['lat']     = coords[0][0],  coords[0][1]
        nodes[ni+1]['long'], nodes[ni+1]['lat'] = coords[-1][0], coords[-1][1]

        # edge indexes this node connects to - trivial for now
        nodes[ni]['edges']   = [i]
        nodes[ni+1]['edges'] = [i]

        nodes[ni]['index']   = ni
        nodes[ni+1]['index'] = ni+1 # saving this (trivial) to do merging properly

        edges[i] = (ni,ni+1)       # tail and head nodes for each edge

        ni += 2

    print("Computed %i nodes for %i edges"%(len(nodes),len(edges)))
    #
    # Step 3: Brute force to find distances between nodes.
    #         If nodes are < some value separated then JOIN
    #
    lat  = [n['lat'] for n in nodes]
    long = [n['long'] for n in nodes]
    elev = [elevation_data.get_elevation(x,y) for (x,y) in zip(lat,long)]

    numnodes    = len(nodes)
    merge_list  = [[] for _ in range(numnodes)] # for each node, list of nodes merging with
    #numnodes = 100

    for ni in range(numnodes):
        edgei = ni // 2

        for nj in range(ni+1, numnodes):
            d = gpxpy.geo.distance(lat[ni], long[ni], elev[ni],
                                   lat[nj], long[nj], elev[nj])


            # if d is within threshold AND
            # the nodes are not at the end of the same segment
            if d < threshold_distance and ( (ni,nj) != edges[edgei] ):
                merge_list[ni].append(nj)
                merge_list[nj].append(ni) # add to a merge list

    #
    # now we need to MERGE together all of the nodes
    # and re-assign the edges for them
    #
    ni = -1
    to_append = []
    to_drop   = []

    mask = np.zeros(len(merge_list), dtype=bool)

    edge_relabel = {i:i for i in range(nodes)} # use to relabel nodes in edges list later

    for mi in range(merge_list):
        ni = ni + 1     # current node !

        if len(m) == 0 or mask[mi]: # empty
            continue

        # now we have one!!!
        # we merge by
        #
        #  1)  making new dict that has average lat, long of
        #      nodes to merge
        #
        #  2) re-labelling edges that touch these nodes with the
        #     new node index (take to be the min of nodes being merged)

        # go through list of nodes to merge to double check consistency
        to_merge = [ni]

        # gathers list of nodes that are linked together in common
        # with the current node and all nodes that may be chained
        # off of a linked node that is not directly linked with this one
        # to group ALL of them together.
        #
        # nodes to merge are in `to_merge`
        traverse_merge_list(ni, merge_list, to_merge)

        mask[to_merge] = True # flag these to not repeat merge again

        new_node  = {'index' : np.min(to_merge),
                     'lat'   : np.average([x['lat'] for x in nodes[to_merge]]),
                     'long'  : np.average([x['long'] for x in nodes[to_merge]]),
                     'edges' : [x['edges'] for x in nodes[to_merge]] }

        # need to relabel edges !!
        for nj in to_merge:
            edge_relabel[nj] = new_node['index']

        # add to append list
        to_append.append(new_node)
        # add to drop list
        to_drop.append(to_merge)

    for i in len(edges):
        edges[i] = ()

    nodes.append(to_append)
    for ni in to_drop:
        nodes.pop(ni)



    # Last step
    # Recompute all distances, elevation gains, elevation losses
    # min grade, max grade, average grade for consistency
    #
    return _gdf, nodes, edges #, lat, long, merge_list
