"""

    Author  : Andrew Emerick
    e-mail  : aemerick11@gmail.com
    year    : 2020

    LICENSE :GPLv3


    Processing the trail data from Boulder County into a form useable for
    routing. Identifies most trail intersections and converts all segments
    into interconnected graph of nodes and edges.

    Auto-generated hiking and trail running routes in Boulder, CO
    based on known trail data and given user-specified contraints.
"""

import numpy as np
import pandas as pd
import copy
import os

#
# Packages for handling geospatial data
#
import geopandas as geopd
import gpxpy
import srtm
import shapely
import networkx as nx

# FIX THIS

try:
    import cPickle as pickle
except:
    import pickle

#
# set up global lookup table for making elevation data
#
_elevation_data = srtm.get_data()


def combine_gpx(segments):
    """
    Generates a single shapely LineString objectf from a
    list of segments. Assumes that the passed segments are intended to
    form a continuous line.
    """

    coords = [list(x.coords) for x in segments]

    merged_line = shapely.geometry.LineString([x for sublist in coords for x in sublist])

    return merged_line

#    multi_line  = shapely.geometry.MultiLineString(segments)
#    merged_line = shapely.ops.linemerge(multi_line)

    # this may not always work if there are small gaps. Check for this
#    if isinstance(merged_line,shapely.geometry.MultiLineString):

        # force join the rest of the segments
#        coords = [list(x.coords) for x in merged_line]
        # Flatten the list of sublists and use it to make a new line
#        merged_line = shapely.geometry.LineString([x for sublist in coords for x in sublist])

#    return merged_line


# append for shapely linestring
def LineString_append(self, new_points, mode='append'):
    """
    Append a single coordinate or list of coordinates. Can
    either be a tuple or shapely Point (or list of either).
    Does error checking to make sure point does not already
    exist at the start/end of the line, but ONLY for the
    start (end) / end (start) points of the new_points (line).
    """

    if isinstance(new_points, tuple) or\
       isinstance(new_points, shapely.geometry.Point):
       new_points = [new_points]

    if mode == 'append':
        connect_index = 0
    elif mode == 'prepend':
        connect_index = -1

    connect_coord = new_points[connect_index]
    if hasattr(connect_coord, 'coords'):
        connect_coord = connect_coords.coords[0]

    if mode == 'append':
        if self.coords[-1] == connect_coord:
            new_points.pop(connect_index)
    elif mode == 'prepend':
        if self.coords[0] == connect_coord:
            new_points.pop(connect_index)

    if len(new_points) == 0:
        return self
    elif mode == 'append':
        return shapely.geometry.LineString( list(self.coords) + new_points)
    elif mode == 'prepend':
        return shapely.geometry.LineString( new_points + list(self.coords))

def LineString_prepend(self, new_points):
    return self.append(new_points, mode='prepend')

setattr(shapely.geometry.LineString, "append", LineString_append)
setattr(shapely.geometry.LineString, "prepend", LineString_prepend)


def add_elevations(gpx, smooth=False):
    """
    WARNING: ASSUMES ONLY A SINGLE TRACK AND SEGMENT IN GPX OBJECT

    Given a gpxpy gpx object, adds in elevations to all points (overwriting
    any existing values). Since the srtm elevation can sometimes return None
    for certain points (unknown why) does some error checking to make sure
    that points give valid elevation. Does interpolation / flattening to
    fill in the None-ed points.

    Fails if all are None. Really hope this doesn't happen.
    """

    # this gives `None` in some cases (argh!)
    _elevation_data.add_elevations(gpx, smooth=smooth) #, smooth=True)

    gpx_segment = gpx.tracks[0].segments[0]

    elevations = np.array([x.elevation for x in gpx_segment.points])

    is_none = (elevations == None)
    if all(is_none):
        print("WARNING: Found a segment with all failed elevations. Setting to hard-coded value for now")
        elevations = np.ones(len(elevations)) * 1624.0 # hard code to Boulder cause. why the hell not (BAD)
        is_none = elevations == None
        #raise RuntimeError

    if any(is_none):
        print("WARNING: Found a segment with some failed elevations. Trying to fix.")

        not_none = np.logical_not(is_none)

        # need to fix. just interpolate or copy
        if elevations[0] is None:
            # find first non-None
            first = np.where(not_none)[0][0]
            elevations[:first-1] = elevations[first]

        if elevations[-1] is None:
            last = np.where(not_none)[0][-1]
            elevations[last:] = elevations[last]

        # if that didn't fix them all, then we need to go in the
        # array and check for more
        is_none  = elevations == None
        if any(is_none):
            # more fixing needed
            where_not_none = np.where(np.logical_not(is_none))[0]
            for index in np.where(is_none)[0]:
                if not (elevations[index] is None):
                    continue

                prev_valid = where_not_none[ where_not_none < index][-1]
                next_valid  = where_not_none[ where_not_none > index][0]

                # TO DO: Change this to an interpolation using distance (point-by-point)
                # a      and not just an average
                elevations[index] = np.average([elevations[prev_valid],elevations[next_valid]])

    # copy back to gpx points
    for i in range(len(gpx_segment.points)):
        gpx_segment.points[i].elevation = elevations[i]

    if any(elevations == None):
        print("WARNING: Segment still has bad values for elevation after fixing")

    return gpx # just in case


def gpx_distances(points):
    """
    Compute distances between gpx points, either as list of
    GPXTrackPoints from gpxpy OR list of tuples
    """

    is_GPX = False
    if not isinstance(points[0],tuple):
        is_GPX=True

    if is_GPX:
        p = [(x.latitude,x.longitude,x.elevation) for x in points]
    else:
        p = points


    distance = np.zeros(len(p)-1)
    for i in np.arange(1, len(p)):
        distance[i-1] = gpxpy.geo.distance(p[i][0],p[i][1],p[i][2],
                                           p[i-1][0],p[i-1][1],p[i-1][2])


    return np.abs(distance)


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
            traverse_merge_list(nj, merge_list, to_merge)
        else:
            continue

    to_merge = np.sort(to_merge)

    return


#
# Need a nice, clean way to save and load the data!
#
def save_trail_df(outname, gdf, nodes=None, edges = None):
    """
    Save the (processed?) geopandas dataframe to file so we don't have
    to always re-generate in entirety later.

    outname cannot include a filetype (outfile = outname + '.extension')

    If nodes and edges are provided, these are saved alongside the
    dataframe in a pickle (for now... but should generalize...)
    """

    # save the geopandas as a geojson
    # outname = outname.split('.')[:-1] # toss extension
    gdf.to_file(outname + '.geojson', driver = 'GeoJSON')

    # if nodes are provided, pickle these using same outname
    if not (nodes is None):

        with open(outname+'_nodes.pickle','wb') as outfile:
            pickle.dump(nodes, outfile, protocol = 4)


    else:
        print("WARNING: Are you sure you don't want to save the nodes dictionary?")

    if not (edges is None):

        with open(outname+'_edges.pickle','wb') as outfile:
            pickle.dump(edges, outfile, protocol = 4)

    else:
        print("WARNING: Are you sure you don't want to save the edges dictionary?")

    return

def load_trail_df(inname, load_nodes=True, load_edges=True):
    """
    Load a trail dataframe. If `nodes` or `edges` are true, checks for
    the associated `.nodes` and `.edges` pickles and tries to load them.

    Returns geopandas dataframe. If load_nodes or load_edges are True (default)
    returns the geopandas dataframe, nodes list, and edges list together
    in a list (in that order).
    """

    result = [geopd.read_file(inname + '.geojson')]

    if load_nodes:
        nodefile = inname + '_nodes.pickle'

        if os.path.isfile(nodefile):
            with open(nodefile,'rb') as f:
                nodes = pickle.load(f)
        else:
            print("Cannot file nodefile to load: ", nodefile)
            nodes = None

        result.append(nodes)

    if load_edges:
        edgefile = inname + '_edges.pickle'

        if os.path.isfile(edgefile):
            with open(nodefile,'rb') as f:
                edges = pickle.load(f)
        else:
            print("Cannot file edgefile to load: ", edgefile)
            edges = None

        result.append(edges)

    if (not load_nodes) and (not load_edges):
        return result[0]
    else:
        return result

    return

def save_graph(outname, G):
    """
    Pickle up a networkx graph.
    """

    with open(outname + '_networkx.graph','wb') as outfile:
        pickle.dump(G, outfile, protocol = 4)

    return

def load_graph(inname):
    with open(inname + '_networkx.graph', 'rb') as f:
        return pickle.load(f)

def process_data(input_gdf,
                 outname = None,
                 # gpx_key = 'geometry', - should always be the case
                 threshold_distance = 3.0):
    """
    Given a trail dataset, process it!

    This function is designed to take in one of two trial datasets from the
    Boulder area loaded through geopandas, passed as `input_gdf`. It then
    filters that dataset with the goal of assining nodes at all trail
    start / end points and intersections. Currently this does this through
    brute force, and is a bit slow (5 minutes for the larger of the two
    datasets, comprised of 11,087 initial segments; ).

    This function:
        1) Assigns initial nodes to all trail segment start and end points

        2) Finds most trail junctions by filtering through all nodes to find
           nearby nodes (defined as those within `threshold distance`). These
           are considered connected junctions and the nodes are merged into
           a single new node and tail/head of edges are reassigned accordingly.

        -----) TODO: Check for instances of trail junctions where a node A is in the
                 middle of a segment (1). In this case, split 1 into new
                 segments joined at A.

        -----) Long term TODO: check for nodes that can be joined by short distances
               on road that are not in trail data.

        3) Computes edge quantities we WANT for the networkx graph solve
           and add these and only specified original features to Graph
           edges.

    In the end, this returns the processed dataset, the generated nodes,
    edges, and a networkx graph object constructed from these.

    Parameters:
    ------------

    threshold_distance  :  (float) distance between two nodes to merge (in m!)


    """

    #
    # deal with a local copy for now
    #
    _gdf = input_gdf.copy()


    small_dataset_type = False # assume we're on a larger dataset
    if 'Shapelen' in _gdf.columns:
        small_dataset_type = True # for small dataset columns
        columns_keep = ['OBJECTID','FEATURE_NAME','SURFACE_TYPE',
                        'PED','BIKE','HORSE','DOG','geometry']
    else:
        columns_keep = ['OBJECTID','TRAILNAME','PEDESTRIAN',
                        'BIKE','HORSE','OHV','TRAILTYPE',
                        'SURTYPE','DOGS','GlobalID','BATTrailID',
                        'geometry']

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
    #
    # Step 3: Brute force to find distances between nodes.
    #         If nodes are < some value separated then JOIN
    #
    #

    lat  = [n['lat'] for n in nodes]
    long = [n['long'] for n in nodes]
    elev = [_elevation_data.get_elevation(x,y) for (x,y) in zip(lat,long)]
    for i,val in enumerate(elev):
        nodes[i]['elevation'] = val


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
    to_append = []
    to_drop   = []

    mask = np.zeros(len(merge_list), dtype=bool)
    edge_relabel = {i:i for i in range(numnodes)} # use to relabel nodes in edges list later

    new_node_index = numnodes
    for ni in range(numnodes):

        if (len(merge_list[ni]) == 0) or mask[ni]: # empty or already completed
            continue

        # now we have one!!!
        # we merge by
        #
        #  1)  making new dict that has average lat, long of
        #      nodes to merge
        #
        #  2) re-labelling edges that touch these nodes with the
        #     new node index (take to be the min of nodes being merged)


        # gathers list of nodes that are linked together in common
        # with the current node and all nodes that may be chained
        # off of a linked node that is not directly linked with this one
        # to group ALL of them together.
        #
        # nodes to merge are in `to_merge`
        to_merge = [ni]
        traverse_merge_list(ni, merge_list, to_merge)

        mask[to_merge] = True # flag these to not repeat merge again

        nodes_select = [nodes[nj] for nj in to_merge]

        new_node  = {'index' : new_node_index,
                     'elevation' : np.average([x['elevation'] for x in nodes_select]),
                     'lat'   : np.average([x['lat'] for x in nodes_select]),
                     'long'  : np.average([x['long'] for x in nodes_select]),
                     'edges' : [x['edges'] for x in nodes_select] }

        # need to relabel edges !!
        for nj in to_merge:
            edge_relabel[nj] = new_node['index']

            nodes[nj]['index'] = -999 # flag for deletion

        # add to append list
        to_append.append(new_node)
        # add to drop list
        to_drop.extend(to_merge)

        new_node_index += 1


    # print('append: ', to_append)
    print("Modifying Nodes. Creating %i new nodes from merging %i together"%(len(to_append),len(to_drop)))

    nodes.extend(to_append)

    nodes = [n for n in nodes if n['index'] >= 0]

    print("New node list length (%i) from previous (%i)"%(len(nodes),numnodes))

    # might be nice to do one more step to make node indexes
    # continuous. skip for now.

    # assign new tail head nodes to all edges... and done!!
    for i in range(len(edges)):
        edges[i] = (edge_relabel[edges[i][0]], edge_relabel[edges[i][1]])

    #
    # Last step
    # Recompute all distances, elevation gains, elevation losses
    # min grade, max grade, average grade for consistency
    #

    # add columns to dataframe
    compute_columns = ['distance','elevation_gain','elevation_loss', 'elevation_change',
                       'min_grade','max_grade','average_grade',
                       'min_altitude','max_altitude','average_altitude','traversed_count']

    ncol = len(_gdf.columns)
    zeros = np.zeros(len(_gdf))
    for k in compute_columns:
        try:
            _gdf.insert(5, k, zeros) # insert zeroed and at end
        except ValueError:
            continue # already exists


    all_grades     = [None]*len(_gdf)
    all_elevations = [None]*len(_gdf)
    all_distances  = [None]*len(_gdf)

    for i in range(len(edges)):
        #
        # recompute distances there MUST be a better way to do this
        # than making all these objects
        #

        #
        # make a gpx segment from the coordinates
        #
        tail,head = edges[i]

        #
        # by convention, lets make it such that the tail of every segment
        # starts at the node with the lower index (tail < head!)
        # tail and head are the node IDs, NOT their number in the node list
        if tail > head:
            val = head*1
            head = tail*1
            tail = val*1

        #
        # taili and headi are the list indexes NOT the node ID's
        #
        taili = [j for j,n in enumerate(nodes) if n['index'] == tail][0]
        headi = [j for j,n in enumerate(nodes) if n['index'] == head][0]

        #print(tail,head, taili, headi)

        tail_coords = (nodes[taili]['long'], nodes[taili]['lat'])
        head_coords = (nodes[headi]['long'], nodes[headi]['lat'])

        # now, check and see if the geometry needs to be flipped:
        # compute distances of tail node to each end of the segment.
        # this MAY not work the best if the segment is a closed loop (or
        # of similar shape...)
        tail_to_left  = gpxpy.geo.distance(tail_coords[1],
                                           tail_coords[0],
                                           0.0,   # elevation doesn't matter here
                                           _gdf['geometry'][i].coords[0][1],
                                           _gdf['geometry'][i].coords[0][0],
                                           0.0)

        tail_to_right = gpxpy.geo.distance(tail_coords[1],
                                           tail_coords[0],
                                           0.0,   # elevation doesn't matter here
                                           _gdf['geometry'][i].coords[-1][1],
                                           _gdf['geometry'][i].coords[-1][0],
                                           0.0)

        flip_geometry = False
        if tail_to_right < tail_to_left: # flip the geometry
            flip_geometry = True
            _gdf.at[i,'geometry'] = shapely.geometry.LineString( _gdf['geometry'][i].coords[::-1])

        # append node coords to line to make everything continuous
        new_line = _gdf['geometry'][i].append(head_coords)
        new_line = new_line.prepend(tail_coords)

        #
        # Generate a GPX track object from this data to (easily) add in
        # elevations.
        #
        gpx = gpxpy.gpx.GPX()
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)

        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)

        gpx_points  = [gpxpy.gpx.GPXTrackPoint(x[1],x[0]) for x in new_line.coords]
        gpx_segment.points.extend(gpx_points)

        # add in elevation data
        gpx = add_elevations(gpx, smooth=True)
        gpx_segment = gpx.tracks[0].segments[0]

        # point-point distances and elevations
        #
        # NOTE: Elevation gain and elevation loss requires defining a direction
        #       to travel on the trail. By convention the default direction
        #       will be from the tail -> head, where the node number of tail <
        #       the node number of head. So elevation_gain becomes
        #       elevation_loss when travelling from head to tail!!
        #

        distances   = gpx_distances(gpx_segment.points)
        elevations  = np.array([x.elevation for x in gpx_segment.points])
        dz          = (elevations[1:] - elevations[:-1])  # change in elevations
        grade       = dz / distances * 100.0            # percent grade!
        grade[np.abs(distances) < 0.1] = 0.0            # prevent arbitary large grades for short segs with errors


        # save with elevations
        _gdf.at[i,'geometry']         = shapely.geometry.LineString([(x.longitude,x.latitude,x.elevation) for x in gpx_segment.points])
        _gdf.at[i,'distance']         = np.sum(distances)
        _gdf.at[i,'elevation_gain']   = np.sum(dz[dz>0])            # see note above!
        _gdf.at[i,'elevation_loss']   = np.abs(np.sum( dz[dz<0] ))  # store as pos val
        _gdf.at[i,'elevation_change'] = _gdf.iloc[i]['elevation_gain'] + _gdf.iloc[i]['elevation_loss']
        _gdf.at[i,'min_grade']        = np.min(grade)
        _gdf.at[i,'max_grad']         = np.max(grade)
        _gdf.at[i,'average_grade']    = np.average(grade, weights = distances) # weighted avg!!
        _gdf.at[i,'min_altitude']     = np.min(elevations)
        _gdf.at[i,'max_altitude']     = np.max(elevations)
        _gdf.at[i,'average_altitude'] = np.average(0.5*(elevations[1:]+elevations[:-1]),weights=distances)
        _gdf.at[i,'traversed_count']  = 0

        # apparenlty geopandas uses fiona to do writing to file
        # which DOESN"T support storing lists / np arrays into individual
        # cells. The below is a workaround (and a sin).. converting to a string
        #
        # MAKING ELEVATIONS SAME LENGTH AS DISTANCES!!
        #
        all_elevations[i]  = ','.join(["%6.2E"%(a) for a in 0.5*(elevations[1:]+elevations[:-1])])
        all_grades[i]      = ','.join(["%6.2E"%(a) for a in grade])
        all_distances[i]   = ','.join(["%6.2E"%(a) for a in distances])

    _gdf.insert(5, 'elevations', all_elevations)
    _gdf.insert(5, 'grades', all_grades)
    _gdf.insert(5, 'distances', all_distances)

    # keep these on Graph edges. Its why we made them in the first place
    columns_keep.extend(compute_columns + ['elevations','distances','grades'])

    # make sure the crs is copied over
    _gdf['geometry'].crs = _gdf.crs

    # From the processed data, generate the list of node tuples / dictionaries
    # and edge tuple / dictionaries needed by networkx's Graph class to
    # make a Graph
    trail_nodes = [(n['index'], {k : n[k] for k in ['lat','long','edges','index','elevation']}) for n in nodes]
    trail_edges = [ (e[0],e[1], _gdf.iloc[i][columns_keep]) for i,e in enumerate(edges)]

    if not (outname is None):
        # saves the geopandas dataframe and pickles the trail node
        # and trail edge objects
        save_trail_df(outname, _gdf, trail_nodes, trail_edges)

    return _gdf, trail_nodes, trail_edges


def make_trail_map(segmented_gdf, trail_nodes, trail_edges,
                   outname = None):
    """
    Given a fully segmented geopandas dataframe (e.g. each row corresponds to
    a SINGLE line segment) and information on the nodes and edge of the dataframe,
    generates a TrailMap object / graph. This REQUIRES information of the nodes
    and edges that is not currenlty easily processed from the dataframe itself.
    It would be much more convenient to generate this information from the dataframe
    without relying on having the pickled node and edge files OR having run
    `process_data` on the raw datasets first. But for now. This is fine.

    Parameters
    -----------
    segmented_df   :  (geopandas dataframe) Trail data, where each row
                      corresponds to a single segment of trail. Currently this
                      is just used to get the coordinate data, and `trail_nodes`
                      and `trail_edges` are really what are used to generate the
                      graph. In the future these two will be derived from the
                      dataframe.
    trail_nodes    :  list of trail nodes connecting together the segments.
                      list is of the form : [(n,{}), (n2,{}),...] where
                      n and n2 are node indexes / IDs and {} is the
                      dictionary of properties for each node.
    trail_edges    :  list of trail edges. List is of the form:
                      [ (u,v, {}), (u2, v2, {}), ....] where u,v and u2,v2 are the
                      node tail / heads for each segment and {} the dictionary
                      of properties for the segment
    outname        :  (Optional, string) file to pickle the TrailMap object to

    Returns:
    ---------
    G              : TrailMap graph (derived class of networkx's Graph)
    """
    from autotrail.autotrail.autotrail import TrailMap

    G = TrailMap()
    G.graph['crs'] = segmented_gdf.crs # coordinate system
    # list of tuples [(index, {})....]
    G.add_nodes_from(trail_nodes)
    # edges
    G.add_edges_from(trail_edges)

    if not (outname is None):
        save_graph(outname, G)

    return G
