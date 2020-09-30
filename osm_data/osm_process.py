"""

    Author  : Andrew Emerick
    e-mail  : aemerick11@gmail.com
    year    : 2020

    LICENSE :GPLv3


    Processing OSM map data to the format and featues needed for
    routing.

    Auto-generated hiking and trail running routes anywhere where
    OSM has hiking data available. Optimized routes are
    based on given user-specified contraints.

"""

import numpy as np
import gpxpy
import shapely

from planit.autotrail.trailmap import TrailMap
import planit.autotrail.process_gpx_data as gpx_process

from planit.osm_data import osm_fetch


def process_ox(osx_graph, hiking_only = True):
    """
    Process the osx_graph object generated from doing something like:

       >  osx_graph = osmnx.graph_from_bbox()

    To filter out all non-hiking edges (and nodes), leaving only the hikeable
    trails and paths, defines hiking trails as those with 'highway' feature of
    'path','footway', or 'pedestrian'. Everything else is treated as road for
    now and is ignored.

    Returns the fully-processed graph as as TrailMap object, which includes
    elevation data for each edge and node.

    Parameters:
    ------------
    osx_graph    :   an osmnx graph object
    hiking_only  : (optional, bool) Does nothing for now. Default : True

    Returns:
    -------------
    tmap     : A fully processed and ready-to-run TrailMap object.
    """

    edges = []
    mandatory_features = ['geometry']
    all_hway_types = ['tertiary', 'residential', 'path', 'unclassified', 'secondary',
                      'service', 'footway', 'pedestrian', 'trunk', 'primary', 'trunk_link',
                      'track', 'motorway_link', 'motorway', 'cycleway', 'tertiary_link', 'primary_link',
                      'living_street', 'secondary_link', 'steps']

    highway_types = {}
    for k in ['tertiary','residential','secondary','service','primary','trunk','trunk_link',
              'tertiary_link','primary_link','living_street','secondary_link']:
        highway_types[k] = 'road'

    for k in ['path','footway','pedestrian']:
        highway_types[k] = k

    if hiking_only:
        # because why do a really long list comprehension
        edges = [(u,v,d) for (u,v,d) in osx_graph.edges(data=True) if all(f in d.keys() for f in mandatory_features) and any(f in d['highway'] for f in ['path','footway','track'])]
    else:
        edges = [(u,v,d) for (u,v,d) in osx_graph.edges(data=True) if all(f in d.keys() for f in mandatory_features)]

        #
        # set ones and zeros for each type of path (really road vs hiking)
        #
        for (u,v,d) in edges:
            for k in highway_types:
                if not (highway_types[k] in d.keys()) and k in d['highway']:
                    d[highway_types[k]] = 1
                else:
                    d[highway_types[k]] = 0

    #
    # Get all unique nodes and make the node tuple array
    #
    nodes = np.unique(np.concatenate([(u,v) for (u,v,d) in edges]).ravel())
    nodes = [(n, osx_graph._node[n]) for n in nodes]

    # set node properties
    for n,d in nodes:
        d['lat']  = d['y'] # for compatability with TrailMap
        d['long'] = d['x'] # for compatability with TrailMap
        d['elevation'] = gpx_process._elevation_data.get_elevation(d['lat'],d['long'])
        d['index'] = d['osmid']   # for compatability with TrailMap

    # where most of the work goes, setting edge properties:
    edges = compute_osm_edge_properties(edges, nodes)

    # make the map!
    tmap = TrailMap()
    tmap.graph['crs'] = osx_graph.graph['crs']
    tmap.add_edges_from(edges)
    tmap.add_nodes_from(nodes)

    return tmap



def compute_osm_edge_properties(edges, nodes):
    """
    Given a list of edges and nodes from an OSM graph dataset, compute the
    necessary edge and node properties needed for the TrailMap object to
    do the routing.

    The most useful part of this function is to compute elevation data
    along the edges and to define an orientation to the `geometry` feature
    (coordinate tracks) such that travelling from tail to head (where tail is
    the node with a lower index / osmid) gives you positive elevation_gain and
    negative loss, and head to tail reverses these two. Ensures `geometry`
    tracks are oriented in this fashion.

    Distances computed account for elevation.

    Parameters:
    ------------
    edges   :  list of edge tuples [(u,v,{}),....] for the graph
    nodes   :  list of nodes tuples [(u,{}),....] for the graph

    Returns:
    ------------
    edges   : the edited edge list with new dictionary items.
    """

    for i in range(len(edges)):
        tail, head, d = edges[i]

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
        taili = [j for j,(ni,nd) in enumerate(nodes) if nd['osmid'] == tail][0]
        headi = [j for j,(ni,nd) in enumerate(nodes) if nd['osmid'] == head][0]

        #print(tail,head, taili, headi)

        if len(d['geometry'].coords[0]) == 2:
            tail_coords = (nodes[taili][1]['long'], nodes[taili][1]['lat'])
            head_coords = (nodes[headi][1]['long'], nodes[headi][1]['lat'])
        elif len(d['geometry'].coords[0]) == 3:
            tail_coords = (nodes[taili][1]['long'], nodes[taili][1]['lat'], nodes[taili][1]['elevation'])
            head_coords = (nodes[headi][1]['long'], nodes[headi][1]['lat'], nodes[headi][1]['elevation'])
        else:
            raise RuntimeError


        # now, check and see if the geometry needs to be flipped:
        # compute distances of tail node to each end of the segment.
        # this MAY not work the best if the segment is a closed loop (or
        # of similar shape...)
        tail_to_left  = gpxpy.geo.distance(tail_coords[1],
                                           tail_coords[0],
                                           0.0,   # elevation doesn't matter here
                                           d['geometry'].coords[0][1], # long lat!!
                                           d['geometry'].coords[0][0],
                                           0.0)

        tail_to_right = gpxpy.geo.distance(tail_coords[1],
                                           tail_coords[0],
                                           0.0,   # elevation doesn't matter here
                                           d['geometry'].coords[-1][1],
                                           d['geometry'].coords[-1][0],
                                           0.0)

        flip_geometry = False
        if tail_to_right < tail_to_left: # flip the geometry
            flip_geometry = True
            d['geometry'] = shapely.geometry.LineString(d['geometry'].coords[::-1])

        # append node coords to line to make everything continuous
        new_line = (d['geometry']).append(head_coords)
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
        gpx = gpx_process.add_elevations(gpx, smooth=True)
        gpx_segment = gpx.tracks[0].segments[0]

        # point-point distances and elevations
        #
        # NOTE: Elevation gain and elevation loss requires defining a direction
        #       to travel on the trail. By convention the default direction
        #       will be from the tail -> head, where the node number of tail <
        #       the node number of head. So elevation_gain becomes
        #       elevation_loss when travelling from head to tail!!
        #

        distances   = gpx_process.gpx_distances(gpx_segment.points)
        elevations  = np.array([x.elevation for x in gpx_segment.points])
        dz          = (elevations[1:] - elevations[:-1])  # change in elevations
        grade       = dz / distances * 100.0            # percent grade!
        grade[np.abs(distances) < 0.1] = 0.0            # prevent arbitary large grades for short segs with errors


        d['geometry']         = shapely.geometry.LineString([(x.longitude,x.latitude,x.elevation) for x in gpx_segment.points])
        d['distance']         = np.sum(distances)
        d['elevation_gain']   = np.sum(dz[dz>0])            # see note above!
        d['elevation_loss']   = np.abs(np.sum( dz[dz<0] ))  # store as pos val
        d['elevation_change'] = d['elevation_gain'] + d['elevation_loss']
        d['min_grade']        = np.min(grade)
        d['max_grade']        = np.max(grade)
        d['average_grade']    = np.average(grade, weights = distances) # weighted avg!!

        #
        # Average min and max grade give a BETTER estimate of the conceptual 'how steep is this trail'
        # with average min meaning the typical steepest downhill (not the steepest possible downhill)
        # and average max meaning the typical steepest uphill. Reason why its not absolute min/max of
        # each is becase there could be short sections (e.g. a switchback corner) that is VERY steep.
        # Want to generally be ok with super short steep sections and constrain more on the 'typical'
        # grade of the route. If there are no descents, min grade is the average grade of everything
        # less than the average (I know......) and max is average of everything above max
        #
        # This can give a better idea of "runnable"
        #
        if np.sum(distances[grade>0]) > 0:
            d['average_max_grade']    = np.average(grade[grade>0], weights = distances[grade>0]) # weighted avg!!
        elif np.sum(distances[grade>d['average_grade']]) > 0:
            d['average_max_grade']    = np.average(grade[grade>d['average_grade']], weights=distances[grade>d['average_grade']])
        else:
            d['average_max_grade'] = d['average_grade']

        if np.sum(distances[grade<0]) > 0:
            d['average_min_grade']    = np.average(grade[grade<0], weights = distances[grade<0]) # w
        elif np.sum(distances[grade<d['average_grade']]) > 0:
            d['average_min_grade']    = np.average(grade[grade<d['average_grade']],weights=distances[grade<d['average_grade']])
        else:
            d['average_min_grade'] = d['average_grade']

        d['min_altitude']     = np.min(elevations)
        d['max_altitude']     = np.max(elevations)
        d['average_altitude'] = np.average(0.5*(elevations[1:]+elevations[:-1]),weights=distances)
        d['traversed_count']  = 0

        # apparenlty geopandas uses fiona to do writing to file
        # which DOESN"T support storing lists / np arrays into individual
        # cells. The below is a workaround (and a sin).. converting to a string
        #
        # MAKING ELEVATIONS SAME LENGTH AS DISTANCES!!
        #
        d['elevations']  = ','.join(["%6.2E"%(a) for a in 0.5*(elevations[1:]+elevations[:-1])])
        d['grades']      = ','.join(["%6.2E"%(a) for a in grade])
        d['distances']   = ','.join(["%6.2E"%(a) for a in distances])

    return edges


def osmnx_trailmap(center_point = None,
              ll = None,
              rr = None,
              query=None,
              dist=100000.0, # 100 km !
              save_to_file=True,
              allow_cache_load=True):
    """
    Wrapper around all of the get functions. kwargs match those in
    osm_fetch
    """

    ox_graph = osm_fetch.get_graph(center_point=center_point,ll=ll,rr=rr,
                                   query=query,dist=dist,save_to_file=save_to_file,
                                   allow_cache_load=allow_cache_load)
    tmap     = process_ox(ox_graph)

    return tmap


def test():
    """
    Test
    """

    center_point = ( 34.2070, -118.13684)
    tmap = osmnx_trailmap(center_point=center_point, dist=1.0E4)


    print("Success: ",type(tmap))
    return

if __name__ == '__main__':
    test()
