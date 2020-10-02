"""

    Author  : Andrew Emerick
    e-mail  : aemerick11@gmail.com
    year    : 2020

    LICENSE :GPLv3

    Auto-generated hiking and trail running routes in Boulder, CO
    based on known trail data and given user-specified contraints.


    TODO: 1) move this to do list somewhere else
          2) remove routes that are identical from the results.
          3) penalize grades by gradient (ha)
          4)
"""

import numpy as np
import copy
import networkx as nx

import random
import shapely
import gpxpy

import json

# FIX THIS
from planit.autotrail import process_gpx_data as gpx_process
#import autotrail.autotrail.process_gpx_data as gpx_process

random.seed(12345)


m_to_ft = 3.28084
m_to_mi = 0.000621371

# note to self that I'm not really using multidigraph the most general way
# assuming ALL edges are bi-diretional and there is only one path node-to-node
_IDIR = 0

class TrailMap(nx.MultiDiGraph):
    """
    TrailMap class to handle trail Graphs and a
    traversing algorithm.
    """

    def __init__(self, name = '', debug = False, *args, **kwargs):

        nx.MultiDiGraph.__init__(self, *args, **kwargs)
        self.name = name

        self.debug = debug # debugging print statements


        self.edge_attributes = ['distance','elevation_gain', 'elevation_loss', 'elevation_change',
                                'min_grade','max_grade','average_grade',
                                'average_min_grade', 'average_max_grade',
                                'min_altitude','max_altitude','average_altitude',
                                'traversed_count', 'in_another_route']
        self.ensure_edge_attributes()

        #
        # default factors set to zero to turn off
        # should make setter / getter functions for this
        # to set properly.
        #   0 = not used
        #   > 0 gives relative importance
        #
        self._weight_factors = {}
        self._default_weight_factors = {'distance'         : 1,
                                        'elevation_gain'   : 1,
                                        'elevation_loss'   : 1,      # off
                                        'average_grade' : 0,
                                        'average_min_grade'        : 0,           # off
                                        'average_max_grade'        : 0,           # off
                                        'traversed_count'  : 100,    # very on
                                        'in_another_route' : 5}    # medium on

        # dictionaries to hold min-max scalings for weights
        # and functions to apply scalings
        self._scalings = None
        self._scale_var = None

        self._assign_default_weights()

        return

    def multi_find_route(self, start_node, target_values,
                               end_node=None,
                               n_routes = 5,
                               iterations = 10,                    # number of times to iterate !
                               target_methods=None,
                               primary_weight = 'distance',
                               reinitialize = True,                # reset 'traversed' counter each iteration
                               reset_used_counter = False,  # reset used (binary flag) each iteration
                               n_cpus=1):
        """
        Loops over algorithm multiple times to find multiple routes.
        Scores the results of these routes and returns the top
        `n_routes` (set to None, or inf to return all).

        Parameters:
        ------------

        start_node      : (int) node index corresponding to start point
        target_values   : (dict) Dictionary of target features and the target values.
        end_node        : (Optional, int) End node index. If not provided, uses
                          `start_node`. Default : None
        n_routes        : (Optional, int) Number of routes to return. Default : 5
        iterations      : (Optional, int) Number of iterations. If iterations <= n_routes,
                           sets to 2*n_routes. Default : 10
        target_methods  : (Optional, dict) Dictionary of what functions to use to
                           evaluate targets (e.g. np.sum for `distance`) along route.
                           Uses default methods if not provided. Default : None
        primary_weight  : (Optional, str) the primary feature to constrain. Not tested
                          for anything other than `distance`. Default : `distance`
        reinitialize    : (Optional, bool) Reset `traversed` counter each iteration. Default : True
        reset_used_counter : (Optional, bool) Reset `in_another_route` counter each iteration.
                             Default : True
        n_cpus          : (Optional, int) NOT YET IMPLEMENTED. Wishful thinking to parallelize
                          this with multiple threads / cpus. Does noting. Default : 1


        Returns:
        ---------
        all_totals  :  List of dictionaries of total quantities for each route
        all_routes  :  List of routes, defined as an ordered list of connected nodes
        all_errors  :  The scores for each route.
        """

        if iterations < n_routes:
            iterations = n_routes*2


        all_totals = [None] * iterations
        all_routes = [None] * iterations
        for niter in range(iterations):
            totals, routes = self.find_route(start_node, target_values,
                                             target_methods=target_methods,
                                             end_node=end_node,
                                             primary_weight=primary_weight,
                                             reinitialize=reinitialize)

            all_totals[niter], all_routes[niter] = totals, routes


        # score, sort, and return  - do all for now
        # for now, score on min fractional error
        num_routes = len(all_routes)
        fractional_error = {}
        for k in all_totals[0].keys():

            # treat these as limits / ranges not targets to hit for now
            if k in ['average_grade','average_max_grade','average_min_grade','max_grade','min_grade']:
                continue

            vals                = np.array([all_totals[i][k] for i in range(num_routes)])
            fractional_error[k] = np.abs(vals - target_values[k]) / target_values[k]

        # now slice it the other way
        #    better to do average error or max error?
        average_error = np.zeros(num_routes)
        for i in range(num_routes):
            average_error[i] = np.average([ fractional_error[k][i] for k in fractional_error.keys()])

        #
        # for now, return the best 3
        #
        sorted_index = np.argsort(average_error)

        return [all_totals[x] for x in sorted_index[:n_routes]], [all_routes[x] for x in sorted_index[:n_routes]], average_error[:n_routes]

    def find_route(self, start_node,
                         target_values,
                         target_methods = None,
                         end_node=None,
                         primary_weight = 'distance',
                         reinitialize=True,
                         reset_used_counter = False,
                         epsilon=0.25):
        """
        The core piece of Plan-It

        A smart-dumb algorithm to try and traverse a graph to find the best
        possible path that fits certain constraints.

        Parameters
        -----------
        start_node          : (int) start node (TODO: allow for choosing by label)
        target_values       : (dict) Dictionary of target values for each edge
                              feature. (e.g. target_values = {'distance' : 10} will
                              optimize to target a total distance of this value)

        target_methods      : (dict) Optional dictionary to decide how to evaluate
                               the target value across edges, with keys corresponding to
                               the keys in `target_values` and values are one-argument
                               reduction functions (e.g. np.sum, np.average).
                               By default, this uses functions defined in
                               `default_totals_methods` in code to handle targets
                               that are in `target_values` but not supplied in
                               `target_method`. Default : None

        end_node            : (Optional, int) end node. If 'None' uses start_node.
                              Default : None

       primary_weight       : (Optional, string) Key of primary edge attribute to emphasize
                              in weighting. Algorithm uses an aggregated `weight` attribute
                              to do actual weighting, but does some sanity checking using
                              `primary_weight` on its own to ensure that this target
                              is given preference (Default : distance).

        reinitialize        : (Optional, bool) Resets 'traversed_count' counters
                              in graph (if present). Default : True

        reset_used_counter  : (Optional, bool) If present, resets the `in_another_route`
                              counter meant to penalize segments that may be in an already
                              defined route. Default : True

        epsilon  : (Optional, flot) Parameter for the `get_next_node` algorithm
                   to adjust search method.

        Returns:
        --------------

        totals         : Dictionary of route properties
        possible_route : Ordered list of node IDs route travels along
        """
        default_target_methods  = {'distance' : np.sum,
                                   'average_max_grade'      : np.max,
                                   'average_min_grade' : np.min,
                                   'average_grade'  : self._max_abs,
                                   'elevation_gain' : np.sum, 'elevation_loss' : np.sum,
                                   'traversed_count' : np.sum}

        #
        # set up totals methods dictionary using input and supply default
        # if not overridden.
        #

        if target_methods is None:
            target_methods = {}

        # this will actually house the target methods
        totals_methods = {}
        for k in target_values.keys():
            totals_methods[k] = target_methods[k] if k in target_methods.keys() else default_target_methods[k]


        #
        # do some initial error checking to make sure that things CAN work
        #
        if start_node != end_node:
            if not (self.is_route_feasible(start_node, end_node, target_values, totals_methods)):
                self._print("Route not feasible. Please try different input")
                return None, None

        self.scale_edge_attributes()         # needed to do weighting properly
        self._assign_weights(target_values)   # assigns factors to easily do weighting based on desired constraints


        # AE: To Do - some way to check if target values are tuples (min,max) or single values!
        #             and then incorporate this into the model optimization (min / max)....
        #             should I just target the max value? instead of midpoint?

        # try out a single path
        keep_looping = True
        max_count    = 100
        count        = 0
        current_node = start_node

        # do pre-filtering here using views
        # nx.classes.filters.hide_edges([(2,5)])
        # subG = nx.subgraph_view(G, filter_edge = f)
        #
        #

        #
        # for now - empty dict we can use to copy multiple times if need be
        #
        empty_totals = {k:0 for k in totals_methods.keys()}

        totals = [copy.deepcopy(empty_totals)]

        possible_routes = [[start_node]]
        iroute = 0             # making above iterable in case we want to generate
                               # multiple routes in this funciton later and serve
                               # up different options

        if reinitialize and reset_used_counter:

            for e in self.edges(data=True):
                e[2]['traversed_count'] = 0
                e[2]['in_another_route'] = 0

        elif reinitialize:
            # reset some things
            for e in self.edges(data=True):
                e[2]['traversed_count'] = 0


            self.recompute_edge_weights(target_values=target_values)


        remaining = {k:0 for k in totals_methods.keys()} # dict to get remainders to target
        while (keep_looping):

            subG = self # placeholder to do prefiltering later !!!

            # need to do checking here to ensure distance is one of the targets (or soemwhere)
            # but here for now
            remaining[primary_weight] = target_values[primary_weight] - totals[iroute][primary_weight]

            #
            # get a list of all possible paths within a desired distance:
            #

            # can do better sucess / failure here and try once with target values
            # then try a second time without target values, with a error message
            # saying constraints not satisfied.
            next_node = self.get_intermediate_node(current_node,
                                                   remaining['distance'],
                                                   target_values=target_values,
                                                   G=subG, epsilon=epsilon)
            if next_node < 0:
                self._dprint("Next node not found!")
                # if epsilon fails I could also just pick a next node at random?
                # break

            if (current_node != start_node) or (next_node < 0):
                # make sure that we can still get home within a reasonable
                # distance
                shortest_path_home     = nx.shortest_path(subG, current_node, end_node, weight='weight')
                shortest_edges_home    = self.edges_from_nodes(shortest_path_home)
                shortest_primary_home  = self.reduce_edge_data(primary_weight,edges=shortest_edges_home)

                #
                # would potentialy be v cool to iterate here and check once with weight
                # and once with distance to see if using latter produces better match
                # at the expense of just repeating route. Can I edit the keys in graph view?
                # no... but maybe I can generate a new fake weight to do something like
                # this. or pass flag to recompute weights to ignore some things

                if shortest_primary_home > remaining[primary_weight]:
                    self._dprint("Finding shortest route to get home: ", shortest_path_home, end_node)
                    next_node  = end_node
                    next_path  = shortest_path_home
                    next_edges = shortest_edges_home
                elif (next_node < 0):
                    # likely running out of room
                    # pick one of the nodes along the shortest weighted path home
                    inext      = np.random.randint(0, len(shortest_path_home))
                    next_node  = shortest_path_home[inext]
                    next_path  = shortest_path_home[:inext+1] # AJE: bug here?
                    next_edges = self.edges_from_nodes(next_path)
                    self._dprint("Picking route on way to home %i %i"%(inext,next_node),next_path)
                else:
                    next_path  = nx.shortest_path(subG, current_node, next_node, weight='weight')
                    next_edges = self.edges_from_nodes(next_path)

            else:
                next_path  = nx.shortest_path(subG, current_node, next_node, weight='weight')
                next_edges = self.edges_from_nodes(next_path)


            for tail,head in next_edges:
                self._adj[tail][head][_IDIR]['traversed_count'] += 1
                self._adj[tail][head][_IDIR]['in_another_route'] = 1

            # add path
            self._dprint("Possible and next: ", possible_routes[iroute], next_path)

            possible_routes[iroute].extend(next_path[1:])
            # increment totals
            for k in (totals[iroute]).keys():
                newval = self.reduce_edge_data(k, edges=next_edges, function=totals_methods[k])
                totals[iroute][k] = totals_methods[k]( [totals[iroute][k], newval])

            # recompute weights:
            self.recompute_edge_weights(target_values=target_values) # probably need kwargs

            # use edges_to_hide = nx.classes.filters.hide_edges(edges)
            # to filter out based on min / max grade
            current_node = next_node
            count = count + 1
            if current_node == end_node:
                self._dprint("We found a successful route! Well... got back home at least ...")
                keep_looping = False

            elif count >= max_count:
                self._print("Reached maximum iterations")
                keep_looping = False


        if len(possible_routes[iroute]) <= 1:
            self._print("NO POSSIBLE ROUTE FOUND. Route stays fixed at start node.")


        return totals[iroute], possible_routes[iroute]

    def is_route_feasible(self, start_node, end_node, target_values, target_method):
        """
        Perform a simple sanity check to see if the route is viable. Uses
        nx.shortest_path to determine if nodes even connect, returning False if
        they do not. If they do, prints warnings if the shortest path does
        not meet the contraints, but only returns False if it cannot meet the
        distance constraint.

        Parameters
        -----------
        start_node : (int)   start point
        end_node   : (int)   end point
        target_values : (dict) dictionary of targets and desired values
        target_method : (dict) dictionary of how to evaluate targets

        Returns:
        -----------
        feasible  : (bool) True if route is possible AND within distance.
                    (even if other constrains may not be satisfied). False if
                    not possible or fails distance constraint.
        """

        if start_node == end_node:
            if len(self.nodes[start_node]['edges']) > 0:
                return True
            else:
                self._print("You seem to have picked a start point that does not connect to anything")
                return False

        try:
            route = nx.shortest_path(self, start_node, end_node)
        except nx.NetworkXNoPath:
            self._print("Start and end points do not connect on a known trail: ", start_node, end_node)
            return False


        # if route IS feasible, see if we fit within the constraints

        route_properties = self.route_properties(route, verbose=False)

        for k in target_values.keys():
            val = target_method[k](route_properties[k])
            if val > target_values[k]:
                self._print("WARNING: Route may not be feasible within constraint on ", k)
                self._print("Desired " + k + " is %f, while shortest path yields %f"%(target_values[k],val))

                if k == 'distance':
                    # actually fail on this
                    self._print("Route not possible within desired distance. Please try again with different parameters")
                    return False

        return True

    def route_properties(self, nodes=None, edges = None,
                               verbose=True, header=True, units = 'english'):
        """
        Given a route (defined as sequence of nodes or edges) computes
        summary statistics for that route. If verbose is ON prints out
        those statistics.

        Parameters:
        --------------

        nodes   : (Optional, list) This or `edges` must be provided. Ordered list
                  of node edges to compute along. Default : None
        edges   : (Optional, list) This or `nodes` must be provided. Ordered list
                  of edge tuples (u,v) defining route. Default : None
        verbose : (Optional, bool) Print stats to terminal. Default : True
        header  : (Optional, bool) If `verbose`, print header for the stats. Default : True)
        units   : (Optional, str) Units system to use. `english` (mi,ft) or `metric` (km,m)
                  Default : 'english'

        Returns:
        --------------
        totals  : Dictionary of route properties
        """

        if (nodes is None) and (edges is None):
            self._print("Must supply either nodes list or edge list for route")
            raise ValueError

        if edges is None:
            edges = self.edges_from_nodes(nodes)

        # this function should take in a list of nodes for a possible route
        # and returns the values of some quantity ALONG that route
        elevations = self.reduce_edge_data('elevations', edges=edges, function=None)
        distances = self.reduce_edge_data('distance', edges=edges, function=None)

        repeated = 0.0
        for i, e in enumerate(edges):
            if ( (e[0],e[1]) in edges[:i]) or ((e[0],e[1]) in edges[i+1:]):
                repeated += distances[i]

        totals = { 'distance' : np.sum(distances),
                   'elevation_gain' : self.reduce_edge_data('elevation_gain', edges=edges, function=np.sum),
                   'elevation_loss' : self.reduce_edge_data('elevation_loss', edges=edges, function=np.sum),
                   'average_min_grade' : self.reduce_edge_data('average_min_grade', edges=edges, function=np.min),
                   'average_max_grade' : self.reduce_edge_data('average_max_grade', edges=edges, function=np.max),
                   'average_grade' : self.reduce_edge_data('average_grade', edges=edges, function= self._max_abs),
                   'max_altitude' : np.max(elevations),
                   'min_altitude' : np.min(elevations)}
        totals['repeated_percent'] = repeated / totals['distance'] * 100.0

        if units == 'english':
            du = '(mi)'
            eu = '(ft)'
            totals['distance'] = totals['distance'] * m_to_mi
            for k in ['elevation_gain','elevation_loss','max_altitude','min_altitude']:
                totals[k] = totals[k] * m_to_ft

        else:
            du = '(km)'
            eu = '(m)'
            totals['distance'] = totals['distance'] / 1000.0

        if verbose and header:
            print("%13s %13s %13s %13s %13s %13s %13s %13s %13s"%("Distance "+du,
                                                                  "Elev. + "+eu, "Elev. - "+eu,
                                                                  "Min Elev. "+eu, "Max Elev. "+eu,
                                                                  "Min Grade (%)", "Max Grade (%)",
                                                                  "Avg Grade (%)", "Repeated (%)"))

        if verbose:
            print("%13.2f %13i %13i %13i %13i %13.2f %13.2f %13.2f %13.2f"%(totals['distance'],
                                                                     totals['elevation_gain'], totals['elevation_loss'],
                                                                     totals['min_altitude'],totals['max_altitude'],
                                                                     totals['average_min_grade'],totals['average_max_grade'],
                                                                     totals['average_grade'],
                                                                     totals['repeated_percent']))

        return totals

    def get_intermediate_node(self, current_node, target_distance,
                                    G=None,
                                    epsilon=0.1, shift = 0.1,
                                    weight='distance',
                                    target_values = {},
                                    max_iterations = 100):
        """
        Search for a node to jump to next in the algorithm given knowledge of
        the ultimate target distance for the route, and the current node.

        Parameters
        -----------
        current_node     :  (int) node index to start from
        target_distance  :  (float) the TOTAL distance desired in full route
        G                :  (optional, TrailMap instance) if provided,
                            uses this instance of TrailMap to perform search. This
                            exists to allow passing sub-graph views of self.
                            If None, uses self. Default : None
        epsilon          :  (Optional, float) factor of full route to target for next jump
                            position. Default : 0.25
        shift            :  (Optional, float) delta around epsilon used to set
                            annulus to search within for next point. Default : 0.1
        weight           :  (optional, string) the edge property to use for
                            weighting. Default : 'distance'
        target_values    :  (optional, dict) dictionary of target features and values
                            to (if provided) make better informed routing decisions.
        max_iterations   :  (optional, int) Maximum number of iterations within loop to find a new node.
                            Default : 100

        Returns:
        ----------
        next_node        : (int) Node index of next target node
        """

        if G is None:
            G = self

        next_node = None

        failed = False

        all_next_nodes = []
        next_node_weights = [] # to help choosing least worst if we have to
        iteration_count = -1
        error_code = ''
        while (next_node is None) and (iteration_count < max_iterations):
            iteration_count += 1

            # This should ensure that point is actually reachable
            all_possible_points = nx.single_source_dijkstra(G, current_node,
                                                            weight=weight,
                                                            cutoff=(epsilon+shift)*target_distance)

            if len(all_possible_points[0]) == 1:
                if epsilon > 1.0:
                    self._print("WARNING1: Failed to find an intermediate node. Epsilon maxing out")
                    failed    = True
                    next_node = None
                    break
                    #raise RuntimeError

                epsilon = epsilon + shift
                continue

            farthest_node = np.max(all_possible_points[0].values())
            possible_points = [k for (k,v) in all_possible_points[0].items() if v >= (epsilon-shift)*target_distance]

            if len(possible_points) == 0:
                if epsilon > 1.0:

                    self._print("WARNING2: Failed to find an intermediate node. Epsilon maxing out")
                    failed    = True
                    next_node = None
                    break
                    #raise RuntimeError

                #self._dprint("Increasing epsilon in loop %f"%(epsilon))
                epsilon = epsilon + shift
                continue

            # select one at random!!!

            next_node = None
            random.shuffle(possible_points) # in-place for some reason
            j = 0
            while (next_node is None) and (j < len(possible_points)) and (epsilon <= 1.0):
                next_node = possible_points[j] # choose this

                if len(target_values) > 0:
                    # if target values array exists, then we need to check and see if
                    # this path violates contraints. FOcusing on min max grade for now
                    #
                    value_checks = ['average_max_grade','average_min_grade', 'average_grade']
                    fdict = {'average_max_grade' : np.max, 'average_min_grade': np.min, 'average_grade' : self._max_abs}

                    weighted_path = nx.shortest_path(G, current_node, next_node, weight='weight')

                    for k in value_checks:
                        if k in target_values.keys():
#                            if len(weighted_path) == 0:
#                                self._print(k, current_node, next_node, target_values[k], weighted_path)
                            reduced       = self.reduce_edge_data(k, nodes = weighted_path, function=fdict[k])
                            if reduced > target_values[k]:
                                self._dprint("Next node failing on grade ", next_node, reduced, target_values[k])

                                all_next_nodes.append(next_node)
                                next_node_weights.append( self.reduce_edge_data('weight',nodes=weighted_path,function=np.sum))

                                next_node = None
                                error_code = "Target value fail"
                                break # break out of key for loop



                j = j + 1
            # end while loop
            if (epsilon > 1.0) and (next_node is None):
                self._print("WARNING3: Failed to find an intermediate node. Epsilon maxing out")
                failed = True
                next_node = None


        if (next_node is None) or (iteration_count > max_iterations):
            self._print(next_node, iteration_count, epsilon, error_code)
            if (error_code == "Target value fail") or (iteration_count > max_iterations):
                self._dprint("WARNING4: Unable to satisfy all criteria. Choosing least worst point")
                next_node = all_next_nodes[np.argmin(next_node_weights)]



        if next_node is None: # switch to -1 to throw proper error
            next_node = -1

        return next_node #, error_code

    def scale_edge_attributes(self, reset=False):
        """
        Scale edge attributes to do weighting properly. This does simple
        min-max scaling.

        Saves values as a new property in dictionary to preserve old values.
        This could be done better, but would require some more effort. This
        is simpler for now. New value is keyed as 'key_scaled'.

        Likely do not want to use traversed_count as rescaled (the way weighting
        is done now at least)
        """

        if (self._scalings is None):
            self._scalings = {}

        if (self._scale_var is None):
            self._scale_var = {}

        # compute!
        for k in self.edge_attributes:
            if (not (k in self._scalings.keys())) or reset:
                self._scalings[k] = {}
                self._scalings[k]['min_val'] = min_val = self.reduce_edge_data(k,function=np.min)
                self._scalings[k]['max_val'] = self.reduce_edge_data(k,function=np.max)
                self._scalings[k]['max_min'] = self._scalings[k]['max_val'] - self._scalings[k]['min_val']

        for k in self.edge_attributes: # need error checking for non quantiative values

            if (self._scalings[k]['max_min']) == 0.0: # likely no data here - don't rescale
                continue

            for e in self.edges(data=True):
                e[2][k+'_scaled'] = (e[2][k] - self._scalings[k]['min_val']) / self._scalings[k]['max_min']

        return

    def recompute_edge_weights(self, edges = None,
                                     target_values = {}):
        """
        Recompute edge weights based off of whether or not we
        consider various things. Turning on / off which featues to use
        is determined by the `_weight_factors` pre-factors (e.g. 0 is off).

        Parameters:
        -------------
        edges      : List of edge tuples WITH data dictionary [(u,v,d)...]
                     If not provided, computes for entire graph. Default : None
        target_values : Optional. Dictionary of target values to use more informed
                        weighting.

        """

        # AJE: Maybe weight grade by distance of segment to target? that way
        #      we can allow steep bits if needed

        # need a list of what we are considering
        # and how to treat it. Hard code for now

        # weightings:
        if edges is None:
            edges = self.edges(data=True)

        def _compute_grade_weight(key, edict, absval=False):
            """
            Helper function for doing grade scalings.
            Currently computing distance in scaled variables. But maybe better to compute
            scaling by relative 'error' (desired - edgeval) / desired ??

            weight function here is 1 - 0.5*(desired - val)^2, flattening to 0 when negative.
            gaurunteed to be 0 - 1

            """

#            val = 1 - (self._scale_var[key](key,target_values.get(key,edict[key])) - edict[key+'_scaled'])))**2
            if absval:
                val = 1 - (np.abs(target_values.get(key,edict[key])) - np.abs(edict[key]))**2
            else:
                val = 1 - (target_values.get(key,edict[key]) - edict[key])**2
            val = np.max([val,0.0])
            return self._weight_factors[key] * val

        for u,v,d in edges:

            max_tail_distance = np.max([self._adj[u][v][_IDIR]['distance_scaled'] for v in self._adj[u]])

            # apply weights for all '_scaled' properties using simple sum for now
            # need to control this better later. Handle traversed coutn separately

            d['weight'] = self._weight_factors['distance'] * d['distance_scaled']

            # direction of travel convention, gain is gain when u < v,
            # otherwise it needs to be flipped with loss.
            if u < v:
                d['weight'] += self._weight_factors['elevation_gain'] * d['elevation_gain_scaled']
                d['weight'] += self._weight_factors['elevation_loss'] * d['elevation_loss_scaled']
                d['weight'] += _compute_grade_weight('average_max_grade',d)
                d['weight'] += _compute_grade_weight('average_min_grade',d)
                d['weight'] += _compute_grade_weight('average_grade',d,absval=True)

            else:
                d['weight'] += self._weight_factors['elevation_loss'] * d['elevation_gain_scaled']
                d['weight'] += self._weight_factors['elevation_gain'] * d['elevation_loss_scaled']
                d['weight'] += _compute_grade_weight('average_grade',d,absval=True)
                if (self._weight_factors['average_max_grade'] > 0):
                    d['weight'] += _compute_grade_weight('average_max_grade',d) *\
                                   (self._weight_factors['average_min_grade']/self._weight_factors['average_max_grade'])
                if (self._weight_factors['average_min_grade'] > 0):
                    d['weight'] += _compute_grade_weight('average_min_grade',d) *\
                                   (self._weight_factors['average_max_grade']/self._weight_factors['average_min_grade'])

            #
            # Backtrack penalty
            #
            d['weight'] += self._weight_factors['traversed_count']*d['traversed_count']*max_tail_distance

            #
            # Meta penalty for use when planning multiple routes at once
            #
            d['weight'] += self._weight_factors['in_another_route']*d['in_another_route']*max_tail_distance


        return

    def get_route_coords(self, nodes = None, edges = None, elevation=True,
                         coords_only = False, in_json=False):
        """
        Wrapper around `reduce_edge_data` to get coordinates of the route for a
        variety of contexts. By default, returns coordinates in (lat, long)
        format (not long,lat)!!!

        Couple quirks here. By default, returns the coordinates as a Shapely
        LineSegment. If `in_json` is True, converts this to a JSON object of
        the route. If `coords_only` is provided, returns a list
        of just the coordinates.


        Parameters:
        ------------
        nodes     : (Optional, list) This or `edges` must be provided. List of
                    connected nodes defining the route.
        edges     : (Optional, list) This or `nodes` must be provided. List of
                    edge tuples (u,v) defining the route.
        elevation : (Optional, bool) Include elevation data. Default : True
        coords_only : (Optional, bool) Return list of coordinates instead of JSON.
                       Default : False
        in_json     : (Optional, bool) If `coords_only` is false, converts to JSON
                       if True. Default : False

        Returns:
        ------------
        route_line  : Route coordinates, as either a Shapely LineString,
                      JSON, or list depending on input settings.
        """
        if (nodes is None) and (edges is None):
            self._print("Must provide either nodes or edges")
            raise RuntimeError

        if edges is None:
            edges = self.edges_from_nodes(nodes)

        route_line = self.reduce_edge_data('geometry', edges=edges, function = gpx_process.combine_gpx)

        if coords_only:
            if elevation:
                route_line = [(c[1],c[0],c[2]) for c in route_line.coords]
            else:
                route_line = [(c[1],c[0]) for c in route_line.coords]

        if in_json:
            if elevation:
                return json.dumps(route_line)
            else:
                return json.dumps([(c[1],c[0]) for c in route_line])
        else:
            return route_line

    def nearest_node(self, long, lat, k = 1):
        """
        Get the nearest k nodes to the coordinate. Returns
        the index of the node in the node list and the id index of the node
        itself.

        TODO: generalize coordinates!

        Parameters
        -----------
        long : longitude ()
        lat  : latitude
        k    : (int) Optional. Number of neighbors to return. Default : 1

        Returns:
        ---------
        nearest_indexes  : array containing indexes in node list for nearest nodes
        nearest_node_ids : array containing node IDs from node properties
        """

        # ya I know i'm computing Euclidean disntance from a lat long
        # coordinate which is OBJECTIVELY bad. But this function is meant to
        # just pick closest point on a 2D map click where click is gonna be super
        # close to the point so this is kinda OK for now

        node_id   = self.reduce_node_data('index')
        node_lat  = self.reduce_node_data('lat')
        node_long = self.reduce_node_data('long')

        # compute distance
        # just need d^2 since we're sorting. Don't care about actual values
        dsqr = (node_lat - lat)**2 + (node_long-long)**2

        sort_indexes = np.argsort(dsqr)

        nearest_indexes  = sort_indexes[:k]
        nearest_node_ids = node_id[nearest_indexes]

        return nearest_indexes, nearest_node_ids

    def ensure_edge_attributes(self):
        """
        Ensure that edge attributes exist for ALL edges. Just to make sure
        nothing breaks.
        """

        for e in self.edges(data=True):
            for k in self.edge_attributes:
                if not (k in e[2].keys()):
                    e[2][k] = 0            # MAKE SURE THIS IS OK VALUE (maybe better to nan?)
        return


    def reduce_node_data(self, key, nodes=None, function = None):
        """
        A nice way to perform some combination function over
        the desired property for all provided nodes.

        Parameters
        ----------
        key      : Property keyword associated with the node. No
                   error checking is done to ensure valid. If key is 'index'
                   checks for 'index' key first, then uses the keyed node number.
        nodes    : (Optional) iterable nodes list to combine values
                    from. Default None -> use all nodes (self.nodes)
        function : (Optional) Function to apply to the data. This is a
                   function of one argument (the associated list of values).
                   If function is `None` returns list of data.
                   Default : None

        Returns:
        ---------
        value    : Reduced result from `function`
        """

        if nodes is None:
            nodes = self.nodes(data=True)
        elif not isinstance(nodes[0], tuple):
            nodes = [(n,self.nodes[n]) for n in nodes]

        if (key == 'index') and not (key in list(nodes)[0][1].keys()):
            values_array = np.array([n for n,d in nodes])
        else:
            values_array = np.array([d[key] for n,d in nodes])

        if function is None:
            return values_array
        else:
            value = function(values_array)

            return value


        return

    def reduce_edge_data(self, key, edges=None, nodes = None, function = np.sum):
        """
        A nice way to perform some combination function over
        the desired property for all provided edges.

        Parameters
        ----------
        key      : Property keyword associated with the edge. No
                   error checking is done to ensure valid.
        edges    : (Optional) iterable edge tuples [(u,v)...] to combine values
                    from. Default None -> use all edges (self.edges)
        function : (Optional) Function to apply to the data. This is a
                   function of one argument (the associated list of values).
                   If function is `None` returns list of data.
                   Default : np.sum

        Returns:
        ---------
        value    : Reduced result from `function`
        """

        if nodes is None:
            if edges is None:
                edges = self.edges
            elif isinstance(edges,tuple):
                edges = [edges]
        else:
            edges = self.edges_from_nodes(nodes)


        values_array = [self.get_edge_data(e[0],e[1])[key] for e in edges]

        # only because this is stored (ugh) as a giant string. combine into floats
        if key in ['elevations','distances','grades']:
            values_array = [float(x) for sublist in values_array for x in sublist.split(',')]

        try:
            values_array = np.array(values_array)
        except:
            pass

        if function is None:
            return values_array
        else:
            value = function(values_array)

            return value

    @staticmethod
    def edges_from_nodes(nodes):
        """
        Generates connected edges from a list of nodes under
        the assumption that the nodes are sorted to be connected. So... really
        just returns a list of overlapping tuples of adjacent values in an array.
        """
        #if not isinstance(nodes,list):
        #    raise ValueError

        return [( nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]

    def get_edge_data(self, u, v, default=None):
        """
        Overloading `get_edge_data` from the Graph class in order to properly
        handle directionality in some of the edge properties. In particular,
        checks for elevation_gain, elevation_loss, max_grade, and min_grade
        values. If any of these exist, it ensures they are correct. By convention,
        if u < v, elevation_gain and elevation_loss are correct, otherwise
        they are switched. If this happens, switch is done in a new copy.

        """

        result = super(TrailMap, self).get_edge_data(u,v,default)[_IDIR]


        if u < v:
            return result

        # everything below happens ONLY if we need to flip orientation

        result = copy.deepcopy(result)

        if 'elevation_gain' in result.keys():
            val = result['elevation_gain'] * 1.0
            result['elevation_loss'] = result['elevation_gain']*1.0
            result['elevation_gain'] = val*1.0

        if 'average_min_grade' in result.keys():
            val = result['average_min_grade']
            result['average_min_grade'] = -1.0 * result['average_max_grade']
            result['average_max_grade'] = -1.0 * val
            val = result['min_grade']
            result['min_grade'] = -1.0 * result['max_grade']
            result['max_grade'] = -1.0 * val

        if 'geometry' in result.keys():
            result['geometry'] = shapely.geometry.LineString(result['geometry'].coords[::-1])

        for k in ['elevations','distances','grades']:
            if k in result.keys():
                result[k] = ','.join(result[k].split(',')[::-1]) # dumb!

        return result

    def write_gpx_file(self, outname, nodes = None, edges = None, elevation=True):
        """
        Convert route (defined by connected nodes or edges) to a
        GPX .xml file containing lat and long coordinates with
        elevation.

        Parameters:
        outname       : (str) File name to write to
        nodes         : (Optional, list) This or `edges` must be provided. List
                         of nodes defining the route.
        edges         : (Optional, list) This or `nodes` must be provided. List
                        of edges (u,v) defining the route.
        elevation     : (Optional, bool) Include elevation in route. Default : True


        """
        route_line = self.get_route_coords(nodes=nodes,edges=edges, elevation=elevation)
        #
        # Take points from route_line (LineString object) and place into a
        # gpx track object
        #
        gpx = gpxpy.gpx.GPX()

        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)

        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)

        #
        # Linestings are being saved with (long,lat) coordinates but GPX
        # needs (lat,long). Careful!
        #
        if elevation:
            gpx_segment.points.extend( [gpxpy.gpx.GPXTrackPoint(x[1],x[0], elevation=x[2]) for x in route_line.coords])
        else:
            gpx_segment.points.extend( [gpxpy.gpx.GPXTrackPoint(x[1],x[0]) for x in route_line.coords])


        with open(outname, 'w') as f:
            f.write( gpx.to_xml())

        return


    def _assign_default_weights(self):
        """
        Maps default weight dictionary onto active weight
        factor dictionary. Used to initialize these values
        and reset these to defaults. Sets to zero if not in
        default dictionary.
        """

        for k in self.edge_attributes:
            if not (k in self._default_weight_factors.keys()):
                self._weight_factors[k] = 0.0
            else:
                self._weight_factors[k] = self._default_weight_factors[k]

        return

    def _assign_weights(self, target_values):
        """
        Assign weight values based on what is present in
        the target values dictionary. Turns off those not present.
        """

        self._assign_default_weights()

        for k in self._weight_factors.keys():
            if not (k in target_values.keys()):
                self._weight_factors[k] = 0.0

        # AJE:
        #    maybe set loop here to set values to negative or positive weights

        # for now... need to fix this...
        for k in ['traversed_count','in_another_route']:
            self._weight_factors[k] = self._default_weight_factors[k]

        return

    def _max_abs(self, var):
        """
        Helper function. Was a lambda but that can break pickling
        """
        return np.abs(np.max(var))

    def _print(self, msg, *args, **kwargs):
        """
        Print overload
        """
        print("TrailMap: ", msg, *args, **kwargs)
        return

    def _dprint(self, msg, *args, **kwargs):
        """
        Debug print
        """
        if not self.debug:
            return
        self._print(msg, *args, **kwargs)
        return


class SimpleGraph():
    """
    A soon-to-be defunct class originally made for the trail map graphs.
    Superceded by TrailMap below. Here still because it is used in generating
    the simple test graphs.
    """

    def __init__(self,node_paths):

        self._node_paths   = node_paths


        self._tails        = node_paths[:,0] #.tolist()
        self._heads        = node_paths[:,1] #.tolist()
        self._capacities   = node_paths[:,2] #.tolist() # should all be same!
        self._costs        = node_paths[:,3] #.tolist()

        self._nodes = np.unique(self._tails)


        # set other properties here
        # e.g. self._max_grade = XXX for each arc


        self.num_nodes = np.size(np.unique(self._tails))
        self.num_arcs  = len(self._tails)

        return

    def tail(self,n):
        return self._tails[n]

    def head(self,n):
        return self._heads[n]

    def capacity(self,n):
        return self._capacities[n]

    def cost(self,n):
        return self._costs[n]

    def get_node_paths(self):
        return self._node_paths

    def setup_networkx_graph(self):
        """
        Makes a networkx graph from this Graph object
        """

        self._G_nx = nx.Graph()

        return self._G_nx

def define_graph(graphnum=0):
    """
    Some hand-made example graphs for testing out the code. Select graph
    with integer `graphnum`. Available: 0, 1
    """

    if graphnum == 0:
        node_paths = np.array([
                          (0, 1, 1, 1),
                          (1, 0, 1, 1),
                          (1, 2, 1, 1),
                          (1, 6, 1, 2),
                          (2, 1, 1, 1),
                          (2, 3, 1, 3),
                          (2, 4, 1, 4),
                          (2, 5, 1, 2),
                          (3, 2, 1, 3),
                          (4, 2, 1, 4),
                          (4, 5, 1, 1),
                          (5, 2, 1, 2),
                          (5, 4, 1, 1),
                          (5, 6, 1, 5),
                          (6, 1, 1, 2),
                          (6, 5, 1, 5) ])
    elif graphnum==1:

        node_paths = np.array([
                          (0, 1, 1, 1),
                          (1, 0, 1, 1),
                          (1, 2, 1, 1),
                          (1, 6, 1, 2),
                          (2, 1, 1, 1),
                          (2, 3, 1, 3),
                          (2, 4, 1, 4),
                          (2, 5, 1, 2),
                          (3, 2, 1, 3),
                          (4, 2, 1, 4),
                          (4, 5, 1, 1),
                          (5, 2, 1, 2),
                          (5, 4, 1, 1),
                          (5, 6, 1, 5),
                          (6, 1, 1, 2),
                          (6, 5, 1, 5),
                          (6, 7, 1, 2),
                          (7, 6, 1, 2),
                          (6, 8, 1, 4),
                          (8, 6, 1, 4),
                          (8, 9, 1, 1),
                          (9, 8, 1, 1),
                          (9, 5, 1, 3),
                          (5, 9, 1, 3),
                          (7, 10, 1, 2),
                          (10, 7, 1, 2),
                          (10, 11, 1, 1),
                          (11, 10, 1, 1),
                          (10, 12, 1, 2),
                          (12,10,1,2),
                          (10,13,1,1),
                          (13,10,1,1),
                          (13,14,1,5),
                          (14,13,1,5) ])



    # make sure this is a sorted node path !!!
    graph = SimpleGraph(node_paths)

    return graph
