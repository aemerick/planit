"""

    Author  : Andrew Emerick
    e-mail  : aemerick11@gmail.com
    year    : 2020

    LICENSE :GPLv3

    Auto-generated hiking and trail running routes in Boulder, CO
    based on known trail data and given user-specified contraints.
"""

import numpy as np
import copy
import networkx as nx

import random

random.seed(12345)

class Graph():
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
    Some hand-made example graphs for testing out the mode. Select graph
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
    graph = Graph(node_paths)

    return graph


class TrailMap(nx.Graph):
    """
    TrailMap class to handle trail Graphs and a
    traversing algorithm.
    """

    def __init__(self, name = '', debug = False, *args, **kwargs):

        nx.Graph.__init__(self, *args, **kwargs)
        self.name = name

        self.debug = debug # debugging print statements


        self.edge_attributes = ['distance','elevation_gain', 'elevation_loss', 'elevation_change',
                                'min_grade','max_grade','average_grade',
                                'min_altitude','max_altitude','average_altitude',
                                'traversed_count', 'in_another_route']

        #
        # default factors set to zero to turn off
        # should make setter / getter functions for this
        # to set properly.
        #   0 = not used
        #   > 0 gives relative importance
        #
        self._weight_factors = {}
        self._default_weight_factors = {'distance' : 1,
                                        'elevation_gain' : 0,
                                        'elevation_loss' : 0,      # off
                                        'min_grade' : 0,           # off
                                        'max_grade' : 0,           # off
                                        'traversed_count' : 10,    # very on
                                        'in_another_route' : 5}    # medium on

        self._assign_default_weights()

        return

    def scale_edge_attributes(self):
        """
        Scale edge attributes to do weighting properly. This does simple
        min-max scaling.

        Saves values as a new property in dictionary to preserve old values.
        This could be done better, but would require some more effort. This
        is simpler for now. New value is keyed as 'key_scaled'.

        Likely do not want to use traversed_count as rescaled (the way weighting
        is done now at least)
        """

        for k in self.edge_attributes: # need error checking for non quantiative values

            min_val = self.reduce_edge_data(k, function=np.min)
            max_val = self.reduce_edge_data(k, function=np.max)
            max_min = max_val - min_val

            if (max_min) == 0.0: # likely no data here - don't rescale
                continue

            for tail in self._adj:
                for head in self._adj[tail]:
                    self._adj[tail][head][k+'_scaled'] = (self._adj[tail][head][k] - min_val) / max_min

        return

    def ensure_edge_attributes(self):
        """
        Ensure that edge attributes exist for ALL edges. Just to make sure
        nothing breaks.
        """

        for tail in self._adj:
            for head in self._adj[tail]:
                for k in self.edge_attributes:
                    if not (k in self._adj[tail][head].keys()):
                        self._adj[tail][head][k] = 0 # MAKE SURE THIS IS OK VALUE (maybe better to nan?)
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

        # for now... need to fix this...
        for k in ['traversed_count','in_another_route']:
            self._weight_factors[k] = self._default_weight_factors[k]

        return

    def recompute_edge_weights(self, edges=None):
        """
        Recompute edge weights based off of whether or not we
        consider various things.
        """

        # need a list of what we are considering
        # and how to treat it. Hard code for now

        # weightings:

        for tail in self._adj:
            max_tail_distance = np.max([self._adj[tail][h]['distance_scaled'] for h in self._adj[tail]])
            for head in self._adj[tail]:
                e = self._adj[tail][head]

                # apply weights for all '_scaled' properties using simple sum for now
                # need to control this better later. Handle traversed coutn separately
                self._adj[tail][head]['weight'] = np.sum([ self._weight_factors[k.replace('_scaled', '' )]*e[k] for k in e.keys() if ((k != 'traversed_count_scaled') and ('scaled' in k)) ])
                self._adj[tail][head]['weight'] += self._weight_factors['traversed_count']*e['traversed_count']*max_tail_distance ## increase by size of max_distance off of tail
                self._adj[tail][head]['weight'] += self._weight_factors['in_another_route']*max_tail_distance

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

        if (key == 'index') and not (key in nodes[0].keys()):
            values_array = np.array([n for n,d in nodes])
        else:
            values_array = np.array([d[key] for n,d in nodes])

        if function is None:
            return values_array
        else:
            value = function(values_array)

            return value


        return

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

    def reduce_edge_data(self, key, edges=None, function = np.sum):
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

        if edges is None:
            edges = self.edges
        elif isinstance(edges,tuple):
            edges = [edges]


        values_array = np.array([self.get_edge_data(u,v)[key] for (u,v) in edges])

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


    def multi_find_route(self, start_node, target_values,
                               n_routes = 5,
                               iterations = 10,                    # number of times to iterate !
                               target_methods=None, end_node=None,
                               primary_weight = 'distance',
                               reinitialize = True,                # reset 'traversed' counter each iteration
                               reset_used_counter = False,  # reset used (binary flag) each iteration
                               n_cpus=1):
        """
        Loops over algorithm multiple times to find multiple routes.
        Scores the results of these routes and returns the top
        `n_routes` (set to None, or inf to return all).

        iterations : (int,optional) Number of times to try the route finder.

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

            all_totals[niter], all_routes[niter] = totals[0], routes[0]


        # score, sort, and return  - do all for now
        # for now, score on min fractional error
        num_routes = len(all_routes)
        fractional_error = {}
        for k in all_totals[0].keys():

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
                         epsilon=0.25):
        """
        The core piece of autotrails

        A smart DUMB algorithm to try and traverse a graph to find the best
        possible path that fits certain constraints.

        Parameters
        -----------
        start_node     :   (int) start node (TODO: allow for choosing by label)
        target_values  :   (dict) Dictionary of target values for each edge
                           feature. (e.g. target_values = {'distance' : 10} will
                           optimize to target a total distance of this value)

        target_methods :   (dict) Optional dictionary to decide how to evaluate
                           the target value across edges, with keys corresponding to
                           the keys in `target_values` and values are one-argument
                           reduction functions (e.g. np.sum, np.average).
                           By default, this uses functions defined in
                           `default_totals_methods` in code to handle targets
                           that are in `target_values` but not supplied in
                           `target_method`. Default : None

        end_node      :   (Optional, int) end node. If 'None' uses start_node.
                          Default : None

        primary_weight : (Optional, string) Key of primary edge attribute to emphasize
                         in weighting. Algorithm uses an aggregated `weight` attribute
                         to do actual weighting, but does some sanity checking using
                         `primary_weight` on its own to ensure that this target
                         is given preference (Default : distance).

        reinitialize  :   (Optional, bool) Resets 'traversed_count' counters
                          in graph (if present). Default : True

        """

        self.scale_edge_attributes()         # needed to do weighting properly
        self.assign_weights(target_values)   # assigns factors to easily do weighting based on desired constraints


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
        default_target_methods  = {'distance' : np.sum,       'max_grade'      : np.max,
                                   'min_grade' : np.min,      'average_grade'  : np.average,
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
        # for now - empty dict we can use to copy multiple times if need be
        #
        empty_totals = {k:0 for k in totals_methods.keys()}

        totals = [copy.deepcopy(empty_totals)]

        possible_routes = [[start_node]]
        iroute = 0             # making above iterable in case we want to generate
                               # multiple routes in this funciton later and serve
                               # up different options

        if reinitialize:
            # reset some things
            for t in self._adj:
                for h in self._adj[t]:
                    self._adj[t][h]['traversed_count'] = 0

            self.recompute_edge_weights()


        remaining = {k:0 for k in totals_methods.keys()} # dict to get remainders to target
        while (keep_looping):

            subG = self # placeholder to do prefiltering later !!!

            # need to do checking here to ensure distance is one of the targets (or soemwhere)
            # but here for now
            remaining[primary_weight] = target_values[primary_weight] - totals[iroute][primary_weight]

            #
            # get a list of all possible paths within a desired distance:
            #
            next_node = self.get_intermediate_node(current_node,
                                                   remaining['distance'],
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
                shortest_primary_home = self.reduce_edge_data(primary_weight,edges=shortest_edges_home)

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
                else:
                    # likely running out of room
                    # pick one of the nodes along the shortest path home
                    inext      = np.random.randint(0, len(shortest_path_home))
                    next_node  = shortest_path_home[inext]
                    next_path  = shortest_path_home[:inext+1] # AJE: bug here?
                    next_edges = self.edges_from_nodes(next_path)
                    self._dprint("Picking route on way to home %i %i"%(inext,next_node),next_path)



            else:
                next_path  = nx.shortest_path(subG, current_node, next_node, weight='weight')
                next_edges = self.edges_from_nodes(next_path)


            for tail,head in next_edges:
                self._adj[tail][head]['traversed_count'] += 1
                self._adj[tail][head]['in_another_route'] = 1

            # add path
            self._dprint("Possible and next: ", possible_routes[iroute], next_path)

            possible_routes[iroute].extend(next_path[1:])
            # increment totals
            for k in (totals[iroute]).keys():
                newval = self.reduce_edge_data(k, edges=next_edges, function=totals_methods[k])
                totals[iroute][k] = totals_methods[k]( [totals[iroute][k], newval])

            # recompute weights:
            self.recompute_edge_weights() # probably need kwargs

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



        return totals[iroute], possible_routes[iroute]


    def get_intermediate_node(self, current_node, target_distance,
                                    epsilon=0.25, shift = 0.1, weight='distance',
                                    G=None):

        """
        Search for a node to jump to next in the algorithm given knowledge of
        the ultimate target distance for the route, and the current node.

        Parameters
        -----------
        current_node     :  (int) node index to start from
        target_distance  :  (float) the TOTAL distance desired in full route
        epsilon          :  (Optional, float) factor of full route to target for next jump
                            position. Default : 0.25
        shift            :  (Optional, float) delta around epsilon used to set
                            annulus to search within for next point. Default : 0.1
        weight           :  (optional, string) the edge property to use for
                            weighting. Default : 'distance'
        G                :  (optional, TrailMap instance) if provided,
                            uses this instance of TrailMap to perform search. This
                            exists to allow passing sub-graph views of self.
                            If None, uses self. Default : None

        Returns:
        ----------
        next_node        : (int) Node index of next target node
        """

        if G is None:
            G = self

        next_node = None
        while next_node is None:

            # This should ensure that point is actually reachable
            #print(G)
            #print(current_node, weight, epsilon, shift, target_distance)
            all_possible_points = nx.single_source_dijkstra(G, current_node,
                                                            weight=weight,
                                                            cutoff=(epsilon+shift)*target_distance)
            #self._dprint("get_intermediate_node: epsilon = %.3f"%(epsilon))

            if len(all_possible_points[0]) == 1:
                if epsilon > 1.0:
                    self._dprint("WARNING: Failed to find an intermediate node. Epsilon maxing out")
                    return -1
                    #raise RuntimeError

                #self._dprint("Increasing epsilon in loop %f"%(epsilon))
                epsilon = epsilon + shift
                continue

            farthest_node = np.max(all_possible_points[0].values())
            possible_points = [k for (k,v) in all_possible_points[0].items() if v >= (epsilon-shift)*target_distance]

            # select one at random!!!
            next_node = random.choice(possible_points)

        return next_node

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



def networkx_method():

    arcs = define_graph()._node_paths
    nodes = np.unique(arcs[:,0])

    edges = [(n[0],n[1], {'distance':n[3], 'weight':n[3]}) for n in arcs]

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return


def main():

    graph = define_graph()


if __name__=="__main__":

    main()
