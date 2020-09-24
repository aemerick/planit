try:
    # if installed from pip
    from ortools.graph import pywrapgraph
except:
    # from source (ortools/ortools sub-directory)
    from ortools.ortools.graph import pywrapgraph

import numpy as np
import copy
import networkx as nx

import min_cost_flow_test as mcf_test
import random

random.seed(12345)

#
# lets try and hack together a dumb algorithm to do this thing
#

def initialize_mincostflow(start_nodes, end_nodes, capacities, costs):
    """

    """

    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    min_cost_flow.SetDesiredCost(desired_cost);

    # Add each arc.
    for i in range(0, len(start_nodes)):
      min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                  capacities[i], unit_costs[i])

    # Add node supplies.

    for i in range(0, len(supplies)):
      min_cost_flow.SetNodeSupply(i, supplies[i])

    #if min_cost_flow.SolveWithCostAdjustment() == min_cost_flow.OPTIMAL:

    return min_cost_flow


class Graph():

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


    def setup_mincostflow(self, start, end, active_arcs=None):
        """
        Sets up a mincostflow instance to solve for minimum distance using
        this graph

        active arcs is a mask denoting the active and inactive arcs (ideally)
        if not, assume it is a list of the active arcs
        """

        if active_arcs is None:
            active_arc_mask = bool
        elif len(active_arcs) < self.num_arcs:
            # assume it is a list of indeces
            active_arc_mask = np.zeros(self.num_arcs, dtype=bool)
            active_arc_mask[active_arcs] = True
        else:
            active_arc_mask = np.array(active_arcs,dtype=bool)

        arc_paths, supply = mcf_test.set_nodes_and_supplies(self._node_paths[active_arc_mask], start, end)

        tails = arc_paths[:,0].to_list()
        heads = arc_paths[:,1].to_list()
        capacities = arc_paths[:,2].to_list()
        costs = arc_paths[:,3].to_list()

        mcf_model = initialize_mincostflow(tails,heads,capacities,costs)

        return mcf_model


class GraphPath():

    def __init__(self,parent_graph):

        self._parent_graph = parent_graph

        #
        # all arcs are active initially !
        #    active = available to be traversed
        #  inactive = invalid (basically does not exist)
        #
        self.arc_is_active = np.ones(self.parent_graph.num_arcs)

        # total cost of the path so far
        self.total_cost = 0

        # arcs traverse along this path.
        # a list of arc indeces for where we have gone so far
        self._path_arc_indeces = []

        self.current_head = None
        self.current_tail = None
        self.current_arc  = None

        return

    @property
    def parent_graph(self):
        return self._parent_graph

    def recompute_compute_total_cost(self):
        """

        """

        return

    def traverse_arc(self, arcindex):
        """
        Add arc to the path
        """

        self._path_arc_indeces.append(arcindex)

        self._set_current_state()

        return

    def _update_current_state(self):
        """
        Update variables to current state.
        """

        if len(self._path_arc_indeces) == 0:
            self.current_head = self.current_tail = self.current_arc = None
            return

        arcindex          = self._path_arc_indeces[-1]
        self.current_head = self.parent_graph.head(arcindex)
        self.current_tail = self.parent_graph.tail(arcindex)
        self.current_arc  = arcindex
        self.total_cost   = self.total_cost + self.parent_graph.cost(arcindex)

        return

    def reverse_path(self, r=1):
        """
        Backtrack the last r nodes and remove the reversed paths from active arcs
        """


        self.arc_is_active[self._path_arc_indeces[r:]] = False

        self._path_arc_indeces = self._path_arc_indeces[:len(self._path_arc_indeces)-r]
        self._update_current_state()

        return

def define_graph(graphnum=0):

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

    def __init__(self, name = '', *args, **kwargs):

        nx.Graph.__init__(self, *args, **kwargs)
        self.name = name

        self.debug = True # debugging print statements


        self.edge_attributes = ['distance','elevation_gain','elevation_loss', 'elevation_change',
                                'min_grade','max_grade','average_grade',
                                'min_altitude','max_altitude','average_altitude',
                                'traversed_count']

        #
        # default factors set to zero to turn off
        # should make setter / getter functions for this
        # to set properly.
        #   0 = not used
        #   > 0 gives relative importance
        #
        self._weight_factors = {'distance' : 1,
                               'elevation_gain' : 1,
                               'elevation_loss' : 0,   # off
                               'min_grade' : 0,        # off
                               'max_grade' : 0,        # off
                               'traversed_count' : 100} # very on

        # self.backtrack_weight =

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
                    self._adj[tail][head][k+'_scaled'] = (self._adj[tail][head] - min_val) / max_min

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

    def recompute_edge_weights(self, edges=None):
        """
        Recompute edge weights based off of wether or not we
        consider various things.
        """

        # need a list of what we are considering
        # and how to treat it. Hard code for now

        # weightings:

        for tail in self._adj:
            max_tail_distance = np.max([self._adj[tail][h]['distance_scaled'] for h in self._adj[tail]])
            for head in self._adj[tail]:
                e = self._adj[tail][head]

                # NEED TO RESCALE ALL OF THESE TO SAME MAGNITUDE!!!

                # apply weights for all '_scaled' properties using simple sum for now
                # need to control this better later. Handle traversed coutn separately
                self._adj[tail][head]['weight'] = np.sum([ self._weight_factors[k]*e[k] for k in e.keys() if ((k != 'traversed_count_scaled') and ('scaled' in k)) ])
                self._adj[tail][head]['weight'] = self._weight_factors['traversed_count']*e['traversed_count']*max_tail_distance ## increase by size of max_distance off of tail


        return

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


        values_list = [self.get_edge_data(u,v)[key] for (u,v) in edges]

        if function is None:
            return values_list
        else:
            value = function(values_list)

            return value

    @staticmethod
    def edges_from_nodes(nodes):
        """
        Generates connected edges from a list of nodes
        """
        #if not isinstance(nodes,list):
        #    raise ValueError

        return [( nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]

    def find_route(self, start_node,
                         target_values,
                         target_methods = None,
                         end_node=None,
                         primary_weight = 'distance',
                         reinitialize=True):
        """
        A DUMB algorithm to try and traverse a graph to find the best
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


        # nrandom = 2 # pick

        # try out a single path
        keep_looping = True
        max_count    = 10
        count        = 0
        current_node = start_node

        # fraction of total distance to try in each jump
        epsilon = 0.25

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
        for k in target_values.keys():
            totals_methods = target_methods[k] if k in target_methods.keys() else default_target_methods[k]

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
            remaining[primary_key] = target_values[primary_key] - totals[iroute][primary_key]

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
                shortest_primary_home = self.reduce_edge_data(primary_key,edges=shortest_edges_home)

                #
                # would potentialy be v cool to iterate here and check once with weight
                # and once with distance to see if using latter produces better match
                # at the expense of just repeating route. Can I edit the keys in graph view?
                # no... but maybe I can generate a new fake weight to do something like
                # this. or pass flag to recompute weights to ignore some things

                if shortest_primary_home > remaining[primary_key]:
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

            # add path
            self._dprint("Possible and next: ", possible_routes[iroute], next_path)

            possible_routes[iroute].extend(next_path[1:])
            # increment totals
            for k in (totals[iroute]).keys():
                newval = self.reduce_edge_data(k, edges=next_edges, function=target_methods[k])
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

        #graph.setup_mincostflow()

        # if we're out of the loop

        return totals, possible_routes


    def get_intermediate_node(self, current_node, target_distance,
                                    G=None, epsilon=0.25, weight='distance',
                                    shift=0.1):

        if G is None:
            G = self

        next_node = None
        while next_node is None:

            # This should ensure that point is actually reachable
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
