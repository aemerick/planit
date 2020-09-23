try:
    # if installed from pip
    from ortools.graph import pywrapgraph
except:
    # from source (ortools/ortools sub-directory)
    from ortools.ortools.graph import pywrapgraph

import numpy as np

import networkx as nx

import min_cost_flow_test as mcf_test

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

def define_graph():
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


    # make sure this is a sorted node path !!!
    graph = Graph(node_paths)

    return graph


#class TrailMap(nx.Graph):

#    def __init__(self, *args, **kwargs):






def traverse_graph(graph, start_node, end_node, target_distance):
    """
    A DUMB algorithm to try and traverse a graph to find the best
    possible path that fits certain constraints.

    Assumes the node paths are SORTED
    """


    nrandom = 2 # pick

    # try out a single path
    keep_looping = False
    max_count = 100
    count = 0

    current_node = start_node

    # fraction of total distance to try in each jump
    epsilon = 0.25

    # do pre-filtering here

    while (keep_looping):

        # get a list of all possible paths within a desired distance:
        possible_points = nx.single_source_dijkstra(G, current_node,
                                                       weight='distance',
                                                       cutoff=epsilon*target_distance)

        if len(possible_points) == 0:
            _myprint()

        nx.shortest_path()
        current_path = GraphPath()

        # use edges_to_hide = nx.classes.filters.hide_edges(edges)
        # to filter out based on min / max grade



        count = count + 1
        if count >= max_count:
            print("Reaching maximum iterations")
            keep_looping= False

    #graph.setup_mincostflow()

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
