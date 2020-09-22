# """From Bradley, Hax, and Magnanti, 'Applied Mathematical Programming', figure 8.1."""


#
# Covering myself in case I screw up ortools install again
#
try:
    # if installed from pip
    from ortools.graph import pywrapgraph
except:
    # from source (ortools/ortools sub-directory)
    from ortools.ortools.graph import pywrapgraph

import numpy as np


def set_nodes_and_supplies(node_list,start_node,end_node=None):
    """
    Given a list of node connections, adds an additional
    'fake' node in the case that the start node and the end node
    are the same. Returns the new node list and the source list.

    Otherwise does not chage node list but does set the source list.

    Parameters
    ----------
    node_list  : list of size 4 tuples
                 List of node connections: [(start_node, end_node, capacity, cost),...]
    start_node : (int) Start node number
    end_node   : (int,optional) End node number. Assumed same as
                 start node if set to None. Default : None

    Returns
    --------
    node_list : list of size 4 tuples
                List of node connections. Same as input list if start_node and
                end_node are not the same.
    supplies   : list
                List of source values (1 at start_node, -1 at end node, zeros
                elsewhere).
    """

    if len(node_list) == 1:
        print("Only one node provided. Must have multiple nodes.")
        raise ValueError

    if end_node is None:
        end_node = start_node

    unique_start_nodes = np.unique(node_list[:,0])
    unique_end_nodes   = np.unique(node_list[:,1])
    unique_nodes       = np.unique([unique_start_nodes,unique_end_nodes])

    if not (start_node in unique_start_nodes):
        print("Start node not available in list ", start_node)
        raise ValueError

    if not (end_node in unique_end_nodes):
        print("End node not available in list: ", end_node)
        raise ValueError

    fake_n = None # initialized for error catching
    if end_node == start_node:
        # need to add to list of nodes a new fake node that is an
        # identical copy of the start node. It has the same connections
        # but is not directly linked to the start node

        fake_n = np.max(unique_nodes) + 1

        nodes_to_add = []
        for n in node_list:
            if (n[0] == end_node):
                nodes_to_add.append((fake_n, n[1], n[2], n[3]))
            elif (n[1] == end_node):
                nodes_to_add.append((n[0], fake_n, n[2], n[3]))

        print(nodes_to_add)
        node_list= np.insert(node_list,0, nodes_to_add, axis=0)

        # updating for consistency
        unique_start_nodes = np.insert(unique_start_nodes,0,fake_n)
        unique_end_nodes = np.insert(unique_end_nodes,0,fake_n)
        unique_nodes = np.insert(unique_nodes,0,fake_n)

    # assign supplies. start is 1, end is -1, rest are 0
    supplies = [0]*(np.max(unique_nodes)+1) # number of supplies is
    supplies[start_node] = 1
    supplies[end_node if (start_node!=end_node) else fake_n]   = -1

    return node_list, supplies

def main():
  # Define four parallel arrays: start_nodes, end_nodes, capacities, and unit costs
  # between each pair. For instance, the arc from node 0 to node 1 has a
  # capacity of 15 and a unit cost of 4.

  #
  # Need to generate a graph using star / end pairs for each
  # direction possible (if one direction is not possible). This
  # may be easiest to do and keep clean by making tuples and then
  # stripping these into the individual lists. I *think* in my case
  # I'll need to make the capacities = 1 (or same value...) and
  # the unit costs = distance, elevation, etc.
  #
  # This seems to rely on all values being integers. SO pick a precision (0.01?)
  # and work in units of 100's of miles (or mabe just use to nearest foot?)  #
  # tuples are (start, end, capacity, cost)


  start = 0 # start node value
  end   = 0 # end node value
  desired_cost = 1 # desired cost (distance, etc)

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

  # sets up the nodes to
  node_paths, supplies = set_nodes_and_supplies(node_paths, start, end);

  start_nodes = node_paths[:,0].tolist()
  end_nodes   = node_paths[:,1].tolist()
  capacities  = node_paths[:,2].tolist() # should all be same!
  unit_costs  = node_paths[:,3].tolist()



  # Define an array of supplies at each node.
  #
  #  1 is the start node
  # -1 is the end node
  #supplies = [1,
#              0,
#              0,
#              0,
#              -1,
#              0,
#              0]

  print(start_nodes)
  print(end_nodes)
  print(capacities)
  print(unit_costs)
  print(supplies)


  # Instantiate a SimpleMinCostFlow solver.
  min_cost_flow = pywrapgraph.SimpleMinCostFlow()

  min_cost_flow.SetDesiredCost(desired_cost);

  # Add each arc.
  for i in range(0, len(start_nodes)):
    min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                capacities[i], unit_costs[i])

  # Add node supplies.

  for i in range(0, len(supplies)):
    min_cost_flow.SetNodeSupply(i, supplies[i])

  if min_cost_flow.SolveWithCostAdjustment() == min_cost_flow.OPTIMAL:
    print('Minimum cost:', min_cost_flow.OptimalCost())
    print('')
    print('  Arc    Flow / Capacity  Cost')
    for i in range(min_cost_flow.NumArcs()):
      cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
      print('%1s -> %1s   %3s  / %3s       %3s' % (
          min_cost_flow.Tail(i),
          min_cost_flow.Head(i),
          min_cost_flow.Flow(i),
          min_cost_flow.Capacity(i),
          cost))
  else:
    print('There was an issue with the min cost flow input.')

if __name__ == '__main__':
  main()
