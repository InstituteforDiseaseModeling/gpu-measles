# Inter-node Connectivity via Gravity Model

For each pair of nodes i and j, an amount of infectivity is removed from node_i and deposited in node_j. This amount is calculated using the expression:

delta_infectivity_i_j = k_val * initial_infectivity_node_i * population_node_i / total_simulation_population / ( distance_between_nodes_i_and_j ** p_val)

$$
\Delta_{i,j}=(kI_{i}/N)/(d_{i,j})^{p}
$$

The value of delta_infectivity_i_j is the quantity of infectivity removed from node_i and added to node_j. Transfers do not affect the value of initial_infectivity_node_i.

The value distance_between_nodes_i_and_j is in kilometers, and calculated from latitudes and longitudes using a Haversine function (example attached).

The parameters k_val and p_val are adjustable, but constant for a given simulation. For the network of LGAs in Nigeria, k_val = 0.50 and p_val = 1.40 are appropriate values.
