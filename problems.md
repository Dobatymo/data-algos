# Problems

## Travelling salesman problem
- also called: 'TSP'
- https://en.wikipedia.org/wiki/Travelling_salesman_problem
- solved by: 'Concorde TSP Solver' application
- solved by: 'Approximate global optimization'
- hardness: NP-hard

## Vertex cover problem
- https://en.wikipedia.org/wiki/Vertex_cover
- solved approximately by: 'Approximate global optimization'
- hardness: NP-hard
- runtime complexity: polynomial for 'Bipartite Graph', 'Tree Graph'

## Exact cover problem
- https://en.wikipedia.org/wiki/Exact_cover
- hardness: 'NP-complete'
- kind of: 'Graph coloring problem'
- is a: 'Decision problem'

## Closest pair of points problem
- https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
- book: 'Introduction to Algorithms'
- domain: 'Computational geometry'
- brute force time complexity: O(n^2) for a set of points of size n

## Count-distinct problem
- https://en.wikipedia.org/wiki/Count-distinct_problem

## Single-source shortest path problem
- https://en.wikipedia.org/wiki/Shortest_path_problem
- book: 'Introduction to Algorithms'
- find shortest path in graph so that the sum of edge weights is minimized
- is a: 'optimization problem'
- solved by 'Breadth-first search' for unweighted graphs
- solved by 'Dijkstra's algorithm' for directed/undirected graphs and positive weights
- solved by 'Bellman–Ford algorithm' for directed graphs with arbitrary weights
- book: 'Introduction to Algorithms'
- properties: 'optimal substructure'
- domain: 'Graph theory'

## Single-pair shortest path problem
- https://en.wikipedia.org/wiki/Shortest_path_problem#Single-source_shortest_paths
- no algorithms with better worst time complexity than for 'Single-source shortest path problem' are know (which is a generalization)

## All-pairs shortest paths problem
- https://en.wikipedia.org/wiki/Shortest_path_problem#All-pairs_shortest_paths
- book: 'Introduction to Algorithms'
- finds the shortest path for all pairs of vectices in a graph
- solved by: 'Floyd–Warshall algorithm', 'Johnson's algorithm'
- domain: 'graph theory'
- properties: 'optimal substructure'

## Longest common substring problem
- https://en.wikipedia.org/wiki/Longest_common_substring_problem
- book: 'The algorithm design manual'
- cf. 'Longest common subsequence problem'
- properties: 'optimal substructure'
- solutions: 'Generalized suffix tree'
- domain: 'Combinatorics'

## Longest common subsequence problem
- https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
- book: 'Introduction to Algorithms', 'The algorithm design manual'
- cf. 'Longest common substring problem'
- solved by: 'Hunt–McIlroy algorithm'
- solved by application: 'diff'
- applications: 'Version control systems', 'Wiki engines', 'Molecular phylogenetics'
- domain: 'Combinatorics'

## Shortest common supersequence problem
- https://en.wikipedia.org/wiki/Shortest_common_supersequence_problem
- applications: DNA sequencing
- domain: 'combinatorics'

## Shortest common superstring problem
- https://en.wikipedia.org/wiki/Shortest_common_supersequence_problem#Shortest_common_superstring
- book: 'The algorithm design manual'
- applications: sparse matrix compression
- domain: 'combinatorics'

## Hamiltonian path problem
- https://en.wikipedia.org/wiki/Hamiltonian_path_problem
- https://www.hackerearth.com/practice/algorithms/graphs/hamiltonian-path/
- solved by algorithms which solve: 'Boolean satisfiability problem'
- domain: "graph theory"

## Maximum subarray problem
- https://en.wikipedia.org/wiki/Maximum_subarray_problem
- applications: 'genomic sequence analysis', 'computer vision', 'data mining'
- solved by: 'Kadane's algorithm'

## Eulerian path problem
- https://en.wikipedia.org/wiki/Eulerian_path
- application: 'in bioinformatics to reconstruct the DNA sequence from its fragments'
- application: 'CMOS circuit design to find an optimal logic gate ordering'
- compare: 'Hamiltonian path problem'
- if exists, optimal solution for: 'Route inspection problem'
- domain: "graph theory"

## Route inspection problem
- also called: 'Chinese postman problem'
- https://en.wikipedia.org/wiki/Route_inspection_problem
- http://mathworld.wolfram.com/ChinesePostmanProblem.html
- domain: "graph theory"

## Closure problem
- https://en.wikipedia.org/wiki/Closure_problem
- domain: "graph theory"
- applications: 'Open pit mining', 'Military targeting', 'Transportation network design', 'Job scheduling'
- can be reduced to: 'Maximum flow problem'

## Maximum flow problem
- https://en.wikipedia.org/wiki/Maximum_flow_problem
- book: 'Introduction to Algorithms'
- domain: 'graph theory'

## Minimum-cost flow problem
- also called: 'MCFP'
- https://en.wikipedia.org/wiki/Minimum-cost_flow_problem
- generalization of: 'Maximum flow problem'
- solved by: 'Linear programming'

## Point location problem
- https://en.wikipedia.org/wiki/Point_location
- domain: 'Computational geometry'

## Point-in-polygon problem
- https://en.wikipedia.org/wiki/Point_in_polygon
- special case of: 'Point location problem'
- domain: 'Computational geometry'

## Halfspace intersection problem
- book: 'Handbook of Discrete and Computational Geometry'
- implemented in: 'scipy.spatial.HalfspaceIntersection', 'Qhull'
- https://en.wikipedia.org/wiki/Half-space_(geometry)

## Largest empty sphere problem
- https://en.wikipedia.org/wiki/Largest_empty_sphere
- domain: 'Computational geometry'
- special cases can be solved using 'Voronoi diagram' in optimal time O(n log(n))

## Subset sum problem
- https://en.wikipedia.org/wiki/Subset_sum_problem
- book: 'Introduction to Algorithms'
- hardness: NP-complete
- special case of: 'Knapsack problem'

## Graph realization problem
- https://en.wikipedia.org/wiki/Graph_realization_problem
- is a: 'Decision problem'
- domain: 'Graph theory'

## Maximum independent set problem
- https://en.wikipedia.org/wiki/Independent_set_(graph_theory)#Maximum_independent_sets_and_maximum_cliques
- solved approximately by: 'networkx.algorithms.approximation.independent_set.maximum_independent_set'

## Boolean satisfiability problem
- https://en.wikipedia.org/wiki/Boolean_satisfiability_problem
- solved approximately by: 'Approximate global optimization'
- hardness: NP-complete

## Nurse scheduling problem
- https://en.wikipedia.org/wiki/Nurse_scheduling_problem
- solved approximately by: 'Approximate global optimization'
- hardness: NP-hard

## Stable marriage problem
- https://en.wikipedia.org/wiki/Stable_marriage_problem
- is a: 'matching problem'

## Maximum cut problem
- https://en.wikipedia.org/wiki/Maximum_cut
- hardness: 'NP-complete'
- is a: 'Decision problem'
- kind of: 'Partition problem'

## Spanning-tree verification
- book: 'Introduction to Algorithms'
- related to: 'Minimum spanning tree'

## Cycle decomposition
- https://en.wikipedia.org/wiki/Cycle_decomposition_(group_theory)

## Envy-free item assignment
- https://en.wikipedia.org/wiki/Envy-free_item_assignment
- type of: 'Fair item assignment', 'Fair division'

## Matrix chain multiplication
- https://en.wikipedia.org/wiki/Matrix_chain_multiplication
- optimization problem
- solved by: 'Hu and Shing algortihm for matrix chain products'

## RSA problem
- https://en.wikipedia.org/wiki/RSA_problem
- see: 'Integer factorization'

## Integer factorization
- https://en.wikipedia.org/wiki/Integer_factorization
- applications: 'cryptography'
- domain: 'Number theory'

## DFA minimization
- https://en.wikipedia.org/wiki/DFA_minimization
- domain: 'Automata theory'

## Connected-component labeling
- book: 'The algorithm design manual'
- https://en.wikipedia.org/wiki/Connected-component_labeling
- domain: 'Graph theory'
- applications: 'Computer vision'

## Topological sorting
- https://en.wikipedia.org/wiki/Topological_sorting
- book: 'The algorithm design manual'
- implemented by: 'posix tsort'
- implemented in: 'boost::graph::topological_sort'
- input: 'directed acyclic graph'
- solved by: 'Depth-first search"
- domain: 'Graph theory'

## Motion planning
- also called: 'Piano mover's problem'
- https://en.wikipedia.org/wiki/Motion_planning
- http://planning.cs.uiuc.edu/node160.html
- book: 'The algorithm design manual'
- hardness: PSPACE-hard

## Biclustering
- https://en.wikipedia.org/wiki/Biclustering

## Covariance matrix estimation
- https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices

## Polynomial factorization
- http://mathworld.wolfram.com/PolynomialFactorization.html
- https://en.wikipedia.org/wiki/Factorization_of_polynomials
- domain: 'Algebra'
