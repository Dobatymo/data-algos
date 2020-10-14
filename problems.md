# Problems

-- uncategorized

## Multiset permutation
- also called: 'Distinct permutation', 'Permutation with replacement'
- https://en.wikipedia.org/wiki/Permutation#Permutations_of_multisets
- solved by (libraries): 'sympy.utilities.iterables.multiset_permutations', 'more_itertools.distinct_permutations'
- solved by (algorithms): 'Takaoka's multiset permutations algorithm'
- related: 'Multinomial theorem' (number of multiset permutations)

## Content-based near-duplicate video detection
- also called: 'NDVD'

## Simplification of Mixed Boolean-Arithmetic expressions
- also called: 'MBA simplification'
- applications paper: 'Information Hiding in Software with Mixed Boolean-Arithmetic Transforms' (2007) <https://doi.org/10.1007/978-3-540-77535-5_5>
- applications: 'data flow obfuscation', 'obfuscation'
- approaches: 'term rewriting / pattern matching'
- solved by (libraries): 'Python quarkslab/arybo'
- domain: 'cryptography', 'computer algebra', 'bit-vector logic'
- thesis: 'Obfuscation with Mixed Boolean-Arithmetic Expressions: Reconstruction, Analysis and Simplification Tools'

## Secure multi-party computation
- also called: 'multi-party computation', 'MPC', 'privacy-preserving computation'
- https://en.wikipedia.org/wiki/Secure_multi-party_computation

## Secure two-party computation
- also called: '2PC'
- https://en.wikipedia.org/wiki/Secure_two-party_computation
- special case of: 'Secure multi-party computation'

## Oblivious transfer
- also called: 'OT', 'Symmetric private information retrieval', 'Symmetric PIR'
- https://en.wikipedia.org/wiki/Oblivious_transfer
- domain: 'cryptography'

## Private information retrieval
- also called: 'PIR'
- https://en.wikipedia.org/wiki/Private_information_retrieval
- special case of: 'Oblivious transfer'
- domain: 'cryptography'

-- implicit formulation with clear solutions

Solution is given by constraints which must be fullfilled, an equation to be solved.

## Greatest common divisor
- https://en.wikipedia.org/wiki/Greatest_common_divisor
- domain: 'Number theory'

## Linear assignment problem
- also called: 'Assignment problem', 'Cost minimizing assignment problem'
- special case of: 'Linear programming', 'Maximum weight matching'
- solved by (libraries): 'google/or-tools'
- review paper: 'Assignment problems: A golden anniversary survey' (2007)

## Quadratic assignment problem
- also called: 'QAP'
- https://en.wikipedia.org/wiki/Quadratic_assignment_problem
- domain: 'Operations research', 'Combinatorial optimization'
- hardness: 'NP-hard'

## Linear bottleneck assignment problem
- also called: 'LBAP', 'Time minimizing assignment problem', 'TMAP'
- https://en.wikipedia.org/wiki/Linear_bottleneck_assignment_problem

## Categorized assignment problem

## Imbalanced time minimizing assignment problem
- also called: 'ITMAP'
- variant of: 'Linear bottleneck assignment problem', 'Categorized assignment problem'
- original paper: 'A variant of time minimizing assignment problem' (1998)
- paper: 'Exact Algorithms for the Imbalanced Time Minimizing Assignment Problem' (2001)
- review paper: 'Assignment problems: A golden anniversary survey' (2007)
- almost same problem: 'Agent Bottleneck Generalized Assignment Problem'
- each machine must do at least one job
- special case of: 'Mixed integer linear programming'

## Agent Bottleneck Generalized Assignment Problem
- also called: 'ABGAP'
- original paper: 'Bottleneck generalized assignment problems' (1988)
- paper: 'The multi-resource agent bottleneck generalised assignment problem' (2010)
- almost same problem: 'Imbalanced time minimizing assignment problem'
- no constraints on how many jobs per machine
- special case of: 'Mixed integer linear programming'

## Quadratic bottleneck assignment problem
- also called: 'QBAP'
- https://en.wikipedia.org/wiki/Quadratic_bottleneck_assignment_problem
- domain: 'Operations research', 'Combinatorial optimization'
- hardness: 'NP-hard'

## Generalized assignment problem
- also called: 'One-to-many assignment problem'

## Many-to-many assignment problem
- solved by (algorithm): 'Kuhn–Munkres algorithm with backtracking'

## Bandwidth reduction problem
- also called: 'Graph bandwidth minimization problem', 'Matrix bandwidth minimization problem'
- https://en.wikipedia.org/wiki/Graph_bandwidth
- http://algorist.com/problems/Bandwidth_Reduction.html
- http://www.csc.kth.se/~viggo/wwwcompendium/node53.html
- hardness: 'NP-complete'?
- hardness: 'NP-hard'?
- applications: 'System of linear equations'
- special case of: 'Quadratic bottleneck assignment problem'

## Matrix inversion
- https://en.wikipedia.org/wiki/Invertible_matrix
- http://mathworld.wolfram.com/MatrixInverse.html
- solved by: 'Mathematica Inverse'

## System of linear equations
- https://en.wikipedia.org/wiki/System_of_linear_equations
- http://algorist.com/problems/Solving_Linear_Equations.html

## Transportation problem
- also called: 'Monge–Kantorovich transportation problem', 'Hitchcock–Koopmans transportation problem'
- https://rosettacode.org/wiki/Transportation_problem
- special case of: 'Linear programming', 'Minimum-cost flow problem'
- commonly used heuristics: 'Northwest Corner Method'
- solved by (algorithms): 'Network simplex algorithm'

## Job shop scheduling
- also called: 'Job-shop problem', 'JSP'
- https://en.wikipedia.org/wiki/Job_shop_scheduling

## Maximum flow problem
- https://en.wikipedia.org/wiki/Maximum_flow_problem
- https://cp-algorithms.com/graph/edmonds_karp.html
- https://cp-algorithms.com/graph/push-relabel.html
- book: 'Introduction to Algorithms'
- domain: 'Graph theory'
- solved by (libraries): 'google/or-tools'
- solved by (algorithms): 'Bellman–Ford algorithm'

## Minimum-cost flow problem
- also called: 'MCFP'
- https://en.wikipedia.org/wiki/Minimum-cost_flow_problem
- https://cp-algorithms.com/graph/min_cost_flow.html
- generalization of: 'Maximum flow problem', 'Shortest path problem'
- specialization of: 'Minimum-cost circulation problem'
- solved by: 'Linear programming'
- solved by (libraries): 'google/or-tools'

## Minimum-cost maximum-flow problem
- variation of: 'Minimum-cost flow problem'

## Minimum-cost circulation problem
- https://en.wikipedia.org/wiki/Circulation_problem

## 0-1 knapsack problem
- https://en.wikipedia.org/wiki/Knapsack_problem

## Graph isomorphism
- https://en.wikipedia.org/wiki/Graph_isomorphism

## Graph automorphism problem
- https://en.wikipedia.org/wiki/Graph_automorphism
- special case of: 'Graph isomorphism'

## Graph classification
- https://github.com/benedekrozemberczki/awesome-graph-classification

-- explicit formulation with clear solutions

Solution is given by mathematical formula or naive computation method.

## Hamming weight
- also called: 'population count', 'popcount', 'sideways sum', 'bit summation'
- https://en.wikipedia.org/wiki/Hamming_weight
- implemented in: 'std::bitset::count', 'gmpy.popcount'

## Discrete Shearlet Transform
- also called: 'DST'
- paper: 'The Discrete Shearlet Transform: A New Directional Transform and Compactly Supported Shearlet Frames (2010)'
- applications: 'Image processing', 'Facial Expression Recognition'
- variant: 'Discrete Separable Shearlet Transform'

## Mellin transform
- https://en.wikipedia.org/wiki/Mellin_transform
- is a: 'Integral transform'
- applications: 'Audio pitch scaling', 'Image recognition'
- domain: 'Complex analysis'

## Fourier transform
- https://en.wikipedia.org/wiki/Fourier_transform
- is a: 'Integral transform'
- discrete version: 'Discrete Fourier transform'
- applications: 'Signal processing', 'Analysis of differential equations', 'Quantum mechanics'
- domain: 'Complex analysis'
- input: function
- output: function

## Discrete Fourier transform
- also called: 'DFT'
- https://en.wikipedia.org/wiki/Discrete_Fourier_transform
- applications: 'Signal processing'
- is a: 'Discrete transform'
- continuous version: 'Fourier transform'
- naive time complexity: O(n^2)

## Discrete wavelet transform
- also called: 'DWT'
- https://en.wikipedia.org/wiki/Discrete_wavelet_transform
- is a: 'Discrete transform'
- applications: 'Image processing'
- solved by (libraries): 'PyWavelets'

## Discrete cosine transform
- also called: 'DCT'
- https://en.wikipedia.org/wiki/Discrete_cosine_transform
- is a: 'Discrete transform'
- applications: 'Lossy compression'
- used by: 'MP3', 'JPEG'
- naive time complexity: O(n^2)

## Normalized cross-correlation
- also called: 'NCC'
- https://en.wikipedia.org/wiki/Cross-correlation#Normalized_cross-correlation_(NCC)

## Image moments
- https://en.wikipedia.org/wiki/Image_moment

## Matrix chain multiplication
- https://en.wikipedia.org/wiki/Matrix_chain_multiplication
- optimization problem
- solved by: 'Hu and Shing algortihm for matrix chain products'

-- implicit or explicit formulation with clear solutions

## Pareto frontier extraction
- also called: 'Pareto front searching', 'Pareto set'
- https://en.wikipedia.org/wiki/Pareto_efficiency#Pareto_frontier
- solved by (libraries): 'kummahiih/pypareto'
- domain: 'Multi-objective optimization'

## Disjoint set union problem

## Watershed transformation
- https://en.wikipedia.org/wiki/Watershed_(image_processing)

## Connected-component finding ?rename, doesn't sound good'
- https://en.wikipedia.org/wiki/Component_(graph_theory)
- https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
- domain: 'Graph theory'
- commonly solved using: 'Depth-first search'

## Strongly connected component finding
- https://en.wikipedia.org/wiki/Strongly_connected_component
- solved by (algorithms): 'Depth-first search', 'Tarjan's strongly connected components algorithm'

## Connected-component labeling
- book: 'The Algorithm Design Manual'
- https://en.wikipedia.org/wiki/Connected-component_labeling
- domain: 'Graph theory'
- applications: 'Computer vision'

## Minimum bounding box
- https://en.wikipedia.org/wiki/Minimum_bounding_box
- domain: 'Computational geometry'
- solved by: 'Freeman-Shapira's minimum bounding box' (for convex polygons)
- solved by?: 'Convex Hull' -> 'Freeman-Shapira's minimum bounding box' (for any set of points)
- solved by?: 'Rotating calipers'

## Line segment intersection
- https://en.wikipedia.org/wiki/Line_segment_intersection
- naive time complexity: O(n^2)
- domain: 'Computational geometry'

## Synchronizing word
- https://en.wikipedia.org/wiki/Synchronizing_word
- domain: 'Automata theory'
- solved by: 'David Eppstein's algorithm'
- related: 'Road coloring problem'
- hardness of shortest synchronizing word: 'NP-complete'

## Road coloring problem
- https://en.wikipedia.org/wiki/Road_coloring_theorem
- http://mathworld.wolfram.com/RoadColoringProblem.html
- paper: 'Equivalence of topological Markov shifts (1977)'
- domain: 'Graph theory', 'Automata theory'

## Travelling salesman problem
- also called: 'TSP'
- https://en.wikipedia.org/wiki/Travelling_salesman_problem
- solved by: 'Concorde TSP Solver' application
- solved by: 'Approximate global optimization'
- hardness: NP-hard
- is a: 'Combinatorial optimization problem'

## Vertex cover problem
- also called: 'Minimum vertex cover'
- https://en.wikipedia.org/wiki/Vertex_cover
- http://mathworld.wolfram.com/MinimumVertexCover.html
- solved approximately by: 'Approximate global optimization'
- hardness: 'NP-hard', 'APX-complete'
- implemented in: 'Mathematica FindVertexCover'
- runtime complexity: polynomial for 'Bipartite Graph', 'Tree Graph'
- is a: 'Optimization problem'
- applications: 'dynamic detection of race conditions'
- properties: 'fixed-parameter tractable'

## Vertex cover decision problem
- https://en.wikipedia.org/wiki/Vertex_cover
- solved approximately by: 'Approximate global optimization'
- hardness: 'NP-complete'
- runtime complexity: polynomial for 'Bipartite Graph', 'Tree Graph'
- is a: 'Decision problem'
- kind of: 'Independent set problem'

## Exact cover problem
- also called: 'Minimum exact cover'
- https://en.wikipedia.org/wiki/Exact_cover
- http://www.csc.kth.se/~viggo/wwwcompendium/node147.html
- solved by (algorithm): 'Knuth's Algorithm X'

## Exact cover decision problem
- https://en.wikipedia.org/wiki/Exact_cover
- hardness: 'NP-complete'
- kind of: 'Graph coloring problem'
- is a: 'Decision problem'

## Graph coloring problem (chromatic number)
- also called: 'GCP', 'Vertex coloring'
- https://en.wikipedia.org/wiki/Graph_coloring
- http://mathworld.wolfram.com/MinimumVertexColoring.html
- solved approximately by: 'Brelaz's heuristic algorithm'
- implemented in: 'Mathematica MinimumVertexColoring' (exhaustive search)
- is a: 'Combinatorial optimization problem'
- domain: 'Graph theory'

## Graph coloring problem (k-coloring)
- also called: 'GCP', 'Vertex coloring'
- https://en.wikipedia.org/wiki/Graph_coloring
- http://mathworld.wolfram.com/VertexColoring.html
- is a: 'Decision problem'
- domain: 'Graph theory'

## Closest pair of points problem
- https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
- book: 'Introduction to Algorithms'
- domain: 'Computational geometry'
- naive time complexity: O(n^2) for a set of points of size n

## Count-distinct problem
- https://en.wikipedia.org/wiki/Count-distinct_problem
- naive space complexity: linear in distinct number of elements

## Single-source shortest path problem
- problem type: 'Shortest path problem'
- also called: 'Least cost path'
- https://en.wikipedia.org/wiki/Shortest_path_problem
- book: 'Introduction to Algorithms'
- survey paper: 'A Survey of Shortest-Path Algorithms' (2017)
- find shortest path in graph so that the sum of edge weights is minimized
- is a: 'optimization problem'
- solved by 'Breadth-first search' for unweighted graphs
- solved by 'Dijkstra's algorithm' for directed/undirected graphs and positive weights
- solved by 'Bellman–Ford algorithm' for directed graphs with arbitrary weights
- book: 'Introduction to Algorithms'
- properties: 'optimal substructure'
- domain: 'Graph theory'

## Single-pair shortest path problem
- problem type: 'Shortest path problem'
- https://en.wikipedia.org/wiki/Shortest_path_problem#Single-source_shortest_paths
- no algorithms with better worst time complexity than for 'Single-source shortest path problem' are know (which is a generalization)

## All-pairs shortest paths problem
- problem type: 'Shortest path problem'
- https://en.wikipedia.org/wiki/Shortest_path_problem#All-pairs_shortest_paths
- book: 'Introduction to Algorithms'
- finds the shortest path for all pairs of vectices in a graph
- solved by: 'Floyd–Warshall algorithm', 'Johnson's algorithm'
- domain: 'graph theory'
- properties: 'optimal substructure'

## Graph connectivity
- https://en.wikipedia.org/wiki/Connectivity_(graph_theory)
- description: "Are these nodes a and b of the graph connected?"
- related: 'Connected-component finding'
- domain: 'graph theory'

## Negative cycle detection
- also called: 'Negative weight cycle detection"
- https://cp-algorithms.com/graph/finding-negative-cycle-in-graph.html
- description: "Does the graph contain negative cycles?"
- input: 'directed weighted graph'
- domain: 'graph theory'
- solved by (algorithms): 'Bellman–Ford algorithm'
- related challenges: 'SPOJ: UCV2013B'

## Longest common substring problem
- https://en.wikipedia.org/wiki/Longest_common_substring_problem
- book: 'The Algorithm Design Manual'
- cf. 'Longest common subsequence problem'
- properties: 'optimal substructure'
- solutions: 'Generalized suffix tree'
- naive time complexity: O((n+m)*m^2) for length n and m of two strings (https://www.techiedelight.com/longest-common-substring-problem/)
- domain: 'Combinatorics'

## Longest palindromic substring problem
- also called: 'Longest symmetric factor problem'
- https://en.wikipedia.org/wiki/Longest_palindromic_substring

## Longest common subsequence problem
- https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
- book: 'Introduction to Algorithms', 'The Algorithm Design Manual'
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
- book: 'The Algorithm Design Manual'
- applications: sparse matrix compression
- domain: 'Combinatorics'

## Hamiltonian path problem
- https://en.wikipedia.org/wiki/Hamiltonian_path_problem
- https://www.hackerearth.com/practice/algorithms/graphs/hamiltonian-path/
- solved by algorithms which solve: 'Boolean satisfiability problem'
- domain: "Graph theory"

## Maximum subarray problem
- https://en.wikipedia.org/wiki/Maximum_subarray_problem
- applications: 'Genomic sequence analysis', 'Computer vision', 'Data mining'
- solved by: 'Kadane's algorithm'

## Eulerian path problem
- https://en.wikipedia.org/wiki/Eulerian_path
- application: 'in bioinformatics to reconstruct the DNA sequence from its fragments'
- application: 'CMOS circuit design to find an optimal logic gate ordering'
- compare: 'Hamiltonian path problem'
- if exists, optimal solution for: 'Route inspection problem'
- domain: 'Graph theory'
- solved by (libraries): 'google/or-tools'

## Route inspection problem
- also called: 'Chinese postman problem'
- https://en.wikipedia.org/wiki/Route_inspection_problem
- http://mathworld.wolfram.com/ChinesePostmanProblem.html
- domain: 'Graph theory'

## Closure problem
- https://en.wikipedia.org/wiki/Closure_problem
- domain: 'Graph theory'
- applications: 'Open pit mining', 'Military targeting', 'Transportation network design', 'Job scheduling'
- can be reduced to: 'Maximum flow problem'

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

## Independent set decision problem
- https://en.wikipedia.org/wiki/Independent_set_(graph_theory)
- is a: 'Decision problem'
- hardness: 'NP-complete'

## Finding a maximal independent set
- also called: 'MIS', 'Maximal stable set'
- https://en.wikipedia.org/wiki/Maximal_independent_set
- https://en.wikipedia.org/wiki/Maximal_independent_set#Finding_a_single_maximal_independent_set
- solved by (algorithms): 'Luby's algorithm', 'Blelloch's algorithm'

## Finding all maximal independent sets
- also called: 'MIS', 'Maximal stable set'
- https://en.wikipedia.org/wiki/Maximal_independent_set
- https://en.wikipedia.org/wiki/Maximal_independent_set#Listing_all_maximal_independent_sets

## Maximum independent set problem
- also called: 'MIS'
- https://en.wikipedia.org/wiki/Independent_set_(graph_theory)#Maximum_independent_sets_and_maximum_cliques
- solved by: 'Xiao and Nagamochi's algorithm for the maximum independent set'
- solved approximately by: 'Boppana and Halldórsson's approximation algorithm for the maximum independent set'
- hardness: 'NP-hard'
- naive time complexity: O(n^2 * 2^n)
- domain: 'Graph theory'
- is a: 'Optimization problem'

## Finding a maximum clique
- https://en.wikipedia.org/wiki/Clique_problem#Finding_maximum_cliques_in_arbitrary_graphs
- properties: 'fixed-parameter intractable', 'hard to approximate'

## Finding a maximum weight clique
- https://en.wikipedia.org/wiki/Clique_problem
- generalization of: 'Finding a maximum clique'

## Finding all maximal cliques
- https://en.wikipedia.org/wiki/Clique_problem#Listing_all_maximal_cliques
- http://mathworld.wolfram.com/MaximalClique.html
- solved by (algorithms): 'Bron–Kerbosch algorithm'
- solved by (libraries): 'google/or-tools'

## Clique decision problem
- https://en.wikipedia.org/wiki/Clique_problem
- https://courses.cs.washington.edu/courses/csep521/99sp/lectures/lecture04/sld018.htm
- is a: 'Decision problem'
- hardness: 'NP-complete'

## Edge dominating set decision problem
- hardness: 'NP-complete'
- is a: 'Decision problem'

## Edge dominating set
- https://en.wikipedia.org/wiki/Edge_dominating_set

## Minimum edge dominating set
- https://en.wikipedia.org/wiki/Edge_dominating_set
- http://www.nada.kth.se/~viggo/wwwcompendium/node13.html
- hardness: 'NP-hard'

## Set cover decision problem
- https://en.wikipedia.org/wiki/Set_cover_problem
- hardness: 'NP-complete'
- is a: 'Decision problem'

## Set cover problem
- also called: 'Minimum set cover'
- https://en.wikipedia.org/wiki/Set_cover_problem
- http://www.csc.kth.se/~viggo/wwwcompendium/node146.html
- hardness: 'NP-hard'
- special case of: 'Integer linear programming'

## Weighted set cover problem
- https://en.wikipedia.org/wiki/Set_cover_problem#Weighted_set_cover

## Maximal matching
- https://en.wikipedia.org/wiki/Matching_(graph_theory)#Maximal_matchings

## Maximum matching
- https://en.wikipedia.org/wiki/Matching_(graph_theory)#Maximum_matching
- https://brilliant.org/wiki/matching/
- maximal matching with the maximum number of edges
- is a: 'Combinatorial optimization problem'
- domain: 'Graph theory'
- solved by (algorithm): 'Blossom algorithm', 'Micali and Vazirani's matching algorithm'

## Minimum maximal matching
- https://en.wikipedia.org/wiki/Matching_(graph_theory)
- maximal matching with the minimum number of edges
- same as: Minimum edge dominating set'

## Boolean satisfiability problem
- also called: 'SAT'
- https://en.wikipedia.org/wiki/Boolean_satisfiability_problem
- solved approximately by: 'Approximate global optimization'
- hardness: NP-complete
- domain: 'Computer science'
- solved by: 'DPLL algorithm'

## Nurse scheduling problem
- https://en.wikipedia.org/wiki/Nurse_scheduling_problem
- solved approximately by: 'Approximate global optimization'
- hardness: NP-hard

## Stable marriage problem
- https://en.wikipedia.org/wiki/Stable_marriage_problem
- is a: 'matching problem'

## Partition problem
- also called: 'Number partitioning'
- https://en.wikipedia.org/wiki/Partition_problem
- domain: 'Number theory', 'Computer science'
- hardness: 'NP-complete'
- is a: 'Decision problem'
- special case of: 'Subset sum problem'

## Linear partition problem
- book: 'The Algorithm Design Manual'
- http://www8.cs.umu.se/kurser/TDBAfl/VT06/algorithms/BOOK/BOOK2/NODE45.HTM
- domain: 'Number theory', 'Computer science'
- is a: 'Optimization problem'
- hardness: 'NP-hard'

## Maximum cut problem
- https://en.wikipedia.org/wiki/Maximum_cut
- hardness: 'NP-complete'
- is a: 'Decision problem'
- kind of: 'Partition problem'

## Normalized Cut
- also called: 'NCUT'
- hardness: 'NP-complete'
- paper: 'Normalized cuts and image segmentation' (2000) <https://doi.org/10.1109/34.868688>
- kind of: 'Partition problem'

## Spanning-tree verification
- book: 'Introduction to Algorithms'
- related to: 'Minimum spanning tree'

## Cycle decomposition
- https://en.wikipedia.org/wiki/Cycle_decomposition_(group_theory)

## Envy-free item assignment
- https://en.wikipedia.org/wiki/Envy-free_item_assignment
- type of: 'Fair item assignment', 'Fair division'

## RSA problem
- https://en.wikipedia.org/wiki/RSA_problem
- see: 'Integer factorization'

## Integer factorization
- https://en.wikipedia.org/wiki/Integer_factorization
- applications: 'Cryptography'
- domain: 'Number theory'

## DFA minimization
- https://en.wikipedia.org/wiki/DFA_minimization
- domain: 'Automata theory'

## Topological sorting
- https://en.wikipedia.org/wiki/Topological_sorting
- book: 'The Algorithm Design Manual'
- implemented by: 'posix tsort'
- implemented in: 'boost::graph::topological_sort'
- input: 'directed acyclic graph'
- solved by: 'Depth-first search"
- domain: 'Graph theory'

## Polynomial factorization
- http://mathworld.wolfram.com/PolynomialFactorization.html
- https://en.wikipedia.org/wiki/Factorization_of_polynomials
- domain: 'Algebra'

## Nearest neighbor search
- also called: 'Similarity search', 'NNS', 'post-office problem'
- https://en.wikipedia.org/wiki/Nearest_neighbor_search
- https://en.wikipedia.org/wiki/Similarity_search
- solved by: 'Space partitioning', 'Linear search'
- solved by (libraries): 'C++ ANN', 'nmslib/hnsw', 'nmslib/nmslib'
- approximate variant: 'Approximate nearest neighbor search'

## Approximate nearest neighbor search
- also called: 'ANN', 'Approximate similarity search'
- book: 'Handbook of Discrete and Computational Geometry'
- https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor
- solved by (libraries): 'PyNNDescent', 'spotify/annoy'
- solved by: 'Hierarchical Navigable Small World graphs', 'Locality-sensitive hashing', 'Cover tree', 'Vector quantization'
- exact variant: 'Nearest neighbor search'

## Range searching
- https://en.wikipedia.org/wiki/Range_searching
- related problems: 'Nearest neighbor search'

## Range reporting
- https://en.wikipedia.org/wiki/Range_reporting
- special case of: 'Range searching'
- query name: 'reporting query'
- answer type: list

## Range counting
- special case of: 'Range searching'
- query name: 'Counting query'
- answer type: int

## Emptiness query of range searching
- special case of: 'Range searching'
- query name: 'Emptiness query'
- answer type: bool

## Lowest common ancestor
- also called: 'LCA', 'Least common ancestor'
- https://en.wikipedia.org/wiki/Lowest_common_ancestor

## Longest common prefix problem
- https://en.wikipedia.org/wiki/LCP_array
- https://leetcode.com/problems/longest-common-prefix/

## Range minimum query
- also called: 'RMQ'
- https://en.wikipedia.org/wiki/Range_minimum_query
- http://wcipeg.com/wiki/Range_minimum_query
- related: 'Lowest common ancestor', 'Longest common prefix problem'
- variants: 'dynamic', 'static'
- solved by: 'Segment tree'

## Kernel density estimation
- also called: 'KDE'
- https://en.wikipedia.org/wiki/Kernel_density_estimation
- domain: 'statistics'
- properties: 'non-parametric'
- solved by: 'Mathematica SmoothKernelDistribution', 'org.apache.spark.mllib.stat.KernelDensity'

## Sorting problem
- https://en.wikipedia.org/wiki/Sorting_algorithm

## Selection problem
- https://en.wikipedia.org/wiki/Selection_algorithm

## Partial sorting
- https://en.wikipedia.org/wiki/Partial_sorting
- solved by: heaps, 'quickselsort', 'Quickselect'
- variant of: 'Sorting problem'
- domain: 'Computer science'

## Incremental sorting
- https://en.wikipedia.org/wiki/Partial_sorting#Incremental_sorting
- solved by: 'quickselect', heaps
- domain: 'Computer science'
- variant of: 'Sorting problem'

## List ranking
- https://en.wikipedia.org/wiki/List_ranking
- domain: 'Parallel computing'

## Shortest pair of edge disjoint paths
- special case of: 'Minimum-cost flow problem'
- applications: 'Routing'
- domain: 'Graph theory'

## Minimal Perfect Hashing
- http://iswsa.acm.org/mphf/index.html
- https://en.wikipedia.org/wiki/Perfect_hash_function#Minimal_perfect_hash_function
- domain: 'Computer science'

## K-server problem
- https://en.wikipedia.org/wiki/K-server_problem
- domain: 'Computer science'
- there exists a related unsolved problem

## X + Y sorting
- https://en.wikipedia.org/wiki/X_%2B_Y_sorting
- domain: 'Computer science'
- naive time complexity: O(nm log(nm))
- variant of: 'Sorting problem'
- there exists a related unsolved problem
- applications: 'transit fare minimisation'

## 3SUM
- https://en.wikipedia.org/wiki/3SUM
- related: 'Subset sum problem'

## Motion planning
- also called: 'Navigation problem', 'Piano mover's problem'
- https://en.wikipedia.org/wiki/Motion_planning
- http://planning.cs.uiuc.edu/node160.html
- book: 'The Algorithm Design Manual'
- hardness: PSPACE-hard
- domain: 'Robotics'

## Covariance matrix estimation
- https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices

-- problems with clear solutions for simple metrics (multiple metrics possible)

## Downsampling
- also called: 'Decimation', 'Line simplification', 'Smoothing'
- https://en.wikipedia.org/wiki/Downsampling_(signal_processing)
- domain: 'Signal processing'
- survey paper: 'Performance Evaluation of Line Simplification Algorithms for Vector Generalization' (2013) <https://doi.org/10.1179/000870406X93490>

## Biclustering
- https://en.wikipedia.org/wiki/Biclustering

## Sequence alignment
- https://en.wikipedia.org/wiki/Sequence_alignment

## Template matching
- https://en.wikipedia.org/wiki/Template_matching

## Hierarchical clustering
- https://en.wikipedia.org/wiki/Hierarchical_clustering
- applications: 'Data mining', 'Paleoecology'
- domain: 'Statistics'

## Approximate string matching
- https://en.wikipedia.org/wiki/Approximate_string_matching
- paper: 'Fast Approximate String Matching in a Dictionary'
- applications: 'spell checking', 'nucleotide sequence matching'

-- problems with approximate solutions only (no known way to have 100% perfect results)

Evaluation is usually empirical, based on metrics inpired by human perception or compared to groundtruth based on some metric.

## Audio time stretching and pitch scaling
- https://en.wikipedia.org/wiki/Audio_time_stretching_and_pitch_scaling

## Optical flow
- https://en.wikipedia.org/wiki/Optical_flow
- http://vision.middlebury.edu/flow/eval/
- applications: 'Motion estimation', 'video compression', 'object detection', 'object tracking', 'image dominant plane extraction', 'movement detection', 'robot navigation , 'visual odometry'
- domain: 'machine vision', 'computer vision'
- metrics: 'Endpoint error', 'Average angular error'

## Video denoising
- https://en.wikipedia.org/wiki/Video_denoising
- https://paperswithcode.com/task/video-denoising

## Music source separation
- also called: 'Signal separation'
- https://en.wikipedia.org/wiki/Signal_separation
- https://paperswithcode.com/task/music-source-separation
- applications: 'Music information retrieval', 'Music transcription'
- libraries: 'Spleeter'

## Image segmentation
- https://en.wikipedia.org/wiki/Image_segmentation
- survey paper: 'Survey on Image Segmentation Techniques' (2015) <https://doi.org/10.1016/j.procs.2015.09.027>
- benchmark: http://mosaic.utia.cas.cz/
- benchmark: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
- sota: https://paperswithcode.com/area/computer-vision/semantic-segmentation

## Superpixel
- survey paper: 'Superpixels: An evaluation of the state-of-the-art' (2016) <https://doi.org/10.1016/j.cviu.2017.03.007>
- http://davidstutz.de/projects/superpixel-benchmark/
- metrics: 'Boundary Recall', 'Undersegmentation Error', 'Explained variation'

## Image binarization
- also called: 'Adaptive image binarization', 'Image thresholding'
- https://en.wikipedia.org/wiki/Thresholding_(image_processing)
- survey paper: 'Survey over image thresholding techniques and quantitative performance evaluation' (2004) <https://doi.org/10.1117/1.1631315>
- solved by: 'Sauvola binarization', 'Otsu's method'

## Stemming
- https://en.wikipedia.org/wiki/Stemming
- domain: 'Computational linguistics', 'Natural language processing'

## Part-of-speech tagging
- https://en.wikipedia.org/wiki/Part-of-speech_tagging
- http://nlpprogress.com/english/part-of-speech_tagging.html
- domain: 'Computational linguistics', 'Natural language processing'
- metric: 'accuracy'
- usually solved by: 'machine learning'
- sota implementations (selection): 'zalandoresearch/flair', 'google/meta_tagger'

## Multivariate interpolation
- also called: 'Spatial interpolation'

## Image scaling
- also called: 'Super resolution', 'SR', 'Super-resolution reconstruction', 'SRR'
- https://en.wikipedia.org/wiki/Image_scaling
- https://en.wikipedia.org/wiki/Super-resolution_imaging
- metrics: 'PSNR', 'SSIM', 'IW-SSIM'
- special case of: 'Multivariate interpolation'

## Single image super-resolution
- also called: 'SISR'

## Pixel-art scaling
- https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms
- subfield of: 'Image scaling'

## Demosaicing
- https://en.wikipedia.org/wiki/Demosaicing
- related problem: 'Image scaling'

## Blind deconvolution
- https://en.wikipedia.org/wiki/Blind_deconvolution
- related problems: 'Dereverberation', 'Deblurring'
- applications: 'Astronomical imaging', 'Medical imaging'

## Direction finding
- also called: 'DF', Radio direction finding', 'RDF'

## Frequency estimation
- https://en.wikipedia.org/wiki/Spectral_density_estimation#Frequency_estimation

## Grammatical error correction
- also called: 'GEC'
- https://paperswithcode.com/task/grammatical-error-correction
- domain: 'Computational linguistics'
- usually solved by: 'language models', 'sequence-to-sequence models'
- tasks: 'BEA-2019 Shared Task', 'CoNLL-2014 Shared Task', 'CoNLL-2013 Shared Task'

## Speech recognition
- also called: 'automatic speech recognition', 'ASR', 'speech to text', 'STT'
- https://en.wikipedia.org/wiki/Speech_recognition
- https://paperswithcode.com/task/speech-recognition
- solved by (applications): 'CMU Sphinx'

## Speaker recognition
- also called: 'speaker identification', 'voice recognition'
- https://en.wikipedia.org/wiki/Speaker_recognition
- https://paperswithcode.com/task/speaker-identification
- input: 'audio'
- output: 'text'

## Handwriting recognition
- also called: 'HWR', 'Handwritten Text Recognition', 'HTR'
- https://en.wikipedia.org/wiki/Handwriting_recognition
- input: 'image' (offline) or f(time) -> (real, real, bool) (online)
- output: '(formatted) text'
- conferences: 'International Conference on Frontiers in Handwriting Recognition'

-- subjective (no simple objective performance metric)

Empirical evaluation or based on "dumb" metrics. Even comparison to groundtruth is not easily possible.

## Image registration
- also called: 'Image stitching'?is there any difference?'
- https://en.wikipedia.org/wiki/Image_registration
- https://en.wikipedia.org/wiki/Image_stitching
- solved by (applications): '3DSlicer'
- applications: 'Medical imaging', 'Neuroimaging', 'Astrophotography', 'Panoramic image creation'
- related problems: 'Image rectification'

## Inpainting
- https://en.wikipedia.org/wiki/Inpainting

## Image tracing
- also called: 'raster-to-vector conversion', 'vectorization'
- domain: 'computer graphics'
- related problems: 'Image scaling'
- applications: 'Potrace', 'CorelDRAW PowerTRACE'

## Terrain generation
- also called: 'Heightmap generation'
- https://en.wikipedia.org/wiki/Scenery_generator

## Image retargeting
- also called: 'content-aware image resizing'
- review paper: 'A Comparative Study of Image Retargeting' (2010)
- solved by (selection): 'Seam carving'
- domain: 'Image processing', 'Computer graphics'

## Paraphrase generation
- metrics (selection): 'BLEU'
- usually solved by: 'machine learning'

## Machine translation
- also called: 'MT'
- https://en.wikipedia.org/wiki/Machine_translation
- domain: 'Computational linguistics'
- metrics (selection): 'BLEU', 'ROUGE'
- usually solved by: 'machine learning'
- approaches: 'Statistical machine translation', 'Neural machine translation', 'Rule-based machine translation', 'Dictionary-based machine translation'
