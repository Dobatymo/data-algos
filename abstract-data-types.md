# Abstract data types

## Collection
- also called: 'Sequence'
- supports iteration (multiple times)
- known length

## List
- also called: 'Iterable', 'Stream'
- https://en.wikipedia.org/wiki/List_(abstract_data_type)
- usually implemented as: 'Linked list'
- supports iteration (once)
- possibly infinite

## Array
- https://en.wikipedia.org/wiki/Array_data_type
- implemented as: 'Array'
- constant time random access

## Stack
- https://en.wikipedia.org/wiki/Stack_(abstract_data_type)
- usually implemented as: 'Array', 'Singly linked list'

## Set
- https://en.wikipedia.org/wiki/Set_(abstract_data_type)
- is a: 'Collection'
- list of unique elements
- similar to mathematical set
- offers fast in-collection checks

### Unordered set (interface)
- usually implemented as: 'Hash table'
- implemented in: 'std::unordered_set', 'Python set', 'Java Set'

### Ordered set (interface)
- usually implemented as: 'Binary search tree'
- could be implemented as deterministic acyclic finite state acceptor
- implemented in: 'std::set', 'Java SortedSet'

## Multiset
- also called: 'Bag'
- https://en.wikipedia.org/wiki/Set_(abstract_data_type)#Multiset
- is a: 'Collection'
- like a set where duplicities of unique values are also stored

### Unordered multiset (interface)
- usually implemented as: 'Hash table'
- implemented in: 'std::unordered_multiset ', 'Python collections.Counter', 'Smalltalk Bag'

### Ordered multiset (interface)
- usually implemented as: 'Binary search tree'
- implemented in: 'std::multiset', 'Java com.google.common.collect.TreeMultiset'

## Map
- also called: 'Associative array'
- https://en.wikipedia.org/wiki/Associative_array
- used to map a unique key to a value (a phone number to a name). Can only be used for exact matches. ie. cannot find all phone numbers which differ only in one digit.

### Unordered map (interface)
- usually implemented as: 'Hash table'

### Ordered map (interface)
- usually implemented as: 'Binary search tree'

## Double-ended queue (deque)
- https://en.wikipedia.org/wiki/Double-ended_queue
- usually implemeted as: 'array with pointers to smaller arrays' or 'Doubly linked list'
- implemented in: 'std::deque', 'Python collections.deque'
- applications: 'Job scheduling'

## Priority queue
- https://en.wikipedia.org/wiki/Priority_queue
- usually implemented as: 'heap'
- implemented in: 'Python heapq'

## Tree
- https://en.wikipedia.org/wiki/Tree_(data_structure)
- specialisation of: 'Graph'

## Graph
- https://en.wikipedia.org/wiki/Graph_(abstract_data_type)

## Boolean function
- https://en.wikipedia.org/wiki/Boolean_function

## Rooted graph
- https://en.wikipedia.org/wiki/Rooted_graph

## Simple graph
- https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)#Simple_graph

# Objects which are properties of other objects

## Exact cover
- https://en.wikipedia.org/wiki/Exact_cover
- related decision problem: 'Exact cover problem'
- based on: 'set of subsets of set'
- found by: 'Knuth's Algorithm X'

## Maximum cut
- https://en.wikipedia.org/wiki/Maximum_cut
- related decision problem: 'Maximum cut problem'
- based on: 'Graph'
- hardness: 'NP-hard', 'APX-hard'

## Minimum cut
- https://en.wikipedia.org/wiki/Minimum_cut
- https://xlinux.nist.gov/dads/HTML/minimumcut.html
- found by: 'Karger's algorithm', 'Stoer–Wagner algorithm'
- based on: 'Graph'
- found by (libraries): 'graph_tool.flow.min_cut'

## Delaunay triangulation
- https://en.wikipedia.org/wiki/Delaunay_triangulation
- http://mathworld.wolfram.com/DelaunayTriangulation.html
- book: 'Handbook of Discrete and Computational Geometry'
- paper: 'Sur la sphère vide. A la mémoire de Georges Voronoï'
- domain: 'Geometry'
- found by: 'Bowyer–Watson algorithm'
- is dual graph of: 'Voronoi diagram'
- related: 'Euclidean minimum spanning tree'
- implemented in: 'scipy.spatial.Delaunay' (using Qhull)
- based on: 'set of points in metric space'

## Voronoi diagram
- also called: 'Voronoi tessellation', 'Voronoi decomposition', 'Voronoi partition'
- https://en.wikipedia.org/wiki/Voronoi_diagram
- http://mathworld.wolfram.com/VoronoiDiagram.html
- book: 'Handbook of Discrete and Computational Geometry', 'The algorithm design manual'
- paper: 'Nouvelles applications des paramètres continus à la théorie de formes quadratiques'
- domain: 'Geometry'
- found by: 'Fortune's algorithm', 'Lloyd's algorithm'
- is dual graph of: 'Delaunay triangulation'
- applications: 'Space partitioning', 'biological structure modelling', 'growth patterns in ecology', 'Epidemiology'
- related: 'Closest pair of points problem', 'Largest empty sphere problem'
- implemented in: 'scipy.spatial.Voronoi' (using Qhull), 'scipy.spatial.SphericalVoronoi'
- based on: 'set of points in metric space'

## Convex hull
- also called: 'minimum convex polygon'
- book: 'Handbook of Discrete and Computational Geometry', 'Introduction to Algorithms', 'The algorithm design manual'
- https://en.wikipedia.org/wiki/Convex_hull
- http://mathworld.wolfram.com/ConvexHull.html
- domain: 'Computational geometry'
- found by: 'Kirkpatrick–Seidel algorithm', 'Chan's algorithm', 'Quickhull algorithm'
- implemented in: 'Mathematica ConvexHullMesh', 'Python scipy.spatial.ConvexHull' (using Quickhull)
- based on: 'set of points in affine space'

## Polynomial of best approximation
- https://www.encyclopediaofmath.org/index.php/Polynomial_of_best_approximation
- domain: 'Approximation theory'
- found by: 'Remez algorithm'
- based on: 'Function'

## Lowest common ancestor
- also called: 'LCA'
- https://en.wikipedia.org/wiki/Lowest_common_ancestor
- domain: 'Graph theory'
- found by: 'Tarjan's off-line lowest common ancestors algorithm'
- based on: 'Directed acyclic graph'

## Minimum spanning tree
- also called: 'MST'
- https://en.wikipedia.org/wiki/Minimum_spanning_tree
- http://algorist.com/problems/Minimum_Spanning_Tree.html
- book: 'Introduction to Algorithms'
- found by: 'Kruskal's algorithm', 'Prim's algorithm'
- unique solution
- applications: 'Network design', 'Image segmentation', 'Cluster analysis'
- based on: 'connected, edge-weighted (un)directed graph'
- solved by: 'networkx.algorithms.tree.mst.minimum_spanning_tree, 'graph_tool.topology.min_spanning_tree'

## Second-best minimum spanning tree
- book: 'Introduction to Algorithms'
- solution need not be unique
- variant of: 'Minimum spanning tree'

## Bottleneck spanning tree
- book: 'Introduction to Algorithms'
- variant of: 'Minimum spanning tree'
- a 'minimum spanning tree' is a 'bottleneck spanning tree'

## Strongly connected component
- https://en.wikipedia.org/wiki/Strongly_connected_component
- used for 'Dulmage–Mendelsohn decomposition'
- book: 'Introduction to Algorithms'
- domain: 'Graph theory'
- based on: 'Directed graph'

## Eulerian path
- https://en.wikipedia.org/wiki/Eulerian_path
- found by: 'Hierholzer's algorithm'
- based on: 'Finite graph'

## Gröbner basis
- https://en.wikipedia.org/wiki/Gr%C3%B6bner_basis
- found by: 'Buchberger's algorithm', 'Faugère's F4 algorithm'

## Eigenvalues and eigenvectors
- https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
- based on: 'Linear map'

## Maximal matching
- https://en.wikipedia.org/wiki/Matching_(graph_theory)#Maximal_matchings
- https://brilliant.org/wiki/matching/#definitions-and-terminology
- time complexity: linear
- based on: 'Graph'

## Maximum matching
- the 'Maximal matching' with the maximum number of edges
- https://brilliant.org/wiki/matching/#definitions-and-terminology
- solves: 'Assignment problem' on 'weighted bipartite graphs'
- time complexity: polynomial
- based on: 'Graph'

## Minimum maximal matching
- https://en.wikipedia.org/wiki/Matching_(graph_theory)
- no polynomial-time algorithm is known
- solved approximately by: 'networkx.algorithms.approximation.matching.min_maximal_matching'
- based on: 'Graph'

## Longest increasing subsequence
- https://en.wikipedia.org/wiki/Longest_increasing_subsequence
- domain: 'Combinatorics'
- properties: 'optimal substructure'
- based on: 'Sequence'

## Greatest common divisor
- https://en.wikipedia.org/wiki/Greatest_common_divisor
- book: 'Introduction to Algorithms'
- domain: 'Number theory'
- based on: 'pair of integers'

## Maximum common edge subgraph
- https://en.wikipedia.org/wiki/Maximum_common_edge_subgraph
- hardness: NP-complete
- generalization of: 'Subgraph isomorphism problem'
- domain: 'Graph theory'
- based on: 'pair of graphs'

## Maximum common induced subgraph
- https://en.wikipedia.org/wiki/Maximum_common_induced_subgraph
- hardness: NP-complete
- generalization of: 'Induced subgraph isomorphism problem'
- applications: 'Cheminformatics', 'Pharmacophore'
- domain: 'Graph theory'
- based on: 'pair of graphs'

## Singular value decomposition
- https://en.wikipedia.org/wiki/Singular_value_decomposition
- implemented in: 'scipy.linalg.svd', 'surprise.prediction_algorithms.matrix_factorization.SVD'
- applications: 'Matrix decomposition', 'Matrix approximation', 'Signal processing', 'Statistics'
- used for: 'Pseudoinverse', 'Kabsch algorithm'
- solves: 'Total least squares problem'
- domain: 'Linear algebra'

## Cholesky decomposition
- https://en.wikipedia.org/wiki/Cholesky_decomposition
- implemented in: 'numpy.linalg.cholesky, scipy.linalg.cholesky', 'LAPACK'
- applications: 'Matrix decomposition', 'Matrix inversion', 'Non-linear optimization', 'Monte Carlo simulation'
- solves: 'Linear least squares problem'
- used for: 'Kalman filter'
- domain: 'Linear algebra'

## LU decomposition
- https://en.wikipedia.org/wiki/LU_decomposition
- applications: 'Matrix decomposition', 'Matrix inversion', 'System of linear equations', 'Determinant'
- implemented in: 'scipy.linalg.lu'
