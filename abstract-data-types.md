# Abstract data types

## Collection
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
- offer fast in collection checks

### unordered set (interface)
- usually implemented as: 'hash table'
- implemented in: 'std::unordered_set', 'Python set', 'Java Set'

### ordered set (interface)
- usually implemented as: 'binary search trees'
- could be implemented as deterministic acyclic finite state acceptor
- implemented in: 'std::set', 'Java SortedSet'

## Map
- also called: 'Associative array'
- https://en.wikipedia.org/wiki/Associative_array
- used to map a unique key to a value (a phone number to a name). Can only be used for exact matches. ie. cannot find all phone numbers which differ only in one digit.

### unordered map (interface)
- usually implemented as: 'Hash table'

### ordered map (interface)
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
- found by: 'Karger's algorithm'
- based on: 'Graph'

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
- book: 'Handbook of Discrete and Computational Geometry'
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
- book: 'Handbook of Discrete and Computational Geometry'
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
- https://en.wikipedia.org/wiki/Minimum_spanning_tree
- http://algorist.com/problems/Minimum_Spanning_Tree.html
- book: 'Introduction to Algorithms'
- found by: 'Kruskal's algorithm', 'Prim's algorithm'
- unique solution
- applications: 'Network design', 'Image segmentation', 'Cluster analysis'
- based on: 'connected, edge-weighted (un)directed graph'

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
