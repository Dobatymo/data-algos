# Abstract data type

## list / array
- see: implementation/array

## linked list
- see: implementation/linked list

## sets
- list of unique elements
- similar to mathematical set
- offer fast in collection checks

### unordered set (interface)
- usually implemented as hash sets

### ordered set (interface)
- usually implemented as binary search trees
- could be implemented as deterministic acyclic finite state acceptor

## maps / associative arrays
- used to map a unique key to a value (a phone number to a name). Can only be used for exact matches. ie. cannot find all phone numbers which differ only in one digit.

### unordered map (interface)
- usually implemented as hash table

### ordered map (interface)
- usually implemented as binary search trees

## deque (double ended queue)
- usually implemeted as array with pointers to smaller arrays

## priority queue
- usually implemented as heap

# Data structures

## array (certain implementation) (list, tuple in python, vector in c++)
- continuous list of elements in memory. fast to iterate, fast to access by index. slow to find by value, slow to insert/delete within the array (as memory always need to be continuous, they need to be reallocated)
- usually fast to append/delete at the end or beginning (if there is free memory and depending on exact implementation).

## linked list
- single elements in memory which contain a pointer to the next element in the sequence
- double linked list also contain pointers back to the previous element int the sequence (XOR linked list is a clever optimization)
- insert and delete can be constant time if pointers are already found. iteration is slow. access by index is not possible, search by value in linear.

## hash table
- used to implement maps: dictionary in python, unordered_map in c++
- probably the most important data structure in python.
- has basically perfect complexity for a good hash function. (Minimal) perfect hashing can be used if the keys are known in advance.
- ordering depends on implementation (python 3.7 garuantees preservation of insertion order, whereas c++ as the name says does not define any ordering)
- even though hashing is constant in time, it might still be slow. also the hash consumes space
- hash tables usually have set characteristics (ie. the can test in average constant time if an item is the table or not)

### complexity, https://en.wikipedia.org/wiki/Hash_table
Average	Worst case
Space		O(n)	O(n) # often a lot of padding space is needed...
Search		O(1)	O(n)
Insert		O(1)	O(n)
Delete		O(1)	O(n)

## Binary Search Tree

## Fibonacci heap
- implements a priority queue

## Binary heap
- implements a priority queue
- variant implemented in Python heapq, C++ make_heap, push_heap and pop_heap.

## Trie / digital tree
- tree structure
- set and map characteristics
- keys are sequences (eg. strings)
- allow for prefix search (eg. find all strings that start with 'a', or find the longest prefix of 'asd')
- if implemented with hashmaps, indexing by key can be done in O(sequence-length) independent of tree size
- not very space efficient. common prefixed only have to be stored once, but pointers to next element of sequence uses more memory than what is saved.
- for a more space efficient data structure see MAFST

## Radix tree
- is a binary trie

## Red-black tree
- height-balanced binary tree
- better for insert/delete (compared to AVL)
- used to implement c++ map, java TreeMap
- used by MySQL

## AVL trees
- height-balanced binary tree
- stricter balanced and thus better for lookup (compared to Red-black)

## treap
- randomly ordered binary tree
- log(N) lookup even for insertion of items in non-random order
- heap like feature
- used to implement dictionary in LEDA

## BK-tree
- https://en.wikipedia.org/wiki/BK-tree
- is a: Space-partitioning tree, Metric tree
- applications: approximate string matching

## k-d tree
- is a: Space-partitioning tree
- applications: range searching, nearest neighbor search, kernel density estimation
- for high dimensions should be: N >> 2^k, where N is the number of nodes and k is the number of dimensions
- solves 'Recursive partitioning', 'Klee's measure problem', 'Guillotine problem'
- implemented in python 'scipy.spatial.KDTree', 'sklearn.neighbors.KDTree'

## Range tree
- https://en.wikipedia.org/wiki/Range_tree
- applications: range searching

## B+ tree
- https://en.wikipedia.org/wiki/B%2B_tree
- applications: filesystems, range searching, block-oriented data retrieval
- k-ary tree
- used by: Relational database management systems like Microsoft SQL Server, Key–value database management systems like CouchDB

## Iliffe vector
- https://en.wikipedia.org/wiki/Iliffe_vector
- used to implement multi-dimensional arrays

## Dope vector
- used to implement arrays

## van Emde Boas Trees
- Multiway tree
- implement ordered maps with integer keys
- implement priority queues
- see 'Integer sorting'

## Skip list
- probabilistic data structure
- basically binary trees converted to linked lists with additional information
- allows for fast search of sorted sequences
- implemented in Lucence
- applications: Moving median

## DAFSA (deterministic acyclic finite state acceptor)
- used to implement ordered sets

## MAFSA (minimal acyclic finite state automata)
- optimal DAFSA
- space optimized version of tries, with missing map characteristics
- allow for prefix (and possibly suffix) search
- more space efficient than tries as common prefixes and suffixes are only stored once and thus the number of pointers is reduced as well
- for a version with map characteristics see MAFST

## MAFST (minimal acyclic finite state transducer)
- MAFSA with map characteristics
- association of keys with values reduces lookup time from O(sequence-length) to O(sequence-length*log(tree size))???

## B-tree
- used to implement lots of databases and filesystems
- self-balancing
- non-binary

## SS-Tree (Similarity search tree)
- used for similarity indexing / nearest neighbors queries in high dimensional vector spaces

## Cover tree
- paper: "Cover Trees for Nearest Neighbor"
- https://en.wikipedia.org/wiki/Cover_tree
- applications: nearest neighbor search
- is a: Metric tree

## M-tree
- https://en.wikipedia.org/wiki/M-tree
- better disk storage characteristics (because shallower) than 'Ball tree'
- uses: nearest neighbor search
- is a: Space-partitioning tree, Metric tree

## Vantage-point tree
- https://en.wikipedia.org/wiki/Vantage-point_tree
- is a: Space-partitioning tree, Metric tree
- specialisation of: Multi-vantage-point tree

## Ball tree
- https://en.wikipedia.org/wiki/Ball_tree
- is a: Space-partitioning tree, Metric tree, Binary tree
- applications: nearest neighbor search, kernel density estimation, N-point correlation function calculations, generalized N-body Problems.
- specialisation of: M-Tree
- similar: Vantage-point tree
- implemented in: 'sklearn.neighbors.BallTree'
- algorithms for construction: 'Five Balltree Construction Algorithms'

## R-tree
- https://en.wikipedia.org/wiki/R-tree
- applications: range searching, nearest neighbor search

## Generalized suffix tree
- https://en.wikipedia.org/wiki/Generalized_suffix_tree
- build using 'Ukkonen's algorithm' or 'McCreight's algorithm'

## Disjoint-set data structure
- https://en.wikipedia.org/wiki/Disjoint-set_data_structure
- Multiway tree
- applications: connected components of an undirected graph
- implemented in boost::graph::incremental_components
- used for 'Kruskal's algorithm'

## soft heap
- https://en.wikipedia.org/wiki/Soft_heap
- approximate priority queue

# Index data structures

## Inverted index
- https://en.wikipedia.org/wiki/Inverted_index
- used by ElasticSearch
- maps content/text to locations/documents
- Search engine indexing
- cf. Back-of-the-book index, Concordance
- applications: full-text search, sequence assembly

##### algorithms

## cache algorithms

- optimization
-- Unconstrained nonlinear without derivative (black box, direct search), for functions which are not continuous or differentiable
  * Random search (random sampling from hypersphere surrounding the current position using uniform distribution)
    - Randomized algorithm
  * Random optimization (random sampling from hypersphere surrounding the current position using normal distribution)
  * Luus–Jaakola
    - heuristic
  * Pattern search
  * Golden-section search (for Unimodal)
  * Interpolation methods
  * Line search (iterative approache to find a local minimum)
  * Nelder–Mead method
    - heuristic search
    - implemented in Mathematica NMinimize ("NelderMead", "DifferentialEvolution", "SimulatedAnnealing", "RandomSearch")
    - L-BFGS-B variant implemented in scipy.optimize.minimize
  * Successive parabolic interpolation (for continuous unimodal function)
  * genetic algorithm
  * differential evolution
  * Powell's method
    - modified variant implemented in scipy.optimize.minimize
  * Principal Axis (by Brent)
    - implemented in Mathematica FindMinimum(Method->'PrincipalAxis')
    - uses SVD

-- Unconstrained nonlinear with derivate
  * Trust region
  * Broyden–Fletcher–Goldfarb–Shanno algorithm (BFGS)
    - Quasi-Newton method
    - iterative method
    - implemented in scipy.optimize.minimize
    - implemented in Mathematica FindMinimum(Method->'QuasiNewton')
  * Limited-memory BFGS (L-BFGS)
    - Quasi-Newton method
    - BFGS variant for large systems with memory optimisations
    - implemented in Mathematica FindMinimum(Method->'QuasiNewton')
    - implemented in scipy.optimize.minimize
    - applications:
      - "the algorithm of choice" for fitting log-linear (MaxEnt) models and conditional random fields with l2-regularization.[wiki]
  * Davidon–Fletcher–Powell formula
    - Quasi-Newton method
    - superseded by the BFGS
  * Symmetric rank-one (SR1)
    - Quasi-Newton method
    - advantages for sparse or partially separable problems
  * Gauss–Newton
    - non-linear least squares only
  * Levenberg–Marquardt algorithm
    - non-linear least squares
    - iterative procedure
    - generic curve-fitting problems
    - finds local minimum
    - implemented in Mathematica FindMinimum(Method->'LevenbergMarquardt')
    - implemented in scipy.optimize.least_squares(method='lm')
    - variant of Gauss–Newton
  * Berndt–Hall–Hall–Hausman algorithm (BHHH)
  * Gradient descent (gradient, steepest descent)
    - stochastic approximation: Stochastic gradient descent
    - will converge to a global minimum if the function is convex
  * Nonlinear conjugate gradient method
    - implemented in Mathematica FindMinimum(Method->'ConjugateGradient')
    - implemented in scipy.optimize.minimize
  * Truncated Newton
    - implemented in scipy.optimize.minimize

-- unconstrained nonlinear with hessian
  * Newton's method in optimization
    - implemented in Mathematica FindMinimum(Method->'Newton')
    - Anytime algorithm

-- constrained nonlinear
  * Penalty method
  * Sequential quadratic programming (SQP)
    - Sequential Least SQuares Programming (SLSQP) implemented in scipy.optimize.minimize
  * Augmented Lagrangian method
  * Successive Linear Programming (SLP)
  * Interior-point method (aka Barrier method)
    - implemented in Mathematica FindMinimum(Method->'InteriorPoint') (only one for constrained optimization)

-- Metaheuristics (randomized search methods)
  * Evolutionary algorithm
    - for one variant see Genetic algorithm
  * Genetic algorithm (GA)
  * Local search
    - variants: Hill climbing, Tabu search, Simulated annealing
    - applications: vertex cover problem, travelling salesman problem, boolean satisfiability problem, nurse scheduling problem, k-medoid
  * Simulated annealing (SA)
    - applications: combinatorial optimization problems
    - approximate global optimization
  * Tabu search (TS)
    - combinatorial optimization problems
  * Ant colony optimization algorithms
    - applications: combinatorial optimization problems
    - see also: Swarm intelligence
    - works well on graphs with changing topologies

-- Combinatorial optimization (https://en.wikipedia.org/wiki/Combinatorial_optimization)
  * see also Metaheuristics

-- Stochastic optimization
  * Stochastic gradient descent (SGD)
    - variants: momentum, Averaging, AdaGrad, RMSProp, Adam (many implemented in Keras and/or Tensorflow)
  * Stochastic approximation
  * see also Metaheuristics
  * see book 'Clever Algorithms: Nature-Inspired Programming Recipes'

-- convex

-- nonlinear with NOT ENOUGH INFO
  * Newton conjugate gradient
    - implemented in scipy.optimize.minimize
  * Constrained Optimization BY Linear Approximation (COBYLA) algorithm
    - implemented in scipy.optimize.minimize
  * Trust-region dogleg
    - implemented in scipy.optimize.minimize
  * Linear Programming
    - implemented in Mathematica FindMinimum(Method->'LinearProgramming')
    - is this SLP?

-- Linear least squares
  * "Direct" and "IterativeRefinement", and for sparse arrays "Direct" and "Krylov"

-- important practical theorems
  * no free lunch theorem

# named algorithms:

## Median of medians
- https://en.wikipedia.org/wiki/Median_of_medians
- selection algorithm

## Introselect
- https://en.wikipedia.org/wiki/Introselect
- selection algorithm
- implemented in C++ std::nth_element

## Floyd–Rivest algorithm
- https://en.wikipedia.org/wiki/Floyd%E2%80%93Rivest_algorithm
- selection algorithm
- divide and conquer algorithm

## Order statistic tree
- variant of  binary search tree
- additional interface: find the i'th smallest element stored in the tree, find the rank of element x in the tree, i.e. its index in the sorted list of elements of the tree

## Dijkstra's algorithm
- https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
- solves 'shortest path problem' for non-negative weights in directed/undirected graphs in O(v^2) where v is the number of vertices
- graph algorithm
- variant implementation with Fibonacci heaps runs in O(e * v*log v) where e and v are the number of edges and vertices resp.
- implemented in python scipy.sparse.csgraph.shortest_path(method='D')
- Fibonacci implementation is the asymptotically the fastest known single-source shortest-path algorithm for arbitrary directed graphs with unbounded non-negative weights.

## Bellman–Ford algorithm
- https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm
- graph algorithm
- solves variant of the 'shortest path problem' for real-valued edge weights in directed graph in O(v*e) where v and e are the number of vertices and edges respectively.
- negative cycles are detected
- implemented in python scipy.sparse.csgraph.shortest_path(method='BF')

## Johnson's algorithm
- https://en.wikipedia.org/wiki/Johnson%27s_algorithm
- graph algorithm
- solves 'All-pairs shortest paths problem' for real-valued weights in a directed graph in O(v^2 log v + v*e) where v and e are the number of vertices and edges
- negative cycles are not allowed
- implemented in python scipy.sparse.csgraph.shortest_path(method='J'), c++ boost::graph::johnson_all_pairs_shortest_paths
- combination of 'Bellman–Ford' and 'Dijkstra's algorithm'
- is faster than 'Floyd–Warshall algorithm' for sparse graphs

## Floyd–Warshall algorithm
- https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
- graph algorithm
- solves 'All-pairs shortest paths problem' for real-valued weights for directed/undirected graphs in O(v^3) where v is the number of vertices
- negative cycles are not allowed
- dynamic programming algorithm
- implemented in python scipy.sparse.csgraph.shortest_path(method='FW'), c++ boost::graph::floyd_warshall_all_pairs_shortest_paths
- is faster then 'Johnson's algorithm' for dense graphs

## A* search algorithm
- generalization of 'Dijkstra's algorithm'
- heuristic search
- informed search algorithm (best-first search)
- usually implemented using 'priority queue'
- applications: pathfinding, parsing using stochastic grammars in NLP
- dynamic programming

## Linear search
- find element in any ordered seqeunce in O(i) time where i is the index of the element in the sequence
- https://en.wikipedia.org/wiki/Linear_search
- O(1) for list with geometric distributed values
- implemented in c++ std::find (impl. dependent), python list.index

## Binary search algorithm
- find element in sorted finite list in O(log n) time where n is the number of elements in list
- https://en.wikipedia.org/wiki/Binary_search_algorithm
- variants: Exponential search
- implemented in std::binary_search, python bisect

## Naïve string-search algorithm
- find string in string in O(n+m) average time and O(n*m) worst case, where n and m are strings to be search for, resp. in.
- implemented in std::search (impl. dependent), python list.index

## Exponential search
- find element in sorted infinite list in O(log i) time where i is the position of the element in the list
- https://en.wikipedia.org/wiki/Exponential_search

## Funnelsort
- comparison-based sorting algorithm
- cache-oblivious algorithm
- external memory algorithm

## Quicksort
- sorting algorithm
- divide and conquer algorithm
- unstable sort
- in-place algorithm

## Cache-oblivious distribution sort
- comparison-based sorting algorithm
- cache-oblivious algorithm

## Wagner–Fischer algorithm
- dynamic programming
- calculate Levenshtein distance in O(n*m) complexity where n and m are the respective string lenths
- optimal time complexity for problem proven to be O(n^2), so this algorithm is pretty much optimal
- space complexity of O(n*m) could be reduced to O(n+m)

## Aho–Corasick algorithm
- multiple *string searching*
- implemented in: original fgrep
- (pre)constructs finite-state machine from set of search strings
- applications: virus signature detection
- paper 'Efficient string matching: An aid to bibliographic search'
- classification 'constructed search engine', 'match prefix first', 'one-pass'
- https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm
- https://xlinux.nist.gov/dads/HTML/ahoCorasick.html
- shows better results than 'Commentz-Walter' for peptide identification according to 'Commentz-Walter: Any Better Than Aho-Corasick For Peptide Identification?' and for biological sequences according to 'A Comparative Study On String Matching Algorithms Of Biological Sequences'

## Commentz-Walter algorithm
- multiple *string searching*
- classification 'match suffix first'
- https://en.wikipedia.org/wiki/Commentz-Walter_algorithm
- variant implemented in grep

## Boyer–Moore string-search algorithm
- single *string searching*
- implemented in grep, c++ std::boyer_moore_searcher
- variant implemented in python string class
- better for large alphabets like text than Knuth–Morris–Pratt
- paper 'A Fast String Searching Algorithm'
- https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_string-search_algorithm

## Knuth–Morris–Pratt algorithm
- single *string searching*
- implemented in grep
- better for small alphabets like DNA than Boyer–Moore

## Beam search
- heuristic search algorithm, greedy algorithm
- optimization of best-first search
- greedy version of breadth-first search
- applications: machine translation, speech recognition
- approximate solution

## Trémaux's algorithm
- local *Maze solving* algorithm

## Dead-end filling
- global *Maze solving* algorithm

## Wall follower (left-hand rule / right-hand rule)
- local *Maze solving* algorithm for simply connected mazes

## Held–Karp algorithm
- dynamic programming algorithm
- solves 'Travelling salesman problem'
- https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm

## Christofides algorithm
- solves 'Travelling salesman problem' approximately for metrix distances

## Bélády's algorithm / the clairvoyant algorithm
- caching algorithm

## Kruskal's algorithm
- https://en.wikipedia.org/wiki/Kruskal%27s_algorithm
- finds 'Minimum spanning tree'
- greedy algorithm
- implemented in c++ boost::graph::kruskal_minimum_spanning_tree
- see book: 'Introduction to Algorithms'

## Prim's algorithm
- https://en.wikipedia.org/wiki/Prim%27s_algorithm
- finds 'Minimum spanning tree'
- greedy algorithm
- implemented in c++ boost::graph::prim_minimum_spanning_tree
- time complexity depends on used data structures
- see book: 'Introduction to Algorithms'

## Matching pursuit
- https://en.wikipedia.org/wiki/Matching_pursuit
- does 'Sparse approximation'
- greedy algorithm

## Kahn's algorithm
- applications: 'topological sorting'

## Depth-first search
- applications: 'topological sorting', 'Strongly connected component'

## HyperLogLog++
- https://en.wikipedia.org/wiki/HyperLogLog
- solves 'Count-distinct problem' approximately
- implemented by Lucence

## Tarjan's strongly connected components algorithm
- https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
- computes 'Strongly connected component'

## Default algorithm for Huffman Tree
- uses for Huffman coding
- greedy algorithm
- uses 'priority queue'

## t-digest
- paper 'COMPUTING EXTREMELY ACCURATE QUANTILES USING t-DIGESTS'
- Q-digest
- approximates percentiles, 
- distributed algorithm

## Chazelle algorithm for the minimum spanning tree
- paper 'A minimum spanning tree algorithm with inverse-Ackermann type complexity'
- finds 'Minimum spanning tree'
- uses 'soft heaps'

## Fan and Su algortihm for multiple pattern match
- paper 'An Efficient Algorithm for Matching Multiple Patterns'
- "combines the concept of deterministic finite state automata (DFSA) and Boyer-Moore’s algorithm"

## Hu and Shing algortihm for matrix chain products
- paper 'Computation of Matrix Chain Products'
- solves 'Matrix chain multiplication' in O(n log n) where n is the number of matrices

## Long multiplication
- is a: 'multiplication algorithm'
- applications: 'integer multiplication'
- complexity: O(n^2)
- naive algorithm

## Karatsuba algorithm
- https://en.wikipedia.org/wiki/Karatsuba_algorithm
- is a: 'multiplication algorithm'
- applications: 'integer multiplication'
- specialisation of: 'Toom–Cook multiplication'
- complexity: O(n^log_2(3))
- practically outperforms 'Long multiplication' for n > 10*int bits [https://gmplib.org/manual/Multiplication-Algorithms.html]

## Toom–Cook multiplication
- https://en.wikipedia.org/wiki/Toom%E2%80%93Cook_multiplication
- is a: 'multiplication algorithm'
- complexity O(n^log_3(5))
- practically outperforms 'Karatsuba algorithm' for n > 10,000 digits []

## Schönhage–Strassen algorithm
- https://en.wikipedia.org/wiki/Sch%C3%B6nhage%E2%80%93Strassen_algorithm
- is a: 'multiplication algorithm'
- applications: 'integer multiplication'
- complexity: O(n log(n) log(log(n)))
- practically outperforms 'Toom–Cook multiplication' for n > 10,000 digits [https://gmplib.org/manual/Multiplication-Algorithms.html]

## Euclidean algorithm
- https://en.wikipedia.org/wiki/Euclidean_algorithm
- solves 'Greatest common divisor'

## Binary GCD algorithm, Stein's algorithm
- https://en.wikipedia.org/wiki/Binary_GCD_algorithm
- solves 'Greatest common divisor'

## Extended Euclidean algorithm
- https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
- is a: certifying algorithm
- solves 'Greatest common divisor'

## Lucas–Kanade method
- https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method
- applications: 'Optical flow estimation', 'aperture problem'
- local, sparse
- implemented in: 'opencv::calcOpticalFlowPyrLK', 'opencv::CalcOpticalFlowLK' (obsolete)

## Horn–Schunck method
- https://en.wikipedia.org/wiki/Horn%E2%80%93Schunck_method
- paper: 'Determining Optical Flow'
- applications: 'Optical flow estimation', 'aperture problem'
- implemented in: ''opencv::CalcOpticalFlowHS' (obsolete, should be replaced with calcOpticalFlowPyrLK or calcOpticalFlowFarneback according to opencv docs)
- global

## Gunnar-Farneback algorithm
- paper: 'Two-frame motion estimation based on polynomial expansion'
- applications: 'Optical flow estimation'
- implemented in: 'opencv::calcOpticalFlowFarneback'
- dense

## SimpleFlow algorithm
- paper: 'SimpleFlow: A Non-iterative, Sublinear Optical Flow Algorithm'
- implemented in: 'opencv::calcOpticalFlowSF'

## Kadane's algorithm
- https://en.wikipedia.org/wiki/Maximum_subarray_problem#Kadane's_algorithm
- applications: 'Maximum subarray problem'
- runtime complexity: O(n)
- uses: 'optimal substructure', 'dynamic programming'

## Mean shift
- https://en.wikipedia.org/wiki/Mean_shift
- is a: 'mode-seeking algorithm'
- applications: 'cluster analysis', 'visual tracking', 'image smoothing'
- basis for: 'Camshift'

## Scale-invariant feature transform
- https://en.wikipedia.org/wiki/Scale-invariant_feature_transform
- is a: 'feature detection algorithm'
- applications: 'object recognition', 'robotic mapping and navigation', 'image stitching', '3D modeling', 'gesture recognition', 'video tracking'
- patented

## Marching squares
- https://en.wikipedia.org/wiki/Marching_squares
- domain: 'computer graphics', 'cartography'
- application: 'contour finding'

## Lempel–Ziv–Welch
- https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch
- application: 'Lossless compression'
- was patented

## General number field sieve
- https://en.wikipedia.org/wiki/General_number_field_sieve
- applications: 'Integer factorization'

## Shor's algorithm
- https://en.wikipedia.org/wiki/Shor%27s_algorithm
- is a: 'Quantum algorithm'
- applications: 'Integer factorization'

## RSA
- https://en.wikipedia.org/wiki/RSA_(cryptosystem)
- applications: 'Public-key cryptography'
- depends on computational hardness of: 'RSA problem'
- patented

## Adjusted winner procedure
- https://en.wikipedia.org/wiki/Adjusted_winner_procedure
- applications: 'Envy-free item assignment'
- patented

## Karmarkar's algorithm
- https://en.wikipedia.org/wiki/Karmarkar%27s_algorithm
- was patented

# common problems with applications

---# interesting problems with clever solutions
- optimal binary search tree (cf. Shannon coding)

## RSA problem
- https://en.wikipedia.org/wiki/RSA_problem
- see: 'Integer factorization'

## Integer factorization
- https://en.wikipedia.org/wiki/Integer_factorization
- applications: 'cryptography'
- domain: 'number theory'

## Envy-free item assignment
- https://en.wikipedia.org/wiki/Envy-free_item_assignment
- type of: 'Fair item assignment', 'Fair division'

## Strongly connected component
- https://en.wikipedia.org/wiki/Strongly_connected_component
- used for 'Dulmage–Mendelsohn decomposition'
- see book: 'Introduction to Algorithms'
- domain: graph theory

## Greatest common divisor
- https://en.wikipedia.org/wiki/Greatest_common_divisor
- domain: number theory

## Topological sorting
- https://en.wikipedia.org/wiki/Topological_sorting
- implemented in posix tsort.
- only possible on 'directed acyclic graph'

## Travelling salesman problem
- https://en.wikipedia.org/wiki/Travelling_salesman_problem
- solved by 'Concorde TSP Solver' application

## Minimum spanning tree
- https://en.wikipedia.org/wiki/Minimum_spanning_tree
- http://algorist.com/problems/Minimum_Spanning_Tree.html
- see book: 'Introduction to Algorithms'
- solved by 'Kruskal's algorithm', 'Prim's algorithm'
- unique solution

## Second-best minimum spanning tree
- see book: 'Introduction to Algorithms'
- solution need not be unique
- variant of 'Minimum spanning tree'

## Bottleneck spanning tree
- variant of 'Minimum spanning tree'
- see book: 'Introduction to Algorithms'
- a 'minimum spanning tree' is a 'bottleneck spanning tree'

## spanning-tree verification
- related to 'Minimum spanning tree'
- see book: 'Introduction to Algorithms'

## Matrix chain multiplication
- https://en.wikipedia.org/wiki/Matrix_chain_multiplication
- optimization problem
- solved by 'Hu and Shing'

## Count-distinct problem
- https://en.wikipedia.org/wiki/Count-distinct_problem

## Single-source shortest path problem
- https://en.wikipedia.org/wiki/Shortest_path_problem
- find shortest path in graph so that the sum of edge weights is minimized
- optimization problem
- solved by 'Breadth-first search' for unweighted graphs
- solved by 'Dijkstra's algorithm' for directed/undirected graphs and positive weights
- solved by 'Bellman–Ford algorithm' for directed graphs with arbitrary weights
- see book: 'Introduction to Algorithms'
- optimal substructure property
- domain: 'graph theory'

## Single-pair shortest path problem
- no algorithms with better worst time complexity than for 'Single-source shortest path problem' are know (which is a generalization)
- domain: 'graph theory'

## All-pairs shortest paths problem
- https://en.wikipedia.org/wiki/Shortest_path_problem#All-pairs_shortest_paths
- finds the shortest path for all pairs of vectices in a graph
- solved by 'Floyd–Warshall algorithm', 'Johnson's algorithm'
- domain: 'graph theory'

## approximate string matching
- paper "Fast Approximate String Matching in a Dictionary"
- applications: spell checking, nucleotide sequence matching

## Lowest common ancestor (LCA)
- https://en.wikipedia.org/wiki/Lowest_common_ancestor
- domain: 'graph theory'

## Longest common substring problem
- cf. 'Longest common subsequence problem'
- https://en.wikipedia.org/wiki/Longest_common_substring_problem
- solutions: Generalized suffix tree
- domain: 'combinatorics'

## Longest common subsequence problem
- cf. 'Longest common substring problem'
- https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
- Hunt–McIlroy algorithm
- applications: version control systems, wiki engines, and molecular phylogenetics
- domain: 'combinatorics'

## Shortest common supersequence problem
- https://en.wikipedia.org/wiki/Shortest_common_supersequence_problem
- applications: DNA sequencing
- domain: 'combinatorics'

## Shortest common superstring problem
- https://en.wikipedia.org/wiki/Shortest_common_supersequence_problem#Shortest_common_superstring
- applications: sparse matrix compression
- domain: 'combinatorics'

## Maximum subarray problem
- https://en.wikipedia.org/wiki/Maximum_subarray_problem
- applications: 'genomic sequence analysis', 'computer vision', 'data mining'
- solved by: 'Kadane's algorithm'

## Optical flow
- https://en.wikipedia.org/wiki/Optical_flow
- applications: 'Motion estimation', 'video compression', 'object detection', 'object tracking', 'image dominant plane extraction', 'movement detection', 'robot navigation , 'visual odometry'
- domain: 'machine vision', 'computer vision'
