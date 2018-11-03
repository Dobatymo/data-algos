# online books
- https://bradfieldcs.com/algos/
- http://interactivepython.org/RkmcZ/courselib/static/pythonds/index.html

# Abstract data type

## Array / List
- https://en.wikipedia.org/wiki/List_(abstract_data_type)
- implemented as: 'Array', 'Linked list'

## Stack
- https://en.wikipedia.org/wiki/Stack_(abstract_data_type)
- usually implemented as: 'Array', 'Singly linked list'

## Set
- https://en.wikipedia.org/wiki/Set_(abstract_data_type)
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

## Map / Associative array
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

# Data structures

## Array
- https://en.wikipedia.org/wiki/Array_data_structure
- implemented in: 'C++ std::vector', 'Python list', 'Python tuple'
- continuous list of elements in memory. fast to iterate, fast to access by index. slow to find by value, slow to insert/delete within the array (as memory always need to be continuous, they need to be reallocated)
- usually fast to append/delete at the end or beginning (if there is free memory and depending on exact implementation)

## Linked list
- https://en.wikipedia.org/wiki/Linked_list
- single elements in memory which contain a pointer to the next element in the sequence
- double linked list also contain pointers back to the previous element int the sequence (XOR linked list is a clever optimization)
- insert and delete can be constant time if pointers are already found. iteration is slow. access by index is not possible, search by value in linear.
- implemented in: 'C++ std::list'

## Hash table
- https://en.wikipedia.org/wiki/Hash_table
- used to implement maps: 'Python dict', 'std::unordered_map'
- probably the most important data structure in Python
- has basically perfect complexity for a good hash function. (Minimal) perfect hashing can be used if the keys are known in advance.
- ordering depends on implementation (python 3.7 garuantees preservation of insertion order, whereas C++ as the name says does not define any ordering)
- even though hashing is constant in time, it might still be slow. also the hash consumes space
- hash tables usually have set characteristics (ie. the can test in average constant time if an item is the table or not)

### complexity, https://en.wikipedia.org/wiki/Hash_table
Average	Worst case
Space		O(n)	O(n) # often a lot of padding space is needed...
Search		O(1)	O(n)
Insert		O(1)	O(n)
Delete		O(1)	O(n)

## Binary search tree
- https://en.wikipedia.org/wiki/Binary_search_tree
- implements set or map
- is a: 'rooted binary tree'
- variants: 'Huffman tree'

## Fibonacci heap
- https://en.wikipedia.org/wiki/Fibonacci_heap
- implements a: 'Priority queue'

## Binary heap
- https://en.wikipedia.org/wiki/Binary_heap
- implements a: 'Priority queue'
- variant implemented in: 'Python heapq', 'C++ std::make_heap, push_heap and pop_heap'

## Trie / digital tree
- https://en.wikipedia.org/wiki/Trie
- tree structure
- set and map characteristics
- keys are sequences (eg. strings)
- allow for prefix search (eg. find all strings that start with 'a', or find the longest prefix of 'asd')
- if implemented with hashmaps, indexing by key can be done in O(sequence-length) independent of tree size
- not very space efficient. common prefixed only have to be stored once, but pointers to next element of sequence uses more memory than what is saved.
- for a more space efficient data structure see MAFST

## Radix tree
- https://en.wikipedia.org/wiki/Radix_tree
- is a: 'Binary trie'

## Red-black tree
- https://en.wikipedia.org/wiki/Red%E2%80%93black_tree
- height-balanced binary tree
- better for insert/delete (compared to AVL)
- used to implement c++ map, java TreeMap
- used by MySQL

## AVL tree
- https://en.wikipedia.org/wiki/AVL_tree
- height-balanced binary tree
- stricter balanced and thus better for lookup (compared to Red-black)

## treap
- https://en.wikipedia.org/wiki/Treap
- randomly ordered binary tree
- log(N) lookup even for insertion of items in non-random order
- heap like feature
- used to implement dictionary in LEDA

## BK-tree
- https://en.wikipedia.org/wiki/BK-tree
- is a: Space-partitioning tree, Metric tree
- applications: approximate string matching

## k-d tree
- https://en.wikipedia.org/wiki/K-d_tree
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
- https://en.wikipedia.org/wiki/Dope_vector
- used to implement arrays

## van Emde Boas Trees
- https://en.wikipedia.org/wiki/Van_Emde_Boas_tree
- Multiway tree
- implement ordered maps with integer keys
- implement priority queues
- see 'Integer sorting'

## Skip list
- https://en.wikipedia.org/wiki/Skip_list
- probabilistic data structure
- basically binary trees converted to linked lists with additional information
- allows for fast search of sorted sequences
- implemented in Lucence
- applications: Moving median

## DAFSA (deterministic acyclic finite state acceptor)
- https://en.wikipedia.org/wiki/Deterministic_acyclic_finite_state_automaton
- used to implement ordered sets

## MAFSA (minimal acyclic finite state automaton)
- also called: 'minimal acyclic finite state acceptor'
- https://en.wikipedia.org/wiki/Deterministic_acyclic_finite_state_automaton
- https://blog.burntsushi.net/transducers/
- optimal DAFSA
- space optimized version of tries, with missing map characteristics
- allow for prefix (and possibly suffix) search
- more space efficient than tries as common prefixes and suffixes are only stored once and thus the number of pointers is reduced as well
- for a version with map characteristics see MAFST

## MAFST (minimal acyclic finite state transducer)
- https://blog.burntsushi.net/transducers/
- MAFSA with map characteristics
- association of keys with values reduces lookup time from O(sequence-length) to O(sequence-length*log(tree size))???

## B-tree
- https://en.wikipedia.org/wiki/B-tree
- used to implement lots of databases and filesystems
- self-balancing
- non-binary

## SS-Tree (Similarity search tree)
- paper: 'Similarity indexing with the SS-tree'
- applications: 'Similarity indexing', 'Nearest neighbor search'

## Cover tree
- https://en.wikipedia.org/wiki/Cover_tree
- paper: "Cover Trees for Nearest Neighbor"
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

## Soft heap
- https://en.wikipedia.org/wiki/Soft_heap
- approximate priority queue

## Bloom filter
- https://en.wikipedia.org/wiki/Bloom_filter
- probabilistic data structure
- implements: set
- applications: caching strategies, database query optimization, rate-limiting, data synchronization, chemical structure searching

## Optimal binary search tree
- https://en.wikipedia.org/wiki/Optimal_binary_search_tree
- is a: 'Binary search tree'

## Order statistic tree
- https://en.wikipedia.org/wiki/Order_statistic_tree
- variant of: 'Binary search tree'
- additional interface: find the i'th smallest element stored in the tree, find the rank of element x in the tree, i.e. its index in the sorted list of elements of the tree

# Index data structures

## Inverted index
- https://en.wikipedia.org/wiki/Inverted_index
- used by ElasticSearch
- maps content/text to locations/documents
- Search engine indexing
- cf. Back-of-the-book index, Concordance
- applications: full-text search, sequence assembly

# Algorithms

## Breadth-first search
- https://en.wikipedia.org/wiki/Breadth-first_search

## Depth-first search
- https://en.wikipedia.org/wiki/Depth-first_search

## Median of medians
- https://en.wikipedia.org/wiki/Median_of_medians
- selection algorithm

## Introselect
- https://en.wikipedia.org/wiki/Introselect
- implemented in C++ std::nth_element
- is a: 'Selection algorithm'

## Floyd–Rivest algorithm
- https://en.wikipedia.org/wiki/Floyd%E2%80%93Rivest_algorithm
- is a: 'Selection algorithm', 'Divide and conquer algorithm'

## Quickselect
- https://en.wikipedia.org/wiki/Quickselect
- is a: 'Selection algorithm'

## Dijkstra's algorithm
- https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
- uses method: 'dynamic programming'
- solves 'shortest path problem' for non-negative weights in directed/undirected graphs in O(v^2) where v is the number of vertices
- is a: 'graph algorithm'
- variant implementation with Fibonacci heaps runs in O(e * v*log v) where e and v are the number of edges and vertices resp.
- implemented in: 'Python scipy.sparse.csgraph.shortest_path(method='D')'
- Fibonacci implementation is the asymptotically the fastest known single-source shortest-path algorithm for arbitrary directed graphs with unbounded non-negative weights.

## Bellman–Ford algorithm
- https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm
- is a: 'graph algorithm'
- solves variant of the 'shortest path problem' for real-valued edge weights in directed graph in O(v*e) where v and e are the number of vertices and edges respectively.
- negative cycles are detected
- implemented in: 'Python scipy.sparse.csgraph.shortest_path(method="BF")'

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
- uses method: 'dynamic programming'
- implemented in: 'python scipy.sparse.csgraph.shortest_path(method='FW')', 'c++ boost::graph::floyd_warshall_all_pairs_shortest_paths'
- is faster then 'Johnson's algorithm' for dense graphs

## A* search algorithm
- https://en.wikipedia.org/wiki/A*_search_algorithm
- generalization of 'Dijkstra's algorithm'
- heuristic search
- informed search algorithm (best-first search)
- usually implemented using: 'priority queue'
- applications: pathfinding, parsing using stochastic grammars in NLP
- uses method: 'dynamic programming'

## Linear search
- https://en.wikipedia.org/wiki/Linear_search
- find element in any sequence in O(i) time where i is the index of the element in the sequence
- works on: 'Linked list', 'Array', 'List'
- has an advantage when sequential access is fast compared to random access
- O(1) for list with geometric distributed values
- implemented in c++ std::find (impl. dependent), python list.index

## Binary search algorithm
- https://en.wikipedia.org/wiki/Binary_search_algorithm
- find element in sorted finite list in O(log n) time where n is the number of elements in list
- requires: 'Random access'
- variants: 'Exponential search'
- implemented in 'C++ std::binary_search', 'python bisect'

## Naïve string-search algorithm
- https://en.wikipedia.org/wiki/String-searching_algorithm#Na%C3%AFve_string_search
- find string in string in O(n+m) average time and O(n*m) worst case, where n and m are strings to be search for, resp. in.
- implemented in: 'C++ std::search (impl. dependent)', 'python list.index'

## Exponential search
- https://en.wikipedia.org/wiki/Exponential_search
- find element in sorted infinite list in O(log i) time where i is the position of the element in the list

## Funnelsort
- https://en.wikipedia.org/wiki/Funnelsort
- is a: 'cache-oblivious algorithm', 'external memory algorithm', 'Comparison-based sorting algorithm'

## Quicksort
- https://en.wikipedia.org/wiki/Quicksort
- is a: 'Sorting algorithm', 'Unstable sorting algorithm', 'In-place algorithm', 'Divide and conquer algorithm', 'Comparison-based sorting algorithm'
- best case time complexity: O(n log n)
- average case time complexity: O(n log n)
- worst case time complexity: O(n^2)
- space complexity: O(log n) auxiliary

## Merge Sort
- https://en.wikipedia.org/wiki/Merge_sort
- is a: 'Sorting algorithm', 'Stable sorting algorithm' (usually), 'Divide and conquer algorithm', 'Comparison-based sorting algorithm'
- implemented in: 'C++ std::stable_sort (usually)'
- good for sequential access, can work on 'singly linked lists', external sorting
- easily parallelizable

## Heapsort
- https://en.wikipedia.org/wiki/Heapsort
- is a: 'Sorting algorithm', 'In-place algorithm', 'Unstable sorting algorithm', 'Comparison-based sorting algorithm'
- solves: 'Partial sorting'
- best case time complexity: O(n log n)
- average case time complexity: O(n log n)
- worst case time complexity: O(n log n)
- space complexity: O(1)
- uses: 'max heap'
- not easily parallelizable
- can work on 'doubly linked lists'

## Timsort
- https://en.wikipedia.org/wiki/Timsort
- is a: 'Sorting algorithm', 'Stable sorting algorithm', 'Comparison-based sorting algorithm'
- implemented in: 'Python sorted', 'Android Java'

## Introsort
- https://en.wikipedia.org/wiki/Introsort
- is a: 'Sorting algorithm', 'Unstable sorting algorithm', 'Comparison-based sorting algorithm'
- implemented in: 'C++ STL std::sort (usually)', '.net sort'

## Cycle sort
- https://en.wikipedia.org/wiki/Cycle_sort
- is a: 'Sorting algorithm', 'In-place algorithm', 'Unstable sorting algorithm', 'Comparison-based sorting algorithm'
- solves: 'cycle decomposition'
- theoretically optimal in terms of the total number of writes to the original array
- used for sorting where writes are expensive
- applications: EEPROM
- time complexity: O(n^2)
- space complexity: O(1) auxiliary

## Patience sorting
- https://en.wikipedia.org/wiki/Patience_sorting
- is a: 'Sorting algorithm', 'Comparison-based sorting algorithm'
- finds: 'Longest increasing subsequence'
- applications: 'Process control'
- see also: 'Floyd's game'

## Fisher–Yates shuffle
- https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
- is a: 'Shuffling algorithm', 'In-place algorithm'
- unbiased

## Reservoir sampling
- https://en.wikipedia.org/wiki/Reservoir_sampling
- family of 'randomized algorithms'
- version of: 'Fisher–Yates shuffle'

## Cache-oblivious distribution sort
- https://en.wikipedia.org/wiki/Cache-oblivious_distribution_sort
- comparison-based sorting algorithm
- cache-oblivious algorithm

## Wagner–Fischer algorithm
- https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm
- uses method: 'dynamic programming'
- calculate Levenshtein distance in O(n*m) complexity where n and m are the respective string lenths
- optimal time complexity for problem proven to be O(n^2), so this algorithm is pretty much optimal
- space complexity of O(n*m) could be reduced to O(n+m)

## Aho–Corasick algorithm
- https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm
- https://xlinux.nist.gov/dads/HTML/ahoCorasick.html
- multiple *string searching*
- implemented in: original fgrep
- (pre)constructs finite-state machine from set of search strings
- applications: virus signature detection
- paper 'Efficient string matching: An aid to bibliographic search'
- classification 'constructed search engine', 'match prefix first', 'one-pass'
- shows better results than 'Commentz-Walter' for peptide identification according to 'Commentz-Walter: Any Better Than Aho-Corasick For Peptide Identification?' and for biological sequences according to 'A Comparative Study On String Matching Algorithms Of Biological Sequences'

## Commentz-Walter algorithm
- https://en.wikipedia.org/wiki/Commentz-Walter_algorithm
- multiple *string searching*
- classification 'match suffix first'
- variant implemented in grep

## Boyer–Moore string-search algorithm
- https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_string-search_algorithm
- single *string searching*
- implemented in grep, c++ std::boyer_moore_searcher
- variant implemented in python string class
- better for large alphabets like text than Knuth–Morris–Pratt
- paper 'A Fast String Searching Algorithm'

## Knuth–Morris–Pratt algorithm
- https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm
- single *string searching*
- implemented in grep
- better for small alphabets like DNA than Boyer–Moore

## Beam search
- https://en.wikipedia.org/wiki/Beam_search
- heuristic search algorithm, greedy algorithm
- optimization of best-first search
- greedy version of breadth-first search
- applications: machine translation, speech recognition
- approximate solution

## Hu–Tucker algorithm
- superseded by: 'Garsia–Wachs algorithm'

## Garsia–Wachs algorithm
- https://en.wikipedia.org/wiki/Garsia%E2%80%93Wachs_algorithm
- constructs: 'Optimal binary search tree' (special case)

## Knuth's optimal binary search tree algorithm
- https://en.wikipedia.org/wiki/Optimal_binary_search_tree#Knuth%27s_dynamic_programming_algorithm
- paper: Optimum binary search trees
- uses method: 'dynamic programming'
- constructs: 'Optimal binary search tree'
- time comlexity: O(n^2)

## Mehlhorn's nearly optimal binary search tree algorithm
- paper 'Nearly optimal binary search trees'
- constructs approximatly: 'Optimal binary search tree'
- time complexity: O(n)

## Trémaux's algorithm
- https://en.wikipedia.org/wiki/Maze_solving_algorithm#Tr%C3%A9maux%27s_algorithm
- local *Maze solving* algorithm

## Dead-end filling
- https://en.wikipedia.org/wiki/Maze_solving_algorithm#Dead-end_filling
- global *Maze solving* algorithm

## Wall follower (left-hand rule / right-hand rule)
- https://en.wikipedia.org/wiki/Maze_solving_algorithm#Wall_follower
- local *Maze solving* algorithm for simply connected mazes

## Held–Karp algorithm
- https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm
- uses method: 'dynamic programming'
- solves: 'Travelling salesman problem'

## Christofides algorithm
- https://en.wikipedia.org/wiki/Christofides_algorithm
- solves 'Travelling salesman problem' approximately for metric distances

## Push–relabel maximum flow algorithm
- https://en.wikipedia.org/wiki/Push–relabel_maximum_flow_algorithm
- solves: 'Maximum flow problem', 'Minimum-cost flow problem'

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

## Hierholzer's algorithm
- https://en.wikipedia.org/wiki/Eulerian_path#Hierholzer's_algorithm
- https://www.geeksforgeeks.org/hierholzers-algorithm-directed-graph/
- solves: 'Eulerian path'
- more efficient than: 'Fleury's algorithm'

## Matching pursuit
- https://en.wikipedia.org/wiki/Matching_pursuit
- does 'Sparse approximation'
- greedy algorithm

## Kahn's algorithm
- https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
- applications: 'topological sorting'

## Depth-first search
- https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
- applications: 'topological sorting', 'Strongly connected component'

## HyperLogLog++
- https://en.wikipedia.org/wiki/HyperLogLog
- solves 'Count-distinct problem' approximately
- implemented by Lucence

## Tarjan's strongly connected components algorithm
- https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
- computes 'Strongly connected component'

## Default algorithm for Huffman Tree
- https://en.wikipedia.org/wiki/Huffman_coding#Compression
- applications: 'Huffman coding'
- properties: 'greedy'
- uses 'priority queue'

## t-digest
- whitepaper: 'Computing extremely accurate quantiles using t-digests'
- Q-digest
- approximates percentiles
- is a: 'distributed algorithm'

## Chazelle algorithm for the minimum spanning tree
- paper: 'A minimum spanning tree algorithm with inverse-Ackermann type complexity'
- finds 'Minimum spanning tree'
- uses 'soft heaps'

## Fan and Su algortihm for multiple pattern match
- paper 'An Efficient Algorithm for Matching Multiple Patterns'
- "combines the concept of deterministic finite state automata (DFSA) and Boyer-Moore’s algorithm"

## Hu and Shing algortihm for matrix chain products
- paper 'Computation of Matrix Chain Products'
- solves 'Matrix chain multiplication' in O(n log n) where n is the number of matrices

## Long multiplication
- https://en.wikipedia.org/wiki/Multiplication_algorithm#Long_multiplication
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

## Viterbi algorithm
- https://en.wikipedia.org/wiki/Viterbi_algorithm
- uses method: 'Dynamic programming'
- solves: 'Viterbi path'
- applications: 'Speech recognition', 'Convolutional code', 'Speech synthesis', 'Computational linguistics', 'Bioinformatics'
- related theory: 'Hidden Markov model'

## Seam carving
- https://en.wikipedia.org/wiki/Seam_carving
- uses method: 'Dynamic programming'
- applications: 'Image resizing', 'Image processing'
- domain: 'computer graphics'
- implemented in: 'Adobe Photoshop', 'GIMP'

## Dynamic time warping
- https://en.wikipedia.org/wiki/Dynamic_time_warping
- applications: 'Time series analysis', 'Speech recognition', 'Speaker recognition', 'Signature recognition', 'Shape matching', 'Correlation power analysis'
- implemented in: 'Python pydtw'

## De Boor's algorithm
- https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
- domain: 'numerical analysis'
- spline curves in B-spline
- properties: 'numerically stable'
- implemented in: 'Python scipy.interpolate.BSpline'

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
- uses method: 'Dynamic programming'

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
- applications: 'contour finding'

## Lempel–Ziv–Welch
- https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch
- applications: 'Lossless compression'
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

## Exponentiation by squaring
- https://en.wikipedia.org/wiki/Exponentiation_by_squaring
- is a: powers algorithm (for semigroups)
- domain: arithmetic

## Bresenham's line algorithm
- https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
- is a: line drawing algorithm
- domain: computer graphics
- applications: rasterisation

## Xiaolin Wu's line algorithm
- https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm
- paper: 'An efficient antialiasing technique'
- is a: Line drawing algorithm, Anti-aliasing algorithm
- domain: computer graphics
- applications: antialiasing

## MinHash
- https://en.wikipedia.org/wiki/MinHash
- is a: probabilistic algorithm
- applications: 'Locality-sensitive hashing', set similarity, 'Jaccard similarity', 'data mining', 'bioinformatics'
- implemented in: 'ekzhu/datasketch'

## Needleman–Wunsch algorithm
- https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
- applications: 'Sequence alignment (global)', 'Computer stereo vision'
- time complexity: O(m n)
- uses method: 'Dynamic programming'
- domain: 'bioinformatics'

## Smith–Waterman algorithm
- https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
- applications: 'Sequence alignment (local)'
- time complexity: O(m n)
- uses method: 'Dynamic programming'
- domain: 'bioinformatics'

## Hirschberg's algorithm
- https://en.wikipedia.org/wiki/Hirschberg%27s_algorithm
- applications: 'Sequence alignment'
- uses method: 'Dynamic programming'
- loss function: 'Levenshtein distance'

## Karger's algorithm
- https://en.wikipedia.org/wiki/Karger%27s_algorithm
- is a: 'randomized algorithm'
- solves: 'minimum cut of a connected graph'
- domain: 'graph theory'
- improved by: 'Karger–Stein algorithm'

## Boykov-Kolmogorov algorithm
- paper: 'An experimental comparison of min-cut/max- flow algorithms for energy minimization in vision'
- implemented in: 'Python networkx.algorithms.flow.boykov_kolmogorov', 'boost::graph::boykov_kolmogorov_max_flow'

## Ford–Fulkerson algorithm
- https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm
- https://brilliant.org/wiki/ford-fulkerson-algorithm/
- properties: 'greedy', 'incomplete'
- solves: 'maximum flow in a flow network'
- implemented by: 'Edmonds–Karp algorithm'

## Edmonds–Karp algorithm
- https://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm
- https://brilliant.org/wiki/edmonds-karp-algorithm/
- implements: 'Ford–Fulkerson algorithm'
- implemented in: 'Python networkx.algorithms.flow.edmonds_karp'

## Marr–Hildreth algorithm
- https://en.wikipedia.org/wiki/Marr%E2%80%93Hildreth_algorithm
- applications: 'Edge detection'
- domain: 'image processing'

## Otsu's method
- https://en.wikipedia.org/wiki/Otsu%27s_method
- applications: 'Image thresholding'
- domain: 'image processing'
- implemented in: 'cv::threshold(type=THRESH_OTSU)'

# Filters

- these are convolved with some input

## Canny edge detector
- https://en.wikipedia.org/wiki/Canny_edge_detector
- domain: 'image processing'
- applications: 'Edge detection'

## Gabor filter
- https://en.wikipedia.org/wiki/Gabor_filter
- is a: 'linear filter'
- domain: 'image processing'
- applications: 'localize and extract text-only regions', 'facial expression recognition', 'pattern analysis'

## Sobel operator
- https://en.wikipedia.org/wiki/Sobel_operator
- is a: 'discrete differentiation operator'
- domain: 'image processing'
- applications: 'Edge detection'

## Scharr operator
- dissertation: 'Optimale Operatoren in der Digitalen Bildverarbeitung'
- domain: 'image processing'
- applications: 'Edge detection'

## Kuwahara filter
- https://en.wikipedia.org/wiki/Kuwahara_filter
- https://reference.wolfram.com/language/ref/KuwaharaFilter.html
- is a: 'non-linear filter'
- applications: 'adaptive noise reduction', 'medical imaging', 'fine-art photography'
- disadvantages: 'create block artifacts'
- domain: 'image processing'

# Methods

## Locality-sensitive hashing
- https://en.wikipedia.org/wiki/Locality-sensitive_hashing
- used for: 'nearest neighbor search' 

## Dynamic programming
- https://en.wikipedia.org/wiki/Dynamic_programming
- exploits: 'optimal substructure'

## Brute force
- https://en.wikipedia.org/wiki/Brute-force_search

# General problems

## Nearest neighbor search
- https://en.wikipedia.org/wiki/Nearest_neighbor_search
- also called: 'post-office problem'
- solved exactly by: 'Space partitioning', 'Linear search'
- solved approximatly by: 'Hierarchical Navigable Small World graphs', 'Locality-sensitive hashing', 'Cover tree', 'Vector quantization'
- implemented by: 'spotify/annoy', 'C++ ANN', 'nmslib/hnsw', 'nmslib/nmslib'

## Cycle decomposition
- https://en.wikipedia.org/wiki/Cycle_decomposition_(group_theory)

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
- applications: 'Network design', 'Image segmentation', 'Cluster analysis'

## Second-best minimum spanning tree
- see book: 'Introduction to Algorithms'
- solution need not be unique
- variant of 'Minimum spanning tree'

## Bottleneck spanning tree
- see book: 'Introduction to Algorithms'
- variant of 'Minimum spanning tree'
- a 'minimum spanning tree' is a 'bottleneck spanning tree'

## spanning-tree verification
- see book: 'Introduction to Algorithms'
- related to 'Minimum spanning tree'

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
- exhibits: 'optimal substructure'
- domain: 'graph theory'

## Single-pair shortest path problem
- https://en.wikipedia.org/wiki/Shortest_path_problem#Single-source_shortest_paths
- no algorithms with better worst time complexity than for 'Single-source shortest path problem' are know (which is a generalization)

## All-pairs shortest paths problem
- https://en.wikipedia.org/wiki/Shortest_path_problem#All-pairs_shortest_paths
- finds the shortest path for all pairs of vectices in a graph
- solved by 'Floyd–Warshall algorithm', 'Johnson's algorithm'
- domain: 'graph theory'
- exhibits: 'optimal substructure'

## approximate string matching
- https://en.wikipedia.org/wiki/Approximate_string_matching
- paper "Fast Approximate String Matching in a Dictionary"
- applications: s'pell checking', 'nucleotide sequence matching'

## Lowest common ancestor (LCA)
- https://en.wikipedia.org/wiki/Lowest_common_ancestor
- domain: 'graph theory'

## Longest common substring problem
- https://en.wikipedia.org/wiki/Longest_common_substring_problem
- cf. 'Longest common subsequence problem'
- exhibits: 'optimal substructure'
- solutions: Generalized suffix tree
- domain: 'combinatorics'

## Longest common subsequence problem
- https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
- cf. 'Longest common substring problem'
- Hunt–McIlroy algorithm
- applications: version control systems, wiki engines, and molecular phylogenetics
- domain: 'combinatorics'
- solved by: 'diff'

## Shortest common supersequence problem
- https://en.wikipedia.org/wiki/Shortest_common_supersequence_problem
- applications: DNA sequencing
- domain: 'combinatorics'

## Shortest common superstring problem
- https://en.wikipedia.org/wiki/Shortest_common_supersequence_problem#Shortest_common_superstring
- applications: sparse matrix compression
- domain: 'combinatorics'

## Longest increasing subsequence
- https://en.wikipedia.org/wiki/Longest_increasing_subsequence
- domain: 'combinatorics'
- exhibits: 'optimal substructure'

## Maximum subarray problem
- https://en.wikipedia.org/wiki/Maximum_subarray_problem
- applications: 'genomic sequence analysis', 'computer vision', 'data mining'
- solved by: 'Kadane's algorithm'

## Optical flow
- https://en.wikipedia.org/wiki/Optical_flow
- applications: 'Motion estimation', 'video compression', 'object detection', 'object tracking', 'image dominant plane extraction', 'movement detection', 'robot navigation , 'visual odometry'
- domain: 'machine vision', 'computer vision'

## Sequence alignment
- https://en.wikipedia.org/wiki/Sequence_alignment

## Partial sorting
- https://en.wikipedia.org/wiki/Partial_sorting
- solved by: 'heaps', 'quickselsort', 'Quickselect'

## Incremental sorting
- https://en.wikipedia.org/wiki/Partial_sorting#Incremental_sorting
- solved by: 'quickselect', 'heaps'

## Hamiltonian path problem
- https://en.wikipedia.org/wiki/Hamiltonian_path_problem
- https://www.hackerearth.com/practice/algorithms/graphs/hamiltonian-path/
- solved by algorithms which solve: 'Boolean satisfiability problem'
- domain: "graph theory"

## Eulerian path problem
- https://en.wikipedia.org/wiki/Eulerian_path
- application: 'in bioinformatics to reconstruct the DNA sequence from its fragments'
- application: 'CMOS circuit design to find an optimal logic gate ordering'
- compare: 'Hamiltonian path problem'
- if exists, optimal solution for: 'Route inspection problem'
- domain: "graph theory"

## Route inspection problem / Chinese postman problem
- https://en.wikipedia.org/wiki/Route_inspection_problem
- http://mathworld.wolfram.com/ChinesePostmanProblem.html

## Closure problem
- https://en.wikipedia.org/wiki/Closure_problem
- domain: "graph theory"
- applications: 'Open pit mining', 'Military targeting', 'Transportation network design', 'Job scheduling'
- can be reduced to: 'Maximum flow problem'

## Maximum flow problem
- https://en.wikipedia.org/wiki/Maximum_flow_problem
- domain: 'graph theory'

# Specific problems

## Knight's tour
- https://en.wikipedia.org/wiki/Knight%27s_tour
- http://mathworld.wolfram.com/KnightGraph.html
- its graph is a: 'bipartite graph'
- solved by 'depth first search' on graph of legal moves, using 'Warnsdorff's Rule' (heuristic) for improved performance
- version of: 'Hamiltonian path problem'

## Tower of Hanoi
- https://en.wikipedia.org/wiki/Tower_of_Hanoi
- http://mathworld.wolfram.com/TowerofHanoi.html
- applications: 'Backup rotation scheme'
- isomorphic to finding a 'Hamiltonian path' on an n-hypercube
