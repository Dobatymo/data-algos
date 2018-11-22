# resources

## online books
- https://bradfieldcs.com/algos/
- http://interactivepython.org/RkmcZ/courselib/static/pythonds/index.html

## books
- Artificial Intelligence: A Modern Approach
- Introduction to Algorithms

# Abstract data type

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

# Data structures

## Array
- https://en.wikipedia.org/wiki/Array_data_structure
- implemented in: 'C++ std::vector', 'Python list, tuple'
- continuous list of elements in memory. fast to iterate, fast to access by index. slow to find by value, slow to insert/delete within the array (as memory always need to be continuous, they need to be reallocated)
- usually fast to append/delete at the end or beginning (if there is free memory and depending on exact implementation)

## Linked list
- https://en.wikipedia.org/wiki/Linked_list
- single elements in memory which contain a pointer to the next element in the sequence
- double linked list also contain pointers back to the previous element int the sequence (XOR linked list is a clever optimization)
- insert and delete can be constant time if pointers are already found. iteration is slow. access by index is not possible, search by value in linear.
- implemented in: 'C++ std::list, std::forward_list'

## Hash table
- https://en.wikipedia.org/wiki/Hash_table
- used to implement maps: 'Python dict', 'C++ std::unordered_map'
- probably the most important data structure in Python
- has basically perfect complexity for a good hash function. (Minimal) perfect hashing can be used if the keys are known in advance.
- ordering depends on implementation (python 3.7 garuantees preservation of insertion order, whereas C++ as the name says does not define any ordering)
- even though hashing is constant in time, it might still be slow. also the hash consumes space
- hash tables usually have set characteristics (ie. the can test in average constant time if an item is the table or not)
- time complexity (search, insert, delete) (average): O(1)
- time complexity (search, insert, delete) (worst): O(n)
- space complexity (average): O(n)
- space complexity (worst): O(n) # often a lot of padding space is needed...

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

## Trie
- also called: 'digital tree'
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

## Finite-state machine
- https://en.wikipedia.org/wiki/Finite-state_machine

## Finite-state transducer
- https://en.wikipedia.org/wiki/Finite-state_transducer
- is a: 'Finite-state machine'
- implemented in: 'OpenFst'

## Deterministic acyclic finite state automaton
- also called: 'DAFSA', 'deterministic acyclic finite state acceptor'
- https://en.wikipedia.org/wiki/Deterministic_acyclic_finite_state_automaton
- used to implement ordered sets
- is a: 'Finite-state machine'

## Minimal acyclic finite state automaton
- also called: 'MAFSA', 'minimal acyclic finite state acceptor'
- https://en.wikipedia.org/wiki/Deterministic_acyclic_finite_state_automaton
- https://blog.burntsushi.net/transducers/
- minimal: 'Deterministic acyclic finite state automaton'
- space optimized version of tries, with missing map characteristics
- allow for prefix (and possibly suffix) search
- more space efficient than tries as common prefixes and suffixes are only stored once and thus the number of pointers is reduced as well
- for a version with map characteristics see: 'Minimal acyclic finite state transducer'
- implemented in: 'C++ dawgdic'

## Deterministic acyclic finite state transducer
- see: 'Minimal acyclic finite state transducer'
- paper: 'Applications of finite automata representing large vocabularies'
- https://blog.burntsushi.net/transducers/
- is a: 'Finite-state transducer'

## Minimal acyclic finite state transducer
- also called: 'MAFST'
- https://blog.burntsushi.net/transducers/
- http://stevehanov.ca/blog/index.php?id=119
- minimal: 'Deterministic acyclic finite state transducer'
- MAFSA with map characteristics
- association of keys with values reduces lookup time from O(sequence-length) to O(sequence-length*log(tree size))???
- solves: 'Minimal Perfect Hashing'
- implemented in: 'C++ dawgdic'

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
- implemented in: 'boost::graph::incremental_components'
- used for: 'Kruskal's algorithm'

## HAT-trie
- https://en.wikipedia.org/wiki/HAT-trie
- implemented in: 'Python pytries/hat-trie'
- implements: 'Ordered map'
- variant of: 'radix trie'
- properties: 'cache friendly'

## Double-Array Trie
- also called: 'DATrie'
- implemented in: 'pytries/datrie', 'libdatrie'
- paper: 'An Efficient Digital Search Algorithm by Using a Double-Array Structure'

## Ternary search tree
- https://en.wikipedia.org/wiki/Ternary_search_tree
- type of: 'Trie'
- applications: 'Nearest neighbor search'

## Soft heap
- https://en.wikipedia.org/wiki/Soft_heap
- approximates: 'Priority queue'

## Bloom filter
- https://en.wikipedia.org/wiki/Bloom_filter
- properties: 'probabilistic'
- implements: 'Set'
- applications: caching strategies, database query optimization, rate-limiting, data synchronization, chemical structure searching

## Optimal binary search tree
- https://en.wikipedia.org/wiki/Optimal_binary_search_tree
- is a: 'Binary search tree'

## Order statistic tree
- https://en.wikipedia.org/wiki/Order_statistic_tree
- variant of: 'Binary search tree'
- additional interface: find the i'th smallest element stored in the tree, find the rank of element x in the tree, i.e. its index in the sorted list of elements of the tree

## Exponential tree
- https://en.wikipedia.org/wiki/Exponential_tree
- variant of: 'Binary search tree'

# Index data structures

## Inverted index
- https://en.wikipedia.org/wiki/Inverted_index
- used by: 'ElasticSearch', 'Lucene'
- maps content/text to locations/documents
- Search engine indexing
- cf. Back-of-the-book index, Concordance
- applications: full-text search, sequence assembly

# Algorithms

## Breadth-first search
- https://en.wikipedia.org/wiki/Breadth-first_search
- input: 'Graph'

## Depth-first search
- https://en.wikipedia.org/wiki/Depth-first_search
- input: 'Graph'

## k-d tree construction algorithm using sliding midpoint rule
- example paper: Maneewongvatana and Mount 1999
- constructs: 'k-d tree'
- implemented in: 'scipy.spatial.KDTree'
- input: 'List of k-dimensional points'

## Median of medians
- https://en.wikipedia.org/wiki/Median_of_medians
- selection algorithm
- input: 'random access collection'

## Introselect
- https://en.wikipedia.org/wiki/Introselect
- implemented in C++ std::nth_element
- is a: 'Selection algorithm'
- input: 'random access collection'

## Floyd–Rivest algorithm
- https://en.wikipedia.org/wiki/Floyd%E2%80%93Rivest_algorithm
- is a: 'Selection algorithm', 'Divide and conquer algorithm'
- input: 'random access collection'

## Quickselect
- https://en.wikipedia.org/wiki/Quickselect
- is a: 'Selection algorithm'
- input: 'random access collection'

## Dijkstra's algorithm
- https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
- http://mathworld.wolfram.com/DijkstrasAlgorithm.html
- uses method: 'dynamic programming'
- solves 'Shortest path problem' for non-negative weights in directed/undirected graphs in O(v^2) where v is the number of vertices
- variant implementation with Fibonacci heaps runs in O(e * v*log v) where e and v are the number of edges and vertices resp.
- implemented in: 'Python scipy.sparse.csgraph.shortest_path(method='D')'
- Fibonacci implementation is the asymptotically the fastest known single-source shortest-path algorithm for arbitrary directed graphs with unbounded non-negative weights.
- input: 'Directed graph with non-negative weights'

## Bellman–Ford algorithm
- https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm
- solves variant of the 'Shortest path problem' for real-valued edge weights in directed graph in O(v*e) where v and e are the number of vertices and edges respectively.
- negative cycles are detected
- implemented in: 'Python scipy.sparse.csgraph.shortest_path(method="BF")'
- input: 'Weighted directed graph'

## Johnson's algorithm
- https://en.wikipedia.org/wiki/Johnson%27s_algorithm
- solves 'All-pairs shortest paths problem' for real-valued weights in a directed graph in O(v^2 log v + v*e) where v and e are the number of vertices and edges
- implemented in: 'Python scipy.sparse.csgraph.shortest_path(method='J')', 'C++ boost::graph::johnson_all_pairs_shortest_paths'
- combination of 'Bellman–Ford' and 'Dijkstra's algorithm'
- is faster than 'Floyd–Warshall algorithm' for sparse graphs
- input: 'weighted directed graph without negative cycles'

## Floyd–Warshall algorithm
- https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
- http://mathworld.wolfram.com/Floyd-WarshallAlgorithm.html
- graph algorithm
- solves 'All-pairs shortest paths problem' for real-valued weights for directed/undirected graphs in O(v^3) where v is the number of vertices
- negative cycles are not allowed
- uses method: 'dynamic programming'
- implemented in: 'python scipy.sparse.csgraph.shortest_path(method='FW')', 'c++ boost::graph::floyd_warshall_all_pairs_shortest_paths'
- is faster then 'Johnson's algorithm' for dense graphs
- operates in: 'weighted directed graph without negative cycles'

## Suurballe's algorithm
- https://en.wikipedia.org/wiki/Suurballe%27s_algorithm
- paper: 'Disjoint paths in a network (1974)'
- solves: 'Shortest pair of edge disjoint paths'
- input: 'Directed graph with non-negative weights'

## Edge disjoint shortest pair algorithm
- https://en.wikipedia.org/wiki/Edge_disjoint_shortest_pair_algorithm
- paper: 'Survivable networks: algorithms for diverse routing'
- solves: 'Shortest pair of edge disjoint paths'
- superseded by: 'Suurballe's algorithm'
- input: 'Weighted directed graph'

## Reaching algorithm
- http://mathworld.wolfram.com/ReachingAlgorithm.html
- solves: 'Shortest path problem'
- time complexity: O(n), where n is the number of edges
- where does the name come from? is this the same as using topological sorting?
- input: 'Acyclic directed graph'

## Collaborative diffusion
- https://en.wikipedia.org/wiki/Collaborative_diffusion
- applications: 'pathfinding'
- time complexity: constant in the number of agents

## Ukkonen's algorithm
- https://en.wikipedia.org/wiki/Ukkonen%27s_algorithm
- paper: 'On-line construction of suffix trees'
- constructs: 'suffix tree'
- properties: 'online'
- time complexity: O(n), where n is the length of the string
- input: 'List of strings'

## A* search algorithm
- https://en.wikipedia.org/wiki/A*_search_algorithm
- generalization of 'Dijkstra's algorithm'
- heuristic search
- informed search algorithm (best-first search)
- usually implemented using: 'priority queue'
- applications: pathfinding, parsing using stochastic grammars in NLP
- uses method: 'dynamic programming'
- input: 'Weighted graph'

## Linear search
- https://en.wikipedia.org/wiki/Linear_search
- find element in any sequence in O(i) time where i is the index of the element in the sequence
- works on: 'Linked list', 'Array', 'List'
- has an advantage when sequential access is fast compared to random access
- O(1) for list with geometric distributed values
- implemented in c++ std::find (impl. dependent), python list.index
- input: 'List'

## Binary search algorithm
- https://en.wikipedia.org/wiki/Binary_search_algorithm
- find element in sorted finite list in O(log n) time where n is the number of elements in list
- requires: 'Random access'
- variants: 'Exponential search'
- implemented in 'C++ std::binary_search', 'python bisect'
- input: 'Sorted list'

## Naïve string-search algorithm
- https://en.wikipedia.org/wiki/String-searching_algorithm#Na%C3%AFve_string_search
- find string in string in O(n+m) average time and O(n*m) worst case, where n and m are strings to be search for, resp. in.
- implemented in: 'C++ std::search (impl. dependent)', 'python list.index'
- input: 'Buffered list'

## Exponential search
- https://en.wikipedia.org/wiki/Exponential_search
- find element in sorted infinite list in O(log i) time where i is the position of the element in the list
- input: 'Sorted list'

## Funnelsort
- https://en.wikipedia.org/wiki/Funnelsort
- is a: 'cache-oblivious algorithm', 'external memory algorithm', 'Comparison-based sorting algorithm'
- input: 'Collection'

## Quicksort
- https://en.wikipedia.org/wiki/Quicksort
- http://mathworld.wolfram.com/Quicksort.html
- is a: 'Sorting algorithm', 'Unstable sorting algorithm', 'In-place algorithm', 'Divide and conquer algorithm', 'Comparison-based sorting algorithm'
- time complexity (best): O(n log n)
- time complexity (average): O(n log n)
- time complexity (worst): O(n^2)
- space complexity: O(log n) auxiliary
- input: 'Random access collection'

## Radix sort
- https://en.wikipedia.org/wiki/Radix_sort
- input: 'Collection of integers'

## Bubble sort
- also called: 'Sinking sort'
- https://en.wikipedia.org/wiki/Bubble_sort
- input: 'Bidirectional Collection'
- it's bad, only applicable to almost sorted inputs
- properties: 'stable', 'in-place'

## Gnome sort
- https://en.wikipedia.org/wiki/Gnome_sort
- time complexity (average, worst): O(n^2)
- time complexity (best): O(n)
- requires no nested loops

## Cocktail shaker sort
- also called: 'bidirectional bubble sort'
- https://en.wikipedia.org/wiki/Cocktail_shaker_sort
- input: 'Bidirectional Collection'
- properties: 'stable', 'in-place'
- variant of: 'Bubble sort'
- time complexity (average, worst): O(n^2)
- time complexity (best): O(n)

## Merge Sort
- https://en.wikipedia.org/wiki/Merge_sort
- is a: 'Sorting algorithm', 'Stable sorting algorithm' (usually), 'Divide and conquer algorithm', 'Comparison-based sorting algorithm'
- implemented in: 'C++ std::stable_sort (usually)'
- good for sequential access, can work on 'singly linked lists', external sorting
- easily parallelizable
- input: 'Collection'

## Heapsort
- https://en.wikipedia.org/wiki/Heapsort
- http://mathworld.wolfram.com/Heapsort.html
- is a: 'Sorting algorithm', 'In-place algorithm', 'Unstable sorting algorithm', 'Comparison-based sorting algorithm'
- solves: 'Partial sorting'
- time complexity (average, best, worst): O(n log n)
- space complexity: O(1)
- uses: 'max heap'
- not easily parallelizable
- variant works on 'doubly linked lists'
- input: 'Random access collection'

## Timsort
- https://en.wikipedia.org/wiki/Timsort
- is a: 'Sorting algorithm', 'Stable sorting algorithm', 'Comparison-based sorting algorithm'
- implemented in: 'Python sorted', 'Android Java'
- input: 'Random access collection'

## Introsort
- https://en.wikipedia.org/wiki/Introsort
- is a: 'Sorting algorithm', 'Unstable sorting algorithm', 'Comparison-based sorting algorithm'
- implemented in: 'C++ STL std::sort (usually)', '.net sort'
- input: 'Random access collection'

## Selection sort
- https://en.wikipedia.org/wiki/Selection_sort
- http://mathworld.wolfram.com/SelectionSort.html
- is a: 'Sorting algorithm', 'In-place algorithm', 'Unstable sorting algorithm', 'Comparison-based sorting algorithm'
- input: 'Random access collection'

## Insertion sort
- https://en.wikipedia.org/wiki/Insertion_sort
- properties: 'stable', 'in-place'
- input: 'List' (for not-in-place)
- input: 'bidirectional list' (for in-place)

## Shellsort
- https://en.wikipedia.org/wiki/Shellsort
- http://mathworld.wolfram.com/Shellsort.html
- is a: 'Sorting algorithm', 'In-place algorithm', 'Unstable sorting algorithm', 'Comparison-based sorting algorithm', 'Adaptive sort'
- input: 'Random access collection'

## Cycle sort
- https://en.wikipedia.org/wiki/Cycle_sort
- is a: 'Sorting algorithm', 'In-place algorithm', 'Unstable sorting algorithm', 'Comparison-based sorting algorithm'
- solves: 'cycle decomposition'
- theoretically optimal in terms of the total number of writes to the original array
- used for sorting where writes are expensive
- applications: EEPROM
- time complexity: O(n^2)
- space complexity: O(1) auxiliary
- input: 'Random access collection'

## Patience sorting
- https://en.wikipedia.org/wiki/Patience_sorting
- is a: 'Sorting algorithm', 'Comparison-based sorting algorithm'
- finds: 'Longest increasing subsequence'
- applications: 'Process control'
- see also: 'Floyd's game'
- input: 'Collection'
- input: 'Contigious Collection' (time complexity: O(n log(log(n))), using 'Van Emde Boas tree')

## Fisher–Yates shuffle
- https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
- is a: 'Shuffling algorithm', 'In-place algorithm'
- unbiased
- input: 'Random access collection'

## Reservoir sampling
- https://en.wikipedia.org/wiki/Reservoir_sampling
- family of 'randomized algorithms'
- version of: 'Fisher–Yates shuffle'

## Cache-oblivious distribution sort
- https://en.wikipedia.org/wiki/Cache-oblivious_distribution_sort
- comparison-based sorting algorithm
- cache-oblivious algorithm

## Naive Method for SimRank by Jeh and Widom
- paper: 'SimRank: a measure of structural-context similarity (2002)'
- calculate: 'SimRank'

## Wagner–Fischer algorithm
- https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm
- uses method: 'dynamic programming'
- calculates: 'Levenshtein distance'
- O(n*m) complexity where n and m are the respective string lenths
- optimal time complexity for problem proven to be O(n^2), so this algorithm is pretty much optimal
- space complexity of O(n*m) could be reduced to O(n+m)
- input: 'two strings'

## Aho–Corasick algorithm
- https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm
- https://xlinux.nist.gov/dads/HTML/ahoCorasick.html
- multiple *string searching*
- implemented in: original fgrep
- (pre)constructs 'Finite-state machine' from set of search strings
- applications: virus signature detection
- paper 'Efficient string matching: An aid to bibliographic search'
- classification 'constructed search engine', 'match prefix first', 'one-pass'
- shows better results than 'Commentz-Walter' for peptide identification according to 'Commentz-Walter: Any Better Than Aho-Corasick For Peptide Identification?' and for biological sequences according to 'A Comparative Study On String Matching Algorithms Of Biological Sequences'
- input: 'Collection of strings' (construction)
- input: 'List of characters' (searching)

## Commentz-Walter algorithm
- https://en.wikipedia.org/wiki/Commentz-Walter_algorithm
- multiple *string searching*
- classification 'match suffix first'
- implemented in: grep (variant)

## Boyer–Moore string-search algorithm
- https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_string-search_algorithm
- single *string searching*
- implemented in: 'grep', 'C++ std::boyer_moore_searcher'
- implemented in: 'Python str' (variant)
- better for large alphabets like text than: 'Knuth–Morris–Pratt algorithm'
- paper: 'A Fast String Searching Algorithm'

## Knuth–Morris–Pratt algorithm
- https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm
- single *string searching*
- implemented in: 'grep'
- better for small alphabets like DNA than: 'Boyer–Moore string-search algorithm'

## Beam search
- https://en.wikipedia.org/wiki/Beam_search
- is a: 'heuristic search algorithm'
- properties: 'greedy'
- optimization of best-first search
- greedy version of breadth-first search
- applications: 'Machine translation', 'Speech recognition'
- approximate solution

## Hu–Tucker algorithm
- superseded by: 'Garsia–Wachs algorithm'

## Garsia–Wachs algorithm
- https://en.wikipedia.org/wiki/Garsia%E2%80%93Wachs_algorithm
- constructs: 'Optimal binary search tree' (special case)

## Knuth's optimal binary search tree algorithm
- https://en.wikipedia.org/wiki/Optimal_binary_search_tree#Knuth%27s_dynamic_programming_algorithm
- paper: 'Optimum binary search trees'
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

## Wall follower
- also called: 'left-hand rule', 'right-hand rule'
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
- http://mathworld.wolfram.com/KruskalsAlgorithm.html
- finds: 'Minimum spanning tree'
- properties: 'greedy'
- implemented in: 'C++ boost::graph::kruskal_minimum_spanning_tree'
- book: 'Introduction to Algorithms'

## Prim's algorithm
- https://en.wikipedia.org/wiki/Prim%27s_algorithm
- finds 'Minimum spanning tree'
- greedy algorithm
- implemented in c++ boost::graph::prim_minimum_spanning_tree
- time complexity depends on used data structures
- book: 'Introduction to Algorithms'

## Hierholzer's algorithm
- https://en.wikipedia.org/wiki/Eulerian_path#Hierholzer's_algorithm
- https://www.geeksforgeeks.org/hierholzers-algorithm-directed-graph/
- solves: 'Eulerian path'
- more efficient than: 'Fleury's algorithm'

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

## Hunt–McIlroy algorithm
- https://en.wikipedia.org/wiki/Hunt%E2%80%93McIlroy_algorithm
- solves: 'Longest common subsequence problem'
- implemented in: 'diff'

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

## Risch semi-algorithm
- https://en.wikipedia.org/wiki/Risch_algorithm
- http://mathworld.wolfram.com/RischAlgorithm.html
- solves: 'indefinite integration'
- applications: 'Symbolic computation', 'Computer algebra'
- implemented in: 'Axiom'

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
- time complexity: O(n)
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
- properties: 'was patented'

## Exponentiation by squaring
- https://en.wikipedia.org/wiki/Exponentiation_by_squaring
- is a: 'Powers algorithm'
- domain: 'Arithmetic'
- input: 'Semigroup' & 'Positive integer'
- output: 'Exponentiated semigroup'

## Bresenham's line algorithm
- https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
- http://mathworld.wolfram.com/BresenhamsLineAlgorithm.html
- is a: 'Line drawing algorithm'
- domain: 'Computer graphics'
- applications: 'Rasterisation'
- input: 'Start and end points'
- output: 'List of points'

## Berlekamp–Massey algorithm
- https://en.wikipedia.org/wiki/Berlekamp%E2%80%93Massey_algorithm
- http://mathworld.wolfram.com/Berlekamp-MasseyAlgorithm.html
- applications: 'Error detection and correction'
- domain: 'Field theory'
- input: 'List of bools'
- output: 'shortest linear feedback shift register'
- input: 'arbitrary field'
- output: 'minimal polynomial of a linearly recurrent sequence'

## Xiaolin Wu's line algorithm
- https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm
- paper: 'An efficient antialiasing technique'
- is a: Line drawing algorithm, Anti-aliasing algorithm
- domain: computer graphics
- applications: antialiasing
- input: 'Start and end points'
- output: 'List of points with associated graylevel'

## MinHash
- https://en.wikipedia.org/wiki/MinHash
- properties: 'probabilistic'
- applications: 'Locality-sensitive hashing', 'Set similarity', 'data mining', 'bioinformatics'
- implemented in: 'ekzhu/datasketch'
- approximates: 'Jaccard similarity'

## Needleman–Wunsch algorithm
- https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
- applications: 'Sequence alignment (global)', 'Computer stereo vision'
- time complexity: O(m n)
- uses method: 'Dynamic programming'
- domain: 'bioinformatics'
- input: 'two random access collections'
- output: 'Optimal global alignment'

## Smith–Waterman algorithm
- https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
- applications: 'Sequence alignment (local)'
- time complexity: O(m n)
- uses method: 'Dynamic programming'
- domain: 'bioinformatics'
- input: 'two random access collections'
- output: 'Optimal local alignment'

## Hirschberg's algorithm
- https://en.wikipedia.org/wiki/Hirschberg%27s_algorithm
- applications: 'Sequence alignment'
- uses method: 'Dynamic programming'
- loss function: 'Levenshtein distance'
- input: 'two random access collections'
- output: 'Optimal sequence alignment'

## Karger's algorithm
- https://en.wikipedia.org/wiki/Karger%27s_algorithm
- properties: 'randomized'
- input: 'Connected graph'
- output: 'Minimum cut'
- domain: 'Graph theory'
- improved by: 'Karger–Stein algorithm'

## Boykov-Kolmogorov algorithm
- paper: 'An experimental comparison of min-cut/max- flow algorithms for energy minimization in vision'
- implemented in: 'Python networkx.algorithms.flow.boykov_kolmogorov', 'boost::graph::boykov_kolmogorov_max_flow'

## Ford–Fulkerson method
- https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm
- https://brilliant.org/wiki/ford-fulkerson-algorithm/
- properties: 'greedy', 'incomplete'
- solves: 'maximum flow in a flow network'
- implemented by: 'Edmonds–Karp algorithm'

## Edmonds–Karp algorithm
- https://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm
- https://brilliant.org/wiki/edmonds-karp-algorithm/
- implements: 'Ford–Fulkerson method'
- implemented in: 'Python networkx.algorithms.flow.edmonds_karp'
- time complexity: O(v e^2) or O(v^2 e) where v is the number of vertices and e the number of edges
- input: 'Flow network'
- output: 'Maximum flow'
- book: 'Introduction to Algorithms'

## Marr–Hildreth algorithm
- https://en.wikipedia.org/wiki/Marr%E2%80%93Hildreth_algorithm
- paper: 'Theory of edge detection (1980)'
- applications: 'Edge detection'
- domain: 'image processing'
- input: 'Grayscale image'
- output: 'Binary image'

## Otsu's method
- https://en.wikipedia.org/wiki/Otsu%27s_method
- applications: 'Image thresholding'
- domain: 'Image processing'
- implemented in: 'cv::threshold(type=THRESH_OTSU)'
- input: 'Grayscale image'
- output: 'Binary image'

## Soundex
- https://en.wikipedia.org/wiki/Soundex
- is a: 'Phonetic algorithm'
- applications: 'Indexing', 'Phonetic encoding'

## Match rating approach
- https://en.wikipedia.org/wiki/Match_rating_approach
- is a: 'Phonetic algorithm'
- applications: 'Indexing', 'Phonetic encoding', 'Phonetic comparison'
- is a: 'Similarity measure'

## Chamfer matching
- paper: 'Parametric correspondence and chamfer matching: two new techniques for image matching'
- uses: 'Chamfer distance'

## k-nearest neighbors algorithm
- is a: 'machine learning algorithm'
- properties: 'non-parametric', 'instance-based learning'
- special case of: 'Variable kernel density estimation'
- applications: 'Pattern recognition'
- implemented: 'Python sklearn.neighbors.KNeighborsRegressor, sklearn.neighbors.KNeighborsClassifier'

## k-means clustering
- https://en.wikipedia.org/wiki/K-means_clustering
- is a: 'Clustering algorithm'
- implemented in: 'sklearn.cluster.KMeans, scipy.cluster.vq.kmeans, Bio.Cluster.kcluster', 'cv::kmeans'
- partitions space into: 'Voronoi cells'
- applications: 'Vector quantization', 'Cluster analysis', 'Feature learning'
- input: 'Collection of points' & 'Positive integer k'
- output: 'Collection of cluster indices'

## k-medoids algorithm
- also called: 'Partitioning Around Medoids'
- https://en.wikipedia.org/wiki/K-medoids
- is a: 'Clustering algorithm'
- more robust to noise than 'k-means clustering'
- input: 'Collection of points' & 'Positive integer k'
- output: 'Collection of cluster indices'
- implemented in: 'Python Bio.Cluster.kmedoids'

## Linde–Buzo–Gray algorithm
- https://en.wikipedia.org/wiki/Linde%E2%80%93Buzo%E2%80%93Gray_algorithm
- similar: 'k-means clustering'
- generalization of: 'Lloyd's algorithm'

## Single-linkage clustering method
- also called: 'single-link cluster method', 'Nearest Point Algorithm'
- https://en.wikipedia.org/wiki/Single-linkage_clustering
- applications: 'Hierarchical clustering'
- implemented by: 'SLINK algorithm'

## SLINK algorithm
- paper: 'SLINK: An optimally efficient algorithm for the single-link cluster method'
- implementation of: 'Single-linkage clustering method'
- implemented in: 'scipy.cluster.hierarchy.linkage'

## Rohlf's algorithm for hierarchical clustering
- also called: MST-algorithm
- paper: 'Hierarchical clustering using the minimum spanning tree (1973)'
- implementation of: 'Single-linkage clustering method'

## Complete-linkage clustering method
- also called: 'Farthest Point Algorithm'
- https://en.wikipedia.org/wiki/Complete-linkage_clustering
- applications: 'Hierarchical clustering'
- implemented by: 'CLINK algorithm' (not according to 'Modern hierarchical, agglomerative clustering algorithms')

## CLINK algorithm
- paper: 'An efficient algorithm for a complete link method (1977)'
- implementation of: 'Complete-linkage clustering method' (not according to 'Modern hierarchical, agglomerative clustering algorithms')

## Voorhees algorithm
- http://courses.cs.vt.edu/~cs5604/cs5604cnCL/CL-alg-details.html
- paper: 'Implementing agglomerative hierarchic clustering algorithms for use in document retrieval (1986)'
- credits: 'Chris Buckley'
- implementation of: 'Complete-linkage clustering method'
- implemented in: 'scipy.cluster.hierarchy.linkage'

## Unweighted Pair Group Method with Arithmetic Mean
- also called: UPGMA
- https://en.wikipedia.org/wiki/UPGMA
- applications: 'Hierarchical clustering'
- implemented in: 'scipy.cluster.hierarchy.linkage'
- creates: 'Dendrogram', 'Ultrametric tree'

## Weighted Pair Group Method with Arithmetic Mean
- also called: WPGMA
- https://en.wikipedia.org/wiki/WPGMA
- applications: 'Hierarchical clustering'
- implemented in: 'scipy.cluster.hierarchy.linkage'
- creates: 'Dendrogram', 'Ultrametric tree'

## Unweighted Pair Group Method with Centroid Averaging
- also called: UPGMC
- applications: 'Hierarchical clustering'
- implemented in: 'scipy.cluster.hierarchy.linkage'

## Weighted Pair Group Method with Centroid Averaging
- also called: WPGMC
- applications: 'Hierarchical clustering'
- implemented in: 'scipy.cluster.hierarchy.linkage'

## Rocchio algorithm
- similar: 'Nearest centroid classifier'
- https://en.wikipedia.org/wiki/Rocchio_algorithm
- applications: 'Relevance feedback'

## Nearest centroid classifier
- https://en.wikipedia.org/wiki/Nearest_centroid_classifier
- is a: 'Classification model'
- input: 'Collection of points with associated labels' (training)
- input: 'Point' (prediction)
- output: 'Label'
- properties: 'reduced data model'
- relies on: 'Nearest neighbor search'

## Forsythe's algorithm for sampling generalized exponential distributions
- paper: 'Von Neumann's Comparison Method for Random Sampling from the Normal and Other Distributions'
- sample from: exp(−G(x)), where G(x) is 'easy to compute'
- domain: 'Random numbers'

## Ziggurat algorithm
- https://en.wikipedia.org/wiki/Ziggurat_algorithm
- paper: 'The Ziggurat Metho d for Generating Random Variables'
- input: 'List of uniformly-distributed random numbers'
- output: sample from 'monotone decreasing probability distribution', 'symmetric unimodal distribution'
- is a: 'Statistical algorithm'
- domain: 'Random numbers'

## Remez algorithm
- https://en.wikipedia.org/wiki/Remez_algorithm
- http://mathworld.wolfram.com/RemezAlgorithm.html
- is a: 'Minimax approximation algorithm'
- input: 'Function'
- output: 'Function in Chebyshev space of best approximation', e.g. 'Polynomial of best approximation'
- applications: 'Function approximation'

## Neville's algorithm
- https://en.wikipedia.org/wiki/Neville%27s_algorithm
- http://mathworld.wolfram.com/NevillesAlgorithm.html
- applications: 'Polynomial interpolation', 'Numerical differentiation'
- input: 'Collection of points'
- output: 'Polynomial'

## Bron–Kerbosch algorithm
- https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
- http://mathworld.wolfram.com/Bron-KerboschAlgorithm.html
- input: 'Undirected graph'
- output: all 'Maximal Clique'
- domain: 'Graph theory'
- applications: 'Computational chemistry'
- properties: 'not output-sensitive'

## Havel–Hakimi algorithm
- https://en.wikipedia.org/wiki/Havel%E2%80%93Hakimi_algorithm
- solves: 'Graph realization problem'
- domain: 'Graph theory'
- input: 'Collection of non-negative integers'
- output: 'simple graph'

## QR algorithm
- https://en.wikipedia.org/wiki/QR_algorithm
- is a: 'Eigenvalue algorithm'
- uses: 'QR decomposition'
- properties: 'numerically stable'
- modern implicit variant called: 'Francis algorithm'
- supersedes: 'LR algorithm' because of better numerical stability

## Weiler–Atherton clipping algorithm
- https://en.wikipedia.org/wiki/Weiler%E2%80%93Atherton_clipping_algorithm
- solves: 'Clipping problem' for polygons in 2 dimensions
- implemented in: 'Boost.Geometry'

## Sutherland–Hodgman algorithm
- https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm
- solves: 'Clipping problem' for polygons in 2 dimensions

## Vatti clipping algorithm
- https://en.wikipedia.org/wiki/Vatti_clipping_algorithm
- solves: 'Clipping problem' for polygons in 2 dimensions
- supports: 'complex polygons' (compared to 'Weiler–Atherton clipping algorithm' and 'Sutherland–Hodgman algorithm')
- used to implement: 'Boolean operations on polygons'
- implemented in: 'Angus Johnson's Clipper' (extended), 'General Polygon Clipper' (variant)

## Greiner–Hormann clipping algorithm
- https://en.wikipedia.org/wiki/Greiner%E2%80%93Hormann_clipping_algorithm
- solves: 'Clipping problem' for polygons in 2 dimensions
- used to implement: 'Boolean operations on polygons'

## Schutte's algorithm for polygon clipping
- paper: 'An Edge Labeling Approach to Concave Polygon Clipping (1995)'
- solves: 'Clipping problem' for polygons
- implemented in: 'Clippoly'

## PolyBoolean algorithm
- solves: 'Boolean operations on polygons'
- paper: 'Implementation of Boolean operations on sets of polygons in the plane (1998)'
- http://www.complex-a5.ru/polyboolean/comp.html
- extends: 'Schutte algorithm'

## Fast multipole method
- https://en.wikipedia.org/wiki/Fast_multipole_method
- domain: 'Computational electromagnetism'

## Fast Fourier transform method
- https://en.wikipedia.org/wiki/Fast_Fourier_transform
- output: 'Discrete Fourier transform'

## Cooley–Tukey FFT algorithm
- https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
- variation of: 'Fast Fourier transform'
- implemented in: 'Python numpy.fft, scipy.fftpack.fft'

## Kirkpatrick–Seidel algorithm
- https://en.wikipedia.org/wiki/Kirkpatrick%E2%80%93Seidel_algorithm
- paper: 'The Ultimate Planar Convex Hull Algorithm? (1983)'
- input: 'Collection of 2-d points'
- output: 'Convex hull'
- properties: 'output-sensitive'

## Chan's algorithm
- https://en.wikipedia.org/wiki/Chan%27s_algorithm
- input: 'Collection of 2-d or 3-d points'
- output: 'Convex hull'
- properties: 'output-sensitive'
- supersedes: 'Kirkpatrick–Seidel algorithm' (simpler and same complexity)

## Fortune and Hopcroft algorithm for the closest pair of points problem
- paper: 'A note on Rabin's nearest-neighbor algorithm (1978)'
- time complexity: O(n log(log(n))) (assumes constant time floor function)
- properties: 'deterministic'
- input: 'Collection of points'

## Khuller and Matias algorithm for the closest pair of points problem
- paper: 'A Simple Randomized Sieve Algorithm for the Closest-Pair Problem'
- time complexity: O(n)
- properties: 'randomized'
- input: 'Collection of points'

## Hoshen–Kopelman algorithm
- https://en.wikipedia.org/wiki/Hoshen%E2%80%93Kopelman_algorithm
- is a: 'Cluster analysis algorithm'
- input: 'regular network of bools'

## Fiorio's algorithm for connected-component labeling
- paper: 'Two linear time Union-Find strategies for image processing (1996)'
- solves: 'Connected-component labeling'
- implemented in: 'Python skimage.measure.label'
- version of: 'union-find algorithm'
- properties: 'two pass'

## Wu's algorithm for connected-component labeling
- paper: 'Two Strategies to Speed up Connected Component Labeling Algorithms (2005)'
- solves: 'Connected-component labeling'
- implemented in: 'opencv::connectedComponents'

## Fortune's algorithm
- https://en.wikipedia.org/wiki/Fortune%27s_algorithm
- is a: 'Sweep line algorithm'
- input: 'Collection of points'
- output: 'Voronoi diagram'
- time complexity: O(n log n)
- space complexity: O(n)

## Lloyd's algorithm
- https://en.wikipedia.org/wiki/Lloyd%27s_algorithm
- is a: 'Iterative method'
- input: 'Collection of points'
- output: 'Voronoi diagram'
- approximates: 'Centroidal Voronoi tessellation'

## Bowyer–Watson algorithm
- also called: 'Bowyer algorithm', 'Watson algorithm'
- https://en.wikipedia.org/wiki/Bowyer%E2%80%93Watson_algorithm
- is a: 'Incremental algorithm'
- input: 'Collection of points'
- output: 'Delaunay triangulation'
- time complexity (average): O(n log n)
- time complexity (worst): O(n^2)

## Quickhull algorithm
- https://en.wikipedia.org/wiki/Quickhull
- paper: 'The quickhull algorithm for convex hulls (1996)' (modern version)
- implemented in: 'Qhull'
- input: 'Collection of points'
- output: 'Convex hull'

## Simplified Memory Bounded A*
- also called: 'SMA*'
- https://en.wikipedia.org/wiki/SMA*
- based on: 'A*'
- solves: 'Shortest path problem'
- is a: 'heuristic algorithm'
- input: 'Graph'

## Thompson's construction algorithm
- also called: 'McNaughton-Yamada-Thompson algorithm'
- https://en.wikipedia.org/wiki/Thompson%27s_construction
- input: 'Regular expression'
- output: equivalent 'Nondeterministic finite automaton'
- domain: 'Automata theory'

## Daciuk's algorithm for constructing minimal acyclic finite state automata
- paper: 'Incremental Construction of Minimal Acyclic Finite-State Automata'
- input: 'List of strings'
- output: 'Minimal acyclic finite state automaton'

## Glushkov's construction algorithm
- https://en.wikipedia.org/wiki/Glushkov%27s_construction_algorithm
- input: 'Regular expression'
- output: equivalent 'Nondeterministic finite automaton'
- domain: 'Automata theory'

## Powerset construction
- also called: 'Rabin–Scott powerset construction'
- https://en.wikipedia.org/wiki/Powerset_construction
- input: 'Nondeterministic finite automaton'
- output: 'Deterministic finite automaton'
- domain: 'Automata theory'

## Kleene's algorithm
- https://en.wikipedia.org/wiki/Kleene%27s_algorithm
- input: 'Deterministic finite automaton'
- output: 'Regular expression'

## Hopcroft's algorithm
- https://en.wikipedia.org/wiki/DFA_minimization#Hopcroft's_algorithm
- solves: 'DFA minimization'
- input: 'Deterministic finite automaton'
- output: 'Minimal deterministic finite automaton'

## Moore's algorithm
- also called: 'Moore reduction procedure'
- https://en.wikipedia.org/wiki/Moore_reduction_procedure
- solves: 'DFA minimization'
- input: 'Deterministic finite automaton'
- output: 'Minimal deterministic finite automaton'

## Brzozowski's algorithm
- https://en.wikipedia.org/wiki/DFA_minimization#Brzozowski's_algorithm
- solves: 'DFA minimization'
- uses: 'Powerset construction'
- input: 'Deterministic finite automaton'
- output: 'Minimal deterministic finite automaton'

## Featherstone's algorithm
- https://en.wikipedia.org/wiki/Featherstone%27s_algorithm
- approximates: 'Rigid bodies dynamics' (subset of 'Classical mechanics')
- applications: 'Robotics', 'Physics engines', 'Game development'
- domain: 'Computational physics', 'Robot kinematics'
- cf. 'Lagrange multiplier method'
- input: 'Kinematic chain' (Collection of points and constraints)

## Fireworks algorithm
- https://en.wikipedia.org/wiki/Fireworks_algorithm
- see: 'Swarm intelligence'
- domain: 'mathematical optimization'
- input: 'Function'

## Sequence step algorithm
- https://en.wikipedia.org/wiki/Sequence_step_algorithm
- https://www.planopedia.com/sequence-step-algorithm/
- applications: 'Scheduling'

## Fast folding algorithm
- https://en.wikipedia.org/wiki/Fast_folding_algorithm
- paper: 'Fast folding algorithm for detection of periodic pulse trains (1969)'
- applications: 'Time series analysis', 'Pulsar detection'
- domain: 'Signal processing'
- input: 'Buffered list of floats'?

## Faugère's F4 algorithm
- https://en.wikipedia.org/wiki/Faug%C3%A8re%27s_F4_and_F5_algorithms
- domain: 'Algebra'
- output: 'Gröbner basis'
- implemented in: 'Python SymPy'
- input: 'ideal of a multivariate polynomial ring'

## Faugère's F5 algorithm
- https://en.wikipedia.org/wiki/Faug%C3%A8re%27s_F4_and_F5_algorithms
- domain: 'Algebra'

## Hirschberg–Sinclair algorithm
- https://en.wikipedia.org/wiki/Hirschberg%E2%80%93Sinclair_algorithm
- paper: 'Decentralized extrema-finding in circular configurations of processors'
- solves: 'Leader election problem'
- properties: 'distributed'

## Gale–Church alignment algorithm
- https://en.wikipedia.org/wiki/Gale%E2%80%93Church_alignment_algorithm
- paper: 'A Program for Aligning Sentences in Bilingual Corpora'
- domain: 'Computational linguistics'
- applications: 'Sentence alignment'
- input: 'pair of list of sentences'

## Beier–Neely morphing algorithm
- https://en.wikipedia.org/wiki/Beier%E2%80%93Neely_morphing_algorithm
- applications: 'Image processing'
- domain: 'Computer graphics'
- input: 'pair of images'

## Kalman filter
- also called: 'linear quadratic estimation'
- https://en.wikipedia.org/wiki/Kalman_filter
- http://mathworld.wolfram.com/KalmanFilter.html
- domain: 'Control theory'
- applications: 'guidance, navigation, and control', 'time series analysis', 'Trajectory optimization', 'Computer vision', 'Object tracking'
- solves: 'Linear–quadratic–Gaussian control problem'

# Filters

## Wiener filter
- https://en.wikipedia.org/wiki/Wiener_filter
- implemented in: 'scipy.signal.wiener'
- domain: 'Signal processing'
- applications: 'System identification', 'Deconvolution', 'Noise reduction', 'Signal detection'
- is a: 'Linear filter'

## Canny edge detector
- https://en.wikipedia.org/wiki/Canny_edge_detector
- domain: 'image processing'
- applications: 'Edge detection'

## Gabor filter
- https://en.wikipedia.org/wiki/Gabor_filter
- is a: 'Linear filter'
- domain: 'Image processing'
- applications: 'localize and extract text-only regions', 'facial expression recognition', 'pattern analysis'

## Sobel operator
- https://en.wikipedia.org/wiki/Sobel_operator
- is a: 'Discrete differentiation operator'
- domain: 'Image processing'
- applications: 'Edge detection'
- implemented in: 'scipy.ndimage.sobel'

## Prewitt operator
- https://en.wikipedia.org/wiki/Prewitt_operator
- implemented in: 'scipy.ndimage.prewitt'
- applications: 'Edge detection'
- domain: 'Image processing'
- is a: 'Discrete differentiation operator'

## Scharr operator
- dissertation: 'Optimale Operatoren in der Digitalen Bildverarbeitung'
- domain: 'Image processing'
- applications: 'Edge detection'

## Kuwahara filter
- https://en.wikipedia.org/wiki/Kuwahara_filter
- https://reference.wolfram.com/language/ref/KuwaharaFilter.html
- is a: 'Non-linear filter'
- applications: 'adaptive noise reduction', 'medical imaging', 'fine-art photography'
- disadvantages: 'create block artifacts'
- domain: 'Image processing'

# Methods

## Locality-sensitive hashing
- https://en.wikipedia.org/wiki/Locality-sensitive_hashing
- used for: 'Approximate nearest neighbor search'
- is a: 'data-independent method'

## Locality-preserving hashing
- https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Locality-preserving_hashing
- is a: 'data-dependent method'

## Dynamic programming
- https://en.wikipedia.org/wiki/Dynamic_programming
- exploits: 'optimal substructure'

## Brute force
- also called: 'Exhaustive search'
- https://en.wikipedia.org/wiki/Brute-force_search
- http://mathworld.wolfram.com/ExhaustiveSearch.html
- variant: 'British Museum algorithm'

## Backtracking
- https://en.wikipedia.org/wiki/Backtracking

# General problems

## DFA minimization
- https://en.wikipedia.org/wiki/DFA_minimization

## Nearest neighbor search
- https://en.wikipedia.org/wiki/Nearest_neighbor_search
- also called: 'post-office problem'
- solved exactly by: 'Space partitioning', 'Linear search'
- solved approximatly by: 'Hierarchical Navigable Small World graphs', 'Locality-sensitive hashing', 'Cover tree', 'Vector quantization'
- implemented by: 'spotify/annoy', 'C++ ANN', 'nmslib/hnsw', 'nmslib/nmslib'

## Approximate nearest neighbor search
- book: 'Handbook of Discrete and Computational Geometry'
- https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor

## Cycle decomposition
- https://en.wikipedia.org/wiki/Cycle_decomposition_(group_theory)

## RSA problem
- https://en.wikipedia.org/wiki/RSA_problem
- see: 'Integer factorization'

## Integer factorization
- https://en.wikipedia.org/wiki/Integer_factorization
- applications: 'cryptography'
- domain: 'Number theory'

## Envy-free item assignment
- https://en.wikipedia.org/wiki/Envy-free_item_assignment
- type of: 'Fair item assignment', 'Fair division'

## Connected-component labeling
- https://en.wikipedia.org/wiki/Connected-component_labeling
- domain: 'Graph theory'
- applications: 'Computer vision'

## Strongly connected component
- https://en.wikipedia.org/wiki/Strongly_connected_component
- used for 'Dulmage–Mendelsohn decomposition'
- book: 'Introduction to Algorithms'
- domain: 'Graph theory'

## Greatest common divisor
- https://en.wikipedia.org/wiki/Greatest_common_divisor
- domain: 'Number theory'

## Topological sorting
- https://en.wikipedia.org/wiki/Topological_sorting
- implemented in posix tsort.
- only possible on 'directed acyclic graph'

## Travelling salesman problem
- https://en.wikipedia.org/wiki/Travelling_salesman_problem
- solved by: 'Concorde TSP Solver' application
- solved by: 'Approximate global optimization'
- hardness: NP-hard

## Minimum spanning tree
- https://en.wikipedia.org/wiki/Minimum_spanning_tree
- http://algorist.com/problems/Minimum_Spanning_Tree.html
- book: 'Introduction to Algorithms'
- solved by 'Kruskal's algorithm', 'Prim's algorithm'
- unique solution
- applications: 'Network design', 'Image segmentation', 'Cluster analysis'

## Second-best minimum spanning tree
- book: 'Introduction to Algorithms'
- solution need not be unique
- variant of: 'Minimum spanning tree'

## Bottleneck spanning tree
- book: 'Introduction to Algorithms'
- variant of: 'Minimum spanning tree'
- a 'minimum spanning tree' is a 'bottleneck spanning tree'

## spanning-tree verification
- book: 'Introduction to Algorithms'
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
- finds the shortest path for all pairs of vectices in a graph
- solved by: 'Floyd–Warshall algorithm', 'Johnson's algorithm'
- domain: 'graph theory'
- properties: 'optimal substructure'

## Approximate string matching
- https://en.wikipedia.org/wiki/Approximate_string_matching
- paper: 'Fast Approximate String Matching in a Dictionary'
- applications: 'spell checking', 'nucleotide sequence matching'

## Lowest common ancestor (LCA)
- https://en.wikipedia.org/wiki/Lowest_common_ancestor
- domain: 'Graph theory'

## Longest common substring problem
- https://en.wikipedia.org/wiki/Longest_common_substring_problem
- cf. 'Longest common subsequence problem'
- properties: 'optimal substructure'
- solutions: 'Generalized suffix tree'
- domain: 'Combinatorics'

## Longest common subsequence problem
- https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
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
- applications: sparse matrix compression
- domain: 'combinatorics'

## Longest increasing subsequence
- https://en.wikipedia.org/wiki/Longest_increasing_subsequence
- domain: 'Combinatorics'
- properties: 'optimal substructure'

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
- domain: 'graph theory'

## Minimum-cost flow problem
- https://en.wikipedia.org/wiki/Minimum-cost_flow_problem

## Shortest pair of edge disjoint paths
- special case of: 'Minimum-cost flow problem'
- applications: 'Routing'

## Polynomial of best approximation
- https://www.encyclopediaofmath.org/index.php/Polynomial_of_best_approximation
- domain: 'Approximation theory'

## Point location problem
- https://en.wikipedia.org/wiki/Point_location
- domain: 'Computational geometry'

## Point-in-polygon problem
- https://en.wikipedia.org/wiki/Point_in_polygon
- special case of: 'Point location problem'
- domain: 'Computational geometry'

## Convex hull
- also called: 'minimum convex polygon'
- book: 'Handbook of Discrete and Computational Geometry'
- https://en.wikipedia.org/wiki/Convex_hull
- http://mathworld.wolfram.com/ConvexHull.html
- domain: 'Computational geometry'
- implemented in: 'Mathematica ConvexHullMesh', 'Python scipy.spatial.ConvexHull' (using Quickhull)

## Halfspace intersection problem
- book: 'Handbook of Discrete and Computational Geometry'
- implemented in: 'scipy.spatial.HalfspaceIntersection', 'Qhull'
- https://en.wikipedia.org/wiki/Half-space_(geometry)

## Closest pair of points problem
- https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
- domain: 'Computational geometry'
- brute force time complexity: O(n^2) for a set of points of size n

## Largest empty sphere problem
- https://en.wikipedia.org/wiki/Largest_empty_sphere
- domain: 'Computational geometry'
- special cases can be solved using 'Voronoi diagram' in optimal time O(n log(n))

## Hierarchical clustering
- https://en.wikipedia.org/wiki/Hierarchical_clustering
- applications: 'Data mining', 'Paleoecology'
- domain: 'Statistics'

## Delaunay triangulation
- https://en.wikipedia.org/wiki/Delaunay_triangulation
- http://mathworld.wolfram.com/DelaunayTriangulation.html
- book: 'Handbook of Discrete and Computational Geometry'
- paper: 'Sur la sphère vide. A la mémoire de Georges Voronoï'
- domain: 'Geometry'
- is dual graph of: 'Voronoi diagram'
- related: 'Euclidean minimum spanning tree'
- implemented in: 'scipy.spatial.Delaunay' (using Qhull)

## Voronoi diagram
- also called: 'Voronoi tessellation', 'Voronoi decomposition', 'Voronoi partition'
- https://en.wikipedia.org/wiki/Voronoi_diagram
- http://mathworld.wolfram.com/VoronoiDiagram.html
- book: 'Handbook of Discrete and Computational Geometry'
- paper: 'Nouvelles applications des paramètres continus à la théorie de formes quadratiques'
- domain: 'Geometry'
- is dual graph of: 'Delaunay triangulation'
- applications: 'Space partitioning', 'biological structure modelling', 'growth patterns in ecology', 'Epidemiology'
- related: 'Closest pair of points problem', 'Largest empty sphere problem'
- implemented in: 'scipy.spatial.Voronoi' (using Qhull), 'scipy.spatial.SphericalVoronoi'

## Subset sum problem
- https://en.wikipedia.org/wiki/Subset_sum_problem
- hardness: NP-complete

## Minimal Perfect Hashing
- http://iswsa.acm.org/mphf/index.html
- https://en.wikipedia.org/wiki/Perfect_hash_function#Minimal_perfect_hash_function

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

## Optimal solutions for Rubik's Cube
- calculate maximal turn numbers by: diameters of the corresponding Cayley graphs of the Rubik's Cube group
- use 'Korf's algorithm' to calculate optimal solution
- implementation: http://www.cflmath.com/~reid/Rubik/optimal_solver.html
