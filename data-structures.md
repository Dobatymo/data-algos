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
- can be: 'Persistent data structure'

## Rope
- https://en.wikipedia.org/wiki/Rope_(data_structure)
- similar to: 'deque', 'Gap buffer'
- is a: 'Binary tree'
- can be a: 'Persistent data structure'
- implemented in (libraries): 'libstdc++ rope'
- applications: 'Constant folding'
- domain: 'Compiler'

## Threaded binary tree
- also called: 'TBST'
- https://en.wikipedia.org/wiki/Threaded_binary_tree
- is a: 'Binary tree'
- properties: 'can be traversed in-order in constant space'
- implemented in (libraries): 'libavl'

## AVL tree
- paper: 'An algorithm for the organization of information' (1962) <>
- https://en.wikipedia.org/wiki/AVL_tree
- https://www.geeksforgeeks.org/avl-tree-set-1-insertion/
- is a: 'Self-balancing binary search tree'

## Gap buffer
- https://en.wikipedia.org/wiki/Gap_buffer
- applications: 'Text editor'

## Piece table
- https://en.wikipedia.org/wiki/Piece_table
- survey paper: 'Data Structures for Text Sequences' (1998)
- applications: 'Text editor'
- similar: 'Gap buffer'
- implemented by (applications): 'Visual Studio Code'

## Hash table
- https://en.wikipedia.org/wiki/Hash_table
- implements: 'Set', 'Multiset', 'Map'
- used to implement: 'Python dict', 'C++ std::unordered_set', 'C++ std::unordered_multiset', 'C++ std::unordered_map'
- probably the most important data structure in Python
- has basically perfect complexity for a good hash function. (Minimal) perfect hashing can be used if the keys are known in advance.
- ordering depends on implementation (python 3.7 garuantees preservation of insertion order, whereas C++ as the name says does not define any ordering)
- even though hashing is constant in time, it might still be slow. also the hash consumes space.
- time complexity (search, insert, delete) (average): O(1)
- time complexity (search, insert, delete) (worst): O(n)
- space complexity (average): O(n)
- space complexity (worst): O(n) # often a lot of padding space is needed...

## Binary search tree
- https://en.wikipedia.org/wiki/Binary_search_tree
- implements: 'Set', 'Multiset', 'Map'
- used to implement: 'C++ std::set', 'C++ std::multiset'
- is a: 'rooted binary tree'
- variants: 'Huffman tree'

## Zipper
- original paper: 'FUNCTIONAL PEARL: The Zipper' (1997)
- https://en.wikipedia.org/wiki/Zipper_(data_structure)
- generalization of: 'Gap buffer'
- more a general technique than a data structure

## Fibonacci heap
- paper: 'Fibonacci heaps and their uses in improved network optimization algorithms' (1987)
- https://en.wikipedia.org/wiki/Fibonacci_heap
- book: 'Introduction to Algorithms'
- implements: 'Priority queue'

## Pairing heap
- original paper: 'The pairing heap: A new form of self-adjusting heap' (1986)
- https://en.wikipedia.org/wiki/Pairing_heap
- implements: 'Priority queue'

## Binary heap
- paper: 'Algorithm 232 Heapsort' (1964) <https://doi.org/10.1145/512274.512284>
- https://en.wikipedia.org/wiki/Binary_heap
- implements: 'Priority queue'
- variant implemented in: 'Python heapq', 'C++ std::make_heap, push_heap and pop_heap'
- usually implemented as: 'Implicit data structure'

## Segment tree
- also called: 'Statistic tree'
- https://en.wikipedia.org/wiki/Segment_tree
- https://cp-algorithms.com/data_structures/segment_tree.html
- https://www.hackerearth.com/practice/data-structures/advanced-data-structures/segment-trees/tutorial/
- applications: 'Range minimum query'
- implemented in (libraries): 'Python segment-tree'

## Interval tree
- https://en.wikipedia.org/wiki/Interval_tree
- variants: 'centered interval tree', 'augmented interval tree'
- applications: 'windowing queries'
- properties: 'output-sensitive'
- implemented in (libraries): 'Python intervaltree'

## Range query tree (?)
- also called: 'RQT'
- https://en.wikipedia.org/wiki/Range_query_tree
- is a: 'perfect binary tree'
- don't confuse with: 'Segment tree', 'Range tree'
- is this the same as?: 'Segment tree'
- similar: 'Fenwick tree'

## Nested Containment List
- paper: 'Nested Containment List (NCList): a new algorithm for accelerating interval query of genome alignment and interval databases' (2007) <https://doi.org/10.1093/bioinformatics/btl647>
- implemented in (libraries): 'Python ncls'
- applications: 'interval overlap queries'

## Longest common prefix array
- also called: 'LCP array'
- https://en.wikipedia.org/wiki/LCP_array
- applications: 'Longest common prefix problem'

## Binary decision diagram
- also called: 'BDD'
- https://en.wikipedia.org/wiki/Binary_decision_diagram
- compressed representation of sets or relations
- implements: 'Boolean function'
- is a: 'rooted, directed, acyclic graph'
- applications: 'Computer-aided design', 'Formal verification', 'Fault tree analysis', 'Private information retrieval'

## Zero-suppressed decision diagram
- https://en.wikipedia.org/wiki/Zero-suppressed_decision_diagram
- type of: 'Binary decision diagram'

## Propositional directed acyclic graph
- also called: 'PDAG'
- https://en.wikipedia.org/wiki/Propositional_directed_acyclic_graph
- implements: 'Boolean function'
- generalization of: 'Binary decision diagram'

## Trie
- also called: 'digital tree'
- paper: 'File searching using variable length keys' (1959) <https://doi.org/10.1145/1457838.1457895>
- https://en.wikipedia.org/wiki/Trie
- tree structure
- set and map characteristics
- keys are sequences (eg. strings)
- allow for prefix search (eg. find all strings that start with 'a', or find the longest prefix of 'asd')
- if implemented with hashmaps, indexing by key can be done in O(sequence-length) independent of tree size
- not very space efficient. common prefixed only have to be stored once, but pointers to next element of sequence uses more memory than what is saved.
- for a more space efficient data structure see MAFST

## WPL tree
- also called: 'Weighted Path Length Tree'
- paper: 'Weighting Without Waiting: the Weighted Path Length Tree' (1991) <https://doi.org/10.1093/comjnl/34.5.444>

## PQ tree
- paper: 'Testing for the consecutive ones property, interval graphs, and graph planarity using PQ-tree algorithms' (1976) <https://doi.org/10.1016/S0022-0000(76)80045-1>
- https://en.wikipedia.org/wiki/PQ_tree

## Radix tree
- also called: 'Radix trie'
- https://en.wikipedia.org/wiki/Radix_tree
- is a: 'Binary trie'

## Adaptive radix tree
- also called: 'ART'
- paper: 'The adaptive radix tree: ARTful indexing for main-memory databases' (2013) <https://doi.org/10.1109/ICDE.2013.6544812>

## X-fast trie
- paper: 'Log-logarithmic worst-case range queries are possible in space Θ(N)' (1983) <https://doi.org/10.1016/0020-0190(83)90075-3>
- https://en.wikipedia.org/wiki/X-fast_trie
- is a: 'Trie'

## Y-fast trie
- paper: 'Log-logarithmic worst-case range queries are possible in space Θ(N)' (1983) <https://doi.org/10.1016/0020-0190(83)90075-3>
- https://en.wikipedia.org/wiki/Y-fast_trie
- is a: 'Trie'

## Quadtree
- https://en.wikipedia.org/wiki/Quadtree
- is a: 'Tree', 'Space-partitioning tree'
- can be implemented as: 'Implicit data structure'
- variant: 'Region quadtree'
- applications: 'Image processing', 'Connected-component labeling', 'Mesh generation'

## Red-black tree
- https://en.wikipedia.org/wiki/Red%E2%80%93black_tree
- book: 'Introduction to Algorithms'
- height-balanced binary tree
- better for insert/delete (compared to 'AVL tree')
- used to implement: 'C++ std::map', 'Java TreeMap'
- used by: MySQL

## AVL tree
- original paper: 'An algorithm for the organization of information' (1962)
- https://en.wikipedia.org/wiki/AVL_tree
- https://xlinux.nist.gov/dads/HTML/avltree.html
- is a: 'binary tree'
- properties: 'height-balanced'
- stricter balanced and thus better for lookup (compared to 'Red-black tree')

## Treap
- original paper: 'Randomized search trees' (1989)
- paper: 'Randomized search trees' (1996) <https://doi.org/10.1007/BF01940876>
- primary source: https://faculty.washington.edu/aragon/pubs/rst89.pdf
- primary source: https://faculty.washington.edu/aragon/pubs/rst96.pdf
- secondary source: https://en.wikipedia.org/wiki/Treap
- secondary source: https://brilliant.org/wiki/treaps/
- secondary source: https://www.geeksforgeeks.org/treap-a-randomized-binary-search-tree/
- book: 'Open Data Structures' <https://opendatastructures.org/>
- randomly ordered binary tree
- log(N) lookup even for insertion of items in non-random order
- heap like feature
- used to implement dictionary in LEDA

## Implicit treap
- also called: 'Indexed list', 'Treap with implicit keys", 'Cartesian tree with implicit keys"
- paper?: 'Efficient Data Structures and a New Randomized Approach for Sorting Signed Permutations by Reversals' (2003) <https://doi.org/10.1007/3-540-44888-8_13>
- secondary source: https://cp-algorithms.com/data_structures/treap.html#toc-tgt-6
- implemented in: 'https://hackage.haskell.org/package/treap'
- is a: 'Implicit data structure'

## Fenwick tree
- also called: 'Binary indexed tree'
- paper: 'A new data structure for cumulative frequency tables' (1994) <https://doi.org/10.1002/spe.4380240306>
- secondary source: https://en.wikipedia.org/wiki/Fenwick_tree
- secondary source: https://jornhub.dev/articles/fenwick-trees/
- https://brilliant.org/wiki/fenwick-tree/
- https://www.geeksforgeeks.org/binary-indexed-tree-or-fenwick-tree-2/
- usually implemented as: 'Implicit data structure'
- applications: 'Prefix sum'
- implemented in (libraries): 'Python dstein64/fenwick'

## Scapegoat tree
- also called: 'General balanced tree'
- paper: 'Improving partial rebuilding by using simple balance criteria' (1989) <https://doi.org/10.1007/3-540-51542-9_33>
- paper: 'Scapegoat trees' (1993) <https://dl.acm.org/doi/10.5555/313559.313676>
- primary source: https://people.csail.mit.edu/rivest/pubs/GR93.pdf
- secondary source: https://en.wikipedia.org/wiki/Scapegoat_tree
- book: 'Open Data Structures'
- is a: 'Binary search tree'
- properties: 'Self-balancing'

## BK-tree
- paper: 'Some approaches to best-match file searching' (1973) <https://doi.org/10.1145/362003.362025>
- https://en.wikipedia.org/wiki/BK-tree
- is a: 'Space-partitioning tree', 'Metric tree'
- applications: approximate string matching

## Splay tree
- paper: 'Self-adjusting binary search trees' (1985) <https://doi.org/10.1145/3828.3835>
- https://en.wikipedia.org/wiki/Splay_tree
- is a: 'Binary search tree'
- properties: 'self-optimizing'
- applications: 'Caching', 'Garbage collection'
- disadvantages: even concurrent reads require synchronization
- unsolved problem: 'Do splay trees perform as well as any other binary search tree algorithm?'

## k-d tree
- also called: 'k-dimensional tree'
- paper: 'Multidimensional binary search trees used for associative searching' (1975) <https://doi.org/10.1145/361002.361007>
- https://en.wikipedia.org/wiki/K-d_tree
- is a: 'Space-partitioning tree'
- applications: 'Range searching', 'Nearest neighbor search', 'Kernel density estimation'
- for high dimensions should be: N >> 2^k, where N is the number of nodes and k is the number of dimensions
- solves 'Recursive partitioning', 'Klee's measure problem', 'Guillotine problem'
- implemented in: 'Python scipy.spatial.KDTree, sklearn.neighbors.KDTree, Bio.PDB.kdtrees.KDTree'

## Range tree
- paper: 'Multidimensional divide-and-conquer' (1980) <https://doi.org/10.1145/358841.358850>
- https://en.wikipedia.org/wiki/Range_tree
- applications: 'Range searching'

## Compressed range trees
- paper: 'A Functional Approach to Data Structures and Its Use in Multidimensional Searching' (1988) <https://doi.org/10.1137/0217026>
- applications: 'Range searching'
- space complexity: O(n)
- query complexity: O(log n + k*log^ε n)

## Cartesian tree
- https://en.wikipedia.org/wiki/Cartesian_tree
- applications: 'Range searching', 'Range minimum query'

## Iliffe vector
- https://en.wikipedia.org/wiki/Iliffe_vector
- used to implement multi-dimensional arrays

## Dope vector
- https://en.wikipedia.org/wiki/Dope_vector
- used to implement arrays

## van Emde Boas tree
- also called: 'vEB tree', 'van Emde Boas priority queue'
- paper: 'Preserving order in a forest in less than logarithmic time' (1975)
- https://en.wikipedia.org/wiki/Van_Emde_Boas_tree
- https://xlinux.nist.gov/dads/HTML/vanemdeboas.html
- book: 'Introduction to Algorithms'
- Multiway tree
- implement ordered maps with integer keys
- implement priority queues
- see 'Integer sorting'

## Skip list
- paper: 'Concurrent Maintenance of Skip Lists' (1998)
- https://en.wikipedia.org/wiki/Skip_list
- https://xlinux.nist.gov/dads/HTML/skiplist.html
- is a: 'probabilistic data structure', 'ordered linked list'
- basically binary trees converted to linked lists with additional information
- allows for fast search of sorted sequences
- implemented by: 'Lucence', 'Redis'
- implemented in: 'Java ConcurrentSkipListMap'
- applications: 'Moving median'
- space complexity (average): O(n)
- space complexity (worst): O(n log n)
- time complexity (search, insert, delete) (average): O(log n)
- time complexity (search, insert, delete) (worst): O(n)

## Interval Skip List
- paper: 'The interval skip list: A data structure for finding all intervals that overlap a point' (1991) <https://doi.org/10.1007/BFb0028258>
- implemented in (libraries): 'CGAL::Interval_skip_list', 'Dart kseo/interval_skip_list'
- applications: 'stabbing queries'

## AABB tree
- also called: 'Axis-aligned bounding box tree'
- https://en.wikipedia.org/wiki/Minimum_bounding_box#Axis-aligned_minimum_bounding_box
- implemented in (libraries): 'CGAL::AABB_tree'
- implemented in (libraries): 'Python aabbtree'

## Finite-state machine
- also called: 'FSM', 'Finite-state automaton', 'FSA'
- https://en.wikipedia.org/wiki/Finite-state_machine

## Finite-state transducer
- https://en.wikipedia.org/wiki/Finite-state_transducer
- is a: 'Finite-state machine'
- implemented in: 'OpenFst'

# Deterministic finite automaton
- also called: 'DFA', 'Deterministic finite acceptor', 'Deterministic finite state machine', 'DFSM', 'Deterministic finite state automaton', 'DFSA'
- https://en.wikipedia.org/wiki/Deterministic_finite_automaton
- is a: 'Finite-state machine'

## Deterministic acyclic finite state automaton
- also called: 'DAFSA', 'deterministic acyclic finite state acceptor', 'DAWG'
- https://en.wikipedia.org/wiki/Deterministic_acyclic_finite_state_automaton
- used to implement ordered sets
- is a: 'Deterministic finite automaton'

## Minimal acyclic finite state automaton
- also called: 'MAFSA', 'Minimal acyclic finite state acceptor'
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
- paper: 'Organization and maintenance of large ordered indices' (1970) <https://doi.org/10.1145%2F1734663.1734671>
- secondary source: https://en.wikipedia.org/wiki/B-tree
- secondary source: https://xlinux.nist.gov/dads/HTML/btree.html
- book: 'Introduction to Algorithms' (1990)
- used to implement lots of databases and filesystems
- self-balancing
- non-binary

## B+ tree
- secondary source: https://en.wikipedia.org/wiki/B%2B_tree
- secondary source: https://xlinux.nist.gov/dads/HTML/bplustree.html
- applications: filesystems, 'Range searching', block-oriented data retrieval
- k-ary tree
- used by: Relational database management systems like Microsoft SQL Server, Key–value database management systems like CouchDB
- based on: 'B-tree'

## UB-tree
- also called: 'Universal B-Tree'
- original paper: 'The universal B-tree for multidimensional indexing: General concepts' (1997) <https://doi.org/10.1007/3-540-63343-X_48>
- paper: 'Integrating the UB-Tree into a Database System Kernel' (2000) <https://dl.acm.org/doi/10.5555/645926.671872>
- primary source: http://www.vldb.org/conf/2000/P263.pdf
- secondary source: https://en.wikipedia.org/wiki/UB-tree
- secondary source: https://xlinux.nist.gov/dads/HTML/universalBTree.html
- is a: 'Self-balancing search tree'
- based on: 'B+ tree'


## SS-Tree (Similarity search tree)
- paper: 'Similarity indexing with the SS-tree' (1996) <https://doi.org/10.1109/ICDE.1996.492202>
- applications: 'Similarity indexing', 'Nearest neighbor search'

## Cover tree
- paper: 'Cover trees for nearest neighbor' (2006) <https://doi.org/10.1145/1143844.1143857>
- primary source: https://hunch.net/~jl/projects/cover_tree/cover_tree.html
- primary source: https://www.cc.gatech.edu/~isbell/reading/papers/cover-tree-icml.pdf
- secondary source: https://en.wikipedia.org/wiki/Cover_tree
- applications: nearest neighbor search
- is a: Metric tree
- implemented in: 'DNCrane/Cover-Tree', 'loehndorf/covertree'

## M-tree
- paper: 'M-tree: An Efficient Access Method for Similarity Search in Metric Spaces' (1997) <https://dl.acm.org/doi/10.5555/645923.671005>
- primary source: http://www-db.deis.unibo.it/Mtree/main.html
- secondary source: https://en.wikipedia.org/wiki/M-tree
- better disk storage characteristics (because shallower) than 'Ball tree'
- uses: 'nearest neighbor search'
- is a: 'Space-partitioning tree', 'Metric tree'

## Vantage-point tree
- original paper: 'Satisfying general proximity / similarity queries with metric trees' (1991)
- paper: 'Data structures and algorithms for nearest neighbor search in general metric spaces' (1993)
- https://en.wikipedia.org/wiki/Vantage-point_tree
- is a: 'Space-partitioning tree', 'Metric tree'
- specialisation of: 'Multi-vantage-point tree'

## Ball tree
- https://en.wikipedia.org/wiki/Ball_tree
- is a: 'Space-partitioning tree', 'Metric tree', 'Binary tree'
- applications: nearest neighbor search, kernel density estimation, N-point correlation function calculations, generalized N-body Problems.
- specialisation of: M-Tree
- similar: Vantage-point tree
- implemented in: 'sklearn.neighbors.BallTree'
- algorithms for construction: 'Five Balltree Construction Algorithms'

## Winged edge
- paper: 'A polyhedron representation for computer vision' (1975)
- https://en.wikipedia.org/wiki/Winged_edge
- applications: 'Computer graphics', 'Boundary representation'

## Adjacency list
- https://en.wikipedia.org/wiki/Adjacency_list
- book: 'Open Data Structures'
- implements: 'Graph'
- implemented in: 'boost::adjacency_list'

## Incidence matrix
- https://en.wikipedia.org/wiki/Incidence_matrix
- implements: 'Graph'

## R-tree
- paper: 'R-trees: a dynamic index structure for spatial searching' (1984) <https://doi.org/10.1145/971697.602266>
- https://en.wikipedia.org/wiki/R-tree
- applications: 'Spatial index', 'Range searching', 'Nearest neighbor search'
- used by (applications): 'IBM Informix'

## R+ tree
- paper: 'The R+-Tree: A Dynamic Index for Multi-Dimensional Objects' (1987)
- https://en.wikipedia.org/wiki/R%2B_tree
- variant of: 'R-tree'
- applications: 'Spatial index'

## R* tree
- paper: 'The R*-tree: an efficient and robust access method for points and rectangles' (1990)
- https://en.wikipedia.org/wiki/R*_tree
- variant of: 'R-tree'
- applications: 'Spatial index'

## Hash tree
- https://en.wikipedia.org/wiki/Hash_tree_(persistent_data_structure)
- is a: 'Persistent data structure'
- implements: 'Set', 'Map'

## Hash array mapped trie
- also called: 'HAMT'
- https://en.wikipedia.org/wiki/Hash_array_mapped_trie
- specialisation of: 'Hash tree'
- is a: 'Persistent data structure'
- implements: 'Map'

## Merkle tree
- also called: 'Hash tree'
- paper: 'A Digital Signature Based on a Conventional Encryption Function' (1987)
- https://en.wikipedia.org/wiki/Merkle_tree
- https://xlinux.nist.gov/dads/HTML/MerkleTree.html
- https://brilliant.org/wiki/merkle-tree/
- is a: 'Tree'
- applications: 'Hash-based cryptography', 'peer-to-peer networks'
- implementation: 'Tiger tree hash'

## Generalized suffix tree
- https://en.wikipedia.org/wiki/Generalized_suffix_tree
- build using: 'Ukkonen's algorithm', 'McCreight's algorithm'

## Disjoint-set data structure
- also called: 'union–find data structure'
- https://en.wikipedia.org/wiki/Disjoint-set_data_structure
- book: 'Introduction to Algorithms'
- is a: 'Multiway tree'
- applications: connected components of an undirected graph
- implemented in: 'boost::graph::incremental_components'
- used for: 'Kruskal's algorithm'

## HAT-trie
- paper: 'HAT-trie: a cache-conscious trie-based data structure for strings' (2007)
- https://en.wikipedia.org/wiki/HAT-trie
- implemented in: 'Python pytries/hat-trie'
- implements: 'Ordered map'
- variant of: 'Radix tree'
- properties: 'cache friendly'

## Double-Array Trie
- also called: 'DATrie'
- paper: 'An Efficient Digital Search Algorithm by Using a Double-Array Structure' (1989)
- implemented in: 'pytries/datrie', 'libdatrie'

## Ternary search tree
- also called: 'TST'
- paper: 'Fast algorithms for sorting and searching strings' (1997)
- https://en.wikipedia.org/wiki/Ternary_search_tree
- type of: 'Trie'
- applications: 'Nearest neighbor search', 'spell-checking', 'auto-completion'
- implements: 'Map'

## Hierarchical Navigable Small world graphs
- also called: 'Hierarchical NSW', 'HNSW'
- paper: 'Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs' (2018) <https://doi.org/10.1109/TPAMI.2018.2889473>
- applications: 'Approximate nearest Neighbor Search'

## Corner Stitching
- paper: 'Corner Stitching: a Data Structuring Technique for VLSI Layout Tools (1982)'
- applications: 'Very Large Scale Integration'

## Difference list
- https://en.wikipedia.org/wiki/Difference_list
- implemented in: 'Haskell'

## Soft heap
- https://en.wikipedia.org/wiki/Soft_heap
- approximates: 'Priority queue'

## Binomial heap
- paper: 'A data structure for manipulating priority queues (1978)'
- https://en.wikipedia.org/wiki/Binomial_heap
- implements: 'Mergeable heap'

## Brodal queue
- paper: 'Worst-Case Efficient Priority Queues (1996)'
- https://en.wikipedia.org/wiki/Brodal_queue
- implements: 'heap', 'priority queue'

## Bloom filter
- paper: 'Space/time trade-offs in hash coding with allowable errors (1970)'
- https://en.wikipedia.org/wiki/Bloom_filter
- properties: 'probabilistic'
- implements: 'Set'
- applications: 'caching strategies', 'database query optimization', 'rate-limiting', 'data synchronization', 'chemical structure searching'
- is a: 'probabilistic data structure'

## Count–min sketch
- also called: 'CM sketch'
- paper: 'An improved data stream summary: the count-min sketch and its applications' (2005) <https://doi.org/10.1016/j.jalgor.2003.12.001>
- https://en.wikipedia.org/wiki/Count%E2%80%93min_sketch
- is a: 'probabilistic data structure'
- used by: 'MIDAS anomaly detection algorithm'
- properties: 'sub-linear space usage'

## Optimal binary search tree
- https://en.wikipedia.org/wiki/Optimal_binary_search_tree
- is a: 'Binary search tree'

## Order statistic tree
- https://en.wikipedia.org/wiki/Order_statistic_tree
- variant of: 'Binary search tree'
- additional interface: find the i'th smallest element stored in the tree, find the rank of element x in the tree, i.e. its index in the sorted list of elements of the tree

## Exponential tree
- original paper: 'Faster deterministic sorting and searching in linear space' (1996)
- https://en.wikipedia.org/wiki/Exponential_tree
- variant of: 'Binary search tree'

## Log-structured merge-tree
- also called: 'LSM tree'
- original paper: 'The log-structured merge-tree (LSM-tree)' (1996)
- https://en.wikipedia.org/wiki/Log-structured_merge-tree
- used by: 'Apache Cassandra', 'Apache HBase', 'Bigtable', 'RocksDB'
- applications: 'Transaction log'

## Wavelet tree
- paper: 'High-order entropy-compressed text indexes (2003)'
- https://en.wikipedia.org/wiki/Wavelet_Tree
- properties: 'succinct'
- implemented in: 'Succinct Data Structure Library'

## GADDAG
- paper: 'A faster scrabble move generation algorithm' (1994)
- https://en.wikipedia.org/wiki/GADDAG
- uses: 'Directed acyclic graph'
- applications: 'Scrabble'
- implemented in (applications): 'Quackle'

## BD-tree
- original paper: 'The BD-tree - A new N-dimensional data structure with highly efficient dynamic characteristics' (1983)
- survey paper: 'Multidimensional access methods' (1998)
- https://xlinux.nist.gov/dads/HTML/bdtree.html

## MD-tree
- also called: 'Multidimensional tree'
- original paper: 'A Balanced Hierarchical Data Structure for Multidimensional Data with Highly Efficient Dynamic Characteristics' (1993)
- properties: 'height balanced'

## Bounded deformation tree
- also called: 'BD-tree'
- paper: 'BD-tree: output-sensitive collision detection for reduced deformable models'
- http://graphics.cs.cmu.edu/projects/bdtree/

## Chord
- paper: 'Chord: A scalable peer-to-peer lookup service for internet applications (2001)'
- https://en.wikipedia.org/wiki/Chord_(peer-to-peer)
- is a: 'Peer-to-peer distributed hash table', 'Protocol'
- uses: 'Consistent hashing'

## Kademlia
- paper: 'Kademlia: A Peer-to-Peer Information System Based on the XOR Metric (2002)'
- https://en.wikipedia.org/wiki/Kademlia
- is a: 'Peer-to-peer distributed hash table', 'Protocol'
- uses: 'xor metric'

## Koorde
- paper: 'Koorde: A Simple Degree-Optimal Distributed Hash Table (2003)'
- https://en.wikipedia.org/wiki/Koorde
- is a: 'Peer-to-peer distributed hash table', 'Protocol'
- based on: 'Chord'
- uses: 'De Bruijn graph'

## G-Counter
- also called: 'Grow-only Counter'
- https://en.wikipedia.org/wiki/Conflict-free_replicated_data_type#G-Counter_(Grow-only_Counter)
- is a: 'Conflict-free replicated data type', 'Convergent replicated data type'

## PN-Counter
- also called: 'Positive-Negative Counter'
- https://en.wikipedia.org/wiki/Conflict-free_replicated_data_type#PN-Counter_(Positive-Negative_Counter)
- based on: 'G-Counter'
- is a: 'Conflict-free replicated data type'

## G-Set
- also called: 'Grow-only Set'
- is a: 'Conflict-free replicated data type'

## 2P-Set
- also called: 'Two-Phase Set'
- is a: 'Conflict-free replicated data type'

## LWW-Element-Set
- also called: 'Last-Write-Wins-Element-Set'
- is a: 'Conflict-free replicated data type'
- implemented in: 'soundcloud/roshi'

## OR-Set
- also called: 'Observed-Removed Set'
- is a: 'Conflict-free replicated data type'

# Index data structures

## Inverted index
- also called: 'Inverted file'
- https://en.wikipedia.org/wiki/Inverted_index
- https://xlinux.nist.gov/dads/HTML/invertedIndex.html
- used by: 'ElasticSearch', 'Lucene'
- maps content/text to locations/documents
- Search engine indexing
- cf. Back-of-the-book index, Concordance
- applications: 'full-text search', 'sequence assembly'
- consists of: 'dictionary', 'postings list'

## n-gram index
- applications: 'indexed regex queries'
- common variant: '3-gram index'
- used by: 'Google Code Search'

## Sparse index
- https://en.wikipedia.org/wiki/Database_index#Sparse_index

## FM-index
- https://en.wikipedia.org/wiki/FM-index
- applications: 'Bioinformatics'
- is a: 'Substring index'
