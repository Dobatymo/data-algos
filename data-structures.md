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
- implemented in: 'libstdc++ rope'

## Gap buffer
- https://en.wikipedia.org/wiki/Gap_buffer
- applications: 'Text editor'

## Piece table
- https://en.wikipedia.org/wiki/Piece_table
- applications: 'Text editor'
- similar: 'Gap buffer'
- implemented by: 'Visual Studio Code'

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
- https://en.wikipedia.org/wiki/Zipper_(data_structure)
- generalization of: 'Gap buffer'
- more a general technique than a data structure

## Fibonacci heap
- paper: 'Fibonacci heaps and their uses in improved network optimization algorithms (1987)'
- https://en.wikipedia.org/wiki/Fibonacci_heap
- book: 'Introduction to Algorithms'
- implements: 'Priority queue'

## Pairing heap
- https://en.wikipedia.org/wiki/Pairing_heap
- implements: 'Priority queue'

## Binary heap
- https://en.wikipedia.org/wiki/Binary_heap
- implements: 'Priority queue'
- variant implemented in: 'Python heapq', 'C++ std::make_heap, push_heap and pop_heap'

## Binary decision diagram
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

## WPL tree
- paper: 'Weighting Without Waiting: the Weighted Path Length Tree (1991)'

## PQ tree
- paper: 'Testing for the consecutive ones property, interval graphs, and graph planarity using PQ-tree algorithms (1976)'
- https://en.wikipedia.org/wiki/PQ_tree

## Radix tree
- also called: 'Radix trie'
- https://en.wikipedia.org/wiki/Radix_tree
- is a: 'Binary trie'

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
- https://en.wikipedia.org/wiki/AVL_tree
- https://xlinux.nist.gov/dads/HTML/avltree.html
- is a: 'binary tree'
- properties: 'height-balanced'
- stricter balanced and thus better for lookup (compared to 'Red-black tree')

## treap
- https://en.wikipedia.org/wiki/Treap
- randomly ordered binary tree
- log(N) lookup even for insertion of items in non-random order
- heap like feature
- used to implement dictionary in LEDA

## BK-tree
- https://en.wikipedia.org/wiki/BK-tree
- is a: 'Space-partitioning tree', 'Metric tree'
- applications: approximate string matching

## Splay tree
- https://en.wikipedia.org/wiki/Splay_tree
- is a: 'Binary search tree'
- properties: 'self-optimizing'
- applications: 'Caching', 'Garbage collection'
- disadvantages: even concurrent reads require synchronization
- unsolved problem: 'Do splay trees perform as well as any other binary search tree algorithm?'

## k-d tree
- https://en.wikipedia.org/wiki/K-d_tree
- is a: 'Space-partitioning tree'
- applications: range searching, nearest neighbor search, kernel density estimation
- for high dimensions should be: N >> 2^k, where N is the number of nodes and k is the number of dimensions
- solves 'Recursive partitioning', 'Klee's measure problem', 'Guillotine problem'
- implemented in: 'Python scipy.spatial.KDTree, sklearn.neighbors.KDTree, Bio.PDB.kdtrees.KDTree'

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
- book: 'Introduction to Algorithms'
- Multiway tree
- implement ordered maps with integer keys
- implement priority queues
- see 'Integer sorting'

## Skip list
- paper: 'Concurrent Maintenance of Skip Lists (1998)'
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
- https://en.wikipedia.org/wiki/B-tree
- book: 'Introduction to Algorithms'
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
- uses: 'nearest neighbor search'
- is a: 'Space-partitioning tree', 'Metric tree'

## Vantage-point tree
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
- https://en.wikipedia.org/wiki/Winged_edge
- applications: 'Computer graphics'

## Adjacency list
- https://en.wikipedia.org/wiki/Adjacency_list
- implements: 'Graph'
- implemented in: 'boost::adjacency_list'

## Incidence matrix
- https://en.wikipedia.org/wiki/Incidence_matrix
- implements: 'Graph'

## R-tree
- https://en.wikipedia.org/wiki/R-tree
- applications: 'Spatial index', 'Range searching', 'Nearest neighbor search'

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
- https://en.wikipedia.org/wiki/HAT-trie
- implemented in: 'Python pytries/hat-trie'
- implements: 'Ordered map'
- variant of: 'Radix tree'
- properties: 'cache friendly'

## Double-Array Trie
- also called: 'DATrie'
- paper: 'An Efficient Digital Search Algorithm by Using a Double-Array Structure'
- implemented in: 'pytries/datrie', 'libdatrie'

## Ternary search tree
- https://en.wikipedia.org/wiki/Ternary_search_tree
- type of: 'Trie'
- applications: 'Nearest neighbor search'
- implements: 'Map'

## Corner Stitching
- paper: 'Corner Stitching: a Data Structuring Technique for VLSI Layout Tools (1982)'
- applications: 'Very Large Scale Integration'

## Difference list
- https://en.wikipedia.org/wiki/Difference_list

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
- https://en.wikipedia.org/wiki/Bloom_filter
- paper: 'Space/time trade-offs in hash coding with allowable errors (1970)'
- properties: 'probabilistic'
- implements: 'Set'
- applications: 'caching strategies', 'database query optimization', 'rate-limiting', 'data synchronization', 'chemical structure searching'

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

## UB-tree
- https://en.wikipedia.org/wiki/UB-tree
- is a: 'Self-balancing search tree'
- based on: 'B+ tree'

## Log-structured merge-tree
- also called: 'LSM tree'
- https://en.wikipedia.org/wiki/Log-structured_merge-tree
- used by: 'Apache Cassandra', 'Apache HBase', 'Bigtable', 'RocksDB'
- applications: 'Transaction log'

## Wavelet tree
- paper: 'High-order entropy-compressed text indexes (2003)'
- https://en.wikipedia.org/wiki/Wavelet_Tree
- properties: 'succinct'
- implemented in: 'Succinct Data Structure Library'

## GADDAG
- https://en.wikipedia.org/wiki/GADDAG
- uses: 'Directed acyclic graph'
- applications: 'Scrabble'

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
- https://en.wikipedia.org/wiki/Inverted_index
- used by: 'ElasticSearch', 'Lucene'
- maps content/text to locations/documents
- Search engine indexing
- cf. Back-of-the-book index, Concordance
- applications: full-text search, sequence assembly

## Sparse index
- https://en.wikipedia.org/wiki/Database_index#Sparse_index

## FM-index
- https://en.wikipedia.org/wiki/FM-index
- applications: 'Bioinformatics'
- is a: 'Substring index'
