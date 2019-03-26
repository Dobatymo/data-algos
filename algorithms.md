# Algorithms

## Breadth-first search
- https://en.wikipedia.org/wiki/Breadth-first_search
- input: 'Graph'
- implemented in: 'boost::graph::breadth_first_search'

## Depth-first search
- https://en.wikipedia.org/wiki/Depth-first_search
- input: 'Graph'
- implemented in: 'boost::graph::depth_first_search'

## k-d tree construction algorithm using sliding midpoint rule
- example paper: Maneewongvatana and Mount 1999
- constructs: 'k-d tree'
- implemented in: 'scipy.spatial.KDTree'
- input: 'List of k-dimensional points'
- output: 'k-d tree'

## Median of medians
- also called: 'Blum-Floyd-Pratt-Rivest-Tarjan partition algorithm', 'BFPRT'
- paper: 'Time bounds for selection (1973)'
- https://en.wikipedia.org/wiki/Median_of_medians
- selection algorithm
- input: 'random access collection'

## Introselect
- paper: 'Introspective Sorting and Selection Algorithms'
- https://en.wikipedia.org/wiki/Introselect
- implemented in C++ std::nth_element
- is a: 'Selection algorithm'
- input: 'random access collection'

## Floyd–Rivest algorithm
- paper: 'Algorithm 489: the algorithm SELECT—for finding the ith smallest of n elements [M1] (1975)'
- https://en.wikipedia.org/wiki/Floyd%E2%80%93Rivest_algorithm
- is a: 'Selection algorithm', 'Divide and conquer algorithm'
- input: 'random access collection'

## Quickselect
- also called: 'Hoare's selection algorithm'
- paper: 'Algorithm 65: find (1961)'
- https://en.wikipedia.org/wiki/Quickselect
- is a: 'Selection algorithm'
- input: 'random access collection'
- properties: 'parallelizable'

## Dijkstra's algorithm
- paper: 'A note on two problems in connexion with graphs (1959)'
- https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
- http://mathworld.wolfram.com/DijkstrasAlgorithm.html
- uses method: 'Dynamic programming'
- solves 'Shortest path problem' for non-negative weights in directed/undirected graphs in O(v^2) where v is the number of vertices
- variant implementation with 'Fibonacci heap' runs in O(e * v*log v) where e and v are the number of edges and vertices resp.
- implemented in: 'Python scipy.sparse.csgraph.shortest_path(method="D")', 'boost::graph::dijkstra_shortest_paths'
- Fibonacci implementation is the asymptotically the fastest known single-source shortest-path algorithm for arbitrary directed graphs with unbounded non-negative weights.
- input: 'Directed graph with non-negative weights'

## Bellman–Ford algorithm
- paper: 'Structure in communication nets (1955)'
- paper: 'On a routing problem (1958)'
- https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm
- solves variant of the 'Shortest path problem' for real-valued edge weights in directed graph in O(v*e) where v and e are the number of vertices and edges respectively.
- negative cycles are detected
- implemented in: 'Python scipy.sparse.csgraph.shortest_path(method="BF")', 'boost:graph::bellman_ford_shortest_paths'
- input: 'Weighted directed graph'

## Johnson's algorithm
- paper: 'Efficient Algorithms for Shortest Paths in Sparse Networks (1977)'
- https://en.wikipedia.org/wiki/Johnson%27s_algorithm
- solves 'All-pairs shortest paths problem' for real-valued weights in a directed graph in O(v^2 log v + v*e) where v and e are the number of vertices and edges
- implemented in: 'Python scipy.sparse.csgraph.shortest_path(method='J')', 'C++ boost::graph::johnson_all_pairs_shortest_paths'
- combination of 'Bellman–Ford' and 'Dijkstra's algorithm'
- is faster than 'Floyd–Warshall algorithm' for sparse graphs
- input: 'weighted directed graph without negative cycles'

## Floyd–Warshall algorithm
- paper: 'Algorithm 97: Shortest path (1962)'
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
- paper: 'Disjoint paths in a network (1974)'
- https://en.wikipedia.org/wiki/Suurballe%27s_algorithm
- implemented in: 'Python nildo/suurballe'
- uses: 'Dijkstra's algorithm'
- solves: 'Shortest pair of edge disjoint paths'
- input: 'Directed graph with non-negative weights'

## Edge disjoint shortest pair algorithm
- paper: 'Survivable networks: algorithms for diverse routing'
- https://en.wikipedia.org/wiki/Edge_disjoint_shortest_pair_algorithm
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
- also called: 'Dijkstra flow maps'
- paper: 'Collaborative diffusion: programming antiobjects (2006)'
- https://en.wikipedia.org/wiki/Collaborative_diffusion
- applications: 'pathfinding'
- time complexity: constant in the number of agents

## Ukkonen's algorithm
- https://en.wikipedia.org/wiki/Ukkonen%27s_algorithm
- paper: 'On-line construction of suffix trees'
- book: 'Algorithms on Strings, Trees, and Sequences'
- output: 'suffix tree'
- properties: 'online'
- time complexity: O(n), where n is the length of the string
- input: 'List of strings'

## Weiner's linear-time suffix tree algorithm
- book: 'Algorithms on Strings, Trees, and Sequences'
- output: 'suffix tree'
- superseded by: 'Ukkonen's algorithm'

## McCreight's algorithm
- book: 'Algorithms on Strings, Trees, and Sequences'
- output: 'suffix tree'
- superseded by: 'Ukkonen's algorithm'

## A* search algorithm
- paper: 'A Formal Basis for the Heuristic Determination of Minimum Cost Paths (1968)'
- https://en.wikipedia.org/wiki/A*_search_algorithm
- generalization of 'Dijkstra's algorithm'
- heuristic search
- informed search algorithm (best-first search)
- usually implemented using: 'Priority queue'
- applications: 'Pathfinding', 'Parsing using stochastic grammars in NLP'
- uses method: 'Dynamic programming'
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

## Cumulative sum
- also called: 'Prefix sum', 'Accumulate', 'Inclusive scan'
- https://en.wikipedia.org/wiki/Prefix_sum
- http://mathworld.wolfram.com/CumulativeSum.html
- implemented in: 'Python numpy.cumsum', 'Mathematica Accumulate'
- properties: 'parallelizable'

## Wyllie's algorithm
- thesis: 'The Complexity of Parallel Computations (1979)'
- solves: 'List ranking'
- related: 'Cumulative sum'
- is a: 'Parallel algorithm'

## Funnelsort
- paper: 'Cache-oblivious algorithms (1999)'
- https://en.wikipedia.org/wiki/Funnelsort
- is a: 'cache-oblivious algorithm', 'external memory algorithm', 'Comparison-based sorting algorithm'
- input: 'Collection'

## Quicksort
- paper: 'Algorithm 64: Quicksort (1961)'
- https://en.wikipedia.org/wiki/Quicksort
- http://mathworld.wolfram.com/Quicksort.html
- book: 'Introduction to Algorithms'
- is a: 'Sorting algorithm', 'Unstable sorting algorithm', 'In-place algorithm', 'Divide and conquer algorithm', 'Comparison-based sorting algorithm'
- time complexity (best): O(n log n)
- time complexity (average): O(n log n)
- time complexity (worst): O(n^2)
- space complexity: O(log n) auxiliary
- input: 'Random access collection'
- properties: easily parallelizable

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
- also called: 'Stupid sort'
- paper: 'Stupid Sort: A new sorting algorithm'
- https://en.wikipedia.org/wiki/Gnome_sort
- time complexity (average, worst): O(n^2)
- time complexity (best): O(n)
- requires no nested loops

## Splaysort
- paper: 'Splaysort: Fast, Versatile, Practical (1996)'
- https://en.wikipedia.org/wiki/Splaysort
- based on: 'Splay tree'
- properties: 'comparison based'

## Cocktail shaker sort
- paper: 'Sorting by Exchanging (1973)'
- also called: 'bidirectional bubble sort'
- https://en.wikipedia.org/wiki/Cocktail_shaker_sort
- input: 'Bidirectional Collection'
- properties: 'stable', 'in-place'
- variant of: 'Bubble sort'
- time complexity (average, worst): O(n^2)
- time complexity (best): O(n)

## Merge sort
- https://en.wikipedia.org/wiki/Merge_sort
- is a: 'Sorting algorithm', 'Stable sorting algorithm' (usually), 'Divide and conquer algorithm', 'Comparison-based sorting algorithm'
- implemented in: 'C++ std::stable_sort (usually)'
- good for sequential access, can work on 'singly linked lists', external sorting
- properties: easily parallelizable
- input: 'Collection'

## Counting sort
- thesis: 'Information sorting in the application of electronic digital computers to business operations (1954)' by 'H. H. Seward'
- https://en.wikipedia.org/wiki/Counting_sort
- is a: 'Integer sorting algorithm'
- properties: 'parallelizable'

## Heapsort
- https://en.wikipedia.org/wiki/Heapsort
- http://mathworld.wolfram.com/Heapsort.html
- book: 'Introduction to Algorithms'
- is a: 'Sorting algorithm', 'In-place algorithm', 'Unstable sorting algorithm', 'Comparison-based sorting algorithm'
- solves: 'Partial sorting'
- time complexity (average, best, worst): O(n log n)
- space complexity: O(1)
- uses: 'max heap'
- not easily parallelizable
- variant works on 'doubly linked lists'
- input: 'Random access collection'

## Ultimate heapsort
- paper: 'The Ultimate Heapsort (1998)'
- variant of: 'Heapsort'
- comparisons: n log_2 n + O(1)

## Timsort
- https://en.wikipedia.org/wiki/Timsort
- is a: 'Sorting algorithm', 'Stable sorting algorithm', 'Comparison-based sorting algorithm'
- implemented in: 'Python sorted', 'Android Java'
- input: 'Random access collection'

## Introsort
- paper: 'Introspective Sorting and Selection Algorithms'
- https://en.wikipedia.org/wiki/Introsort
- is a: 'Sorting algorithm', 'Unstable sorting algorithm', 'Comparison-based sorting algorithm'
- implemented in: 'C++ STL std::sort (usually)', '.net sort'
- input: 'Random access collection'

## Selection sort
- https://en.wikipedia.org/wiki/Selection_sort
- http://mathworld.wolfram.com/SelectionSort.html
- is a: 'Sorting algorithm', 'In-place algorithm', 'Unstable sorting algorithm', 'Comparison-based sorting algorithm'
- input: 'Random access collection'
- properties: 'parallelizable'

## Insertion sort
- https://en.wikipedia.org/wiki/Insertion_sort
- properties: 'stable', 'in-place'
- input: 'List' (for not-in-place)
- input: 'bidirectional list' (for in-place)
- properties: 'parallelizable'

## Bitonic sorter
- https://en.wikipedia.org/wiki/Bitonic_sorter
- is a: 'Sorting algorithm', 'Parallel algorithm'

## Batcher odd–even mergesort
- https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
- is a: 'Sorting algorithm', 'Parallel algorithm'

## Pairwise sorting network
- paper: 'The pairwise sorting network (1992)'
- https://en.wikipedia.org/wiki/Pairwise_sorting_network
- is a: 'Sorting algorithm', 'Parallel algorithm'

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

## Sleep sort
- https://web.archive.org/web/20151231221001/http://bl0ckeduser.github.io/sleepsort/sleep_sort_trimmed.html
- https://www.cs.princeton.edu/courses/archive/fall13/cos226/lectures/52Tries.pdf
- time complexity: O(n) (for all theoretical purposes...)
- applications: 'Integer sorting'

## Fisher–Yates shuffle
- https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
- is a: 'Shuffling algorithm', 'In-place algorithm'
- unbiased
- input: 'Random access collection'

## Reservoir sampling
- https://en.wikipedia.org/wiki/Reservoir_sampling
- https://xlinux.nist.gov/dads/HTML/reservoirSampling.html
- family of 'randomized algorithms'
- version of: 'Fisher–Yates shuffle'
- properties: 'online'

## Cache-oblivious distribution sort
- https://en.wikipedia.org/wiki/Cache-oblivious_distribution_sort
- comparison-based sorting algorithm
- cache-oblivious algorithm

## Naive Method for SimRank by Jeh and Widom
- paper: 'SimRank: a measure of structural-context similarity (2002)'
- calculate: 'SimRank'

## De Casteljau's algorithm
- https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
- paper: 'Système d'aide à la définition et à l'usinage de surfaces de carosserie (1971)'
- evaluate polynomials in Bernstein form or Bézier curves
- properties: 'numerically stable'
- applications: 'Computer aided geometric design'

## Clenshaw algorithm
- also called: 'Clenshaw summation'
- https://en.wikipedia.org/wiki/Clenshaw_algorithm
- evaluate polynomials in Chebyshev form

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
- book: 'Introduction to Algorithms'
- single *string searching*
- implemented in: 'grep'
- better for small alphabets like DNA than: 'Boyer–Moore string-search algorithm'

## Rabin–Karp algorithm
- also called: 'Karp–Rabin algorithm'
- https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm
- book: 'Introduction to Algorithms'
- is a: 'String-searching algorithm'
- single/multiple *string searching*
- time complexity (worst): O(m+n)
- space complexity: constant
- applications: 'Plagiarism detection'

## Bitap algorithm
- also called: 'shift-or algorithm', 'shift-and algorithm', 'Baeza-Yates–Gonnet algorithm'
- https://en.wikipedia.org/wiki/Bitap_algorithm
- solves: 'Approximate string matching'

## Myers' Diff Algorithm
- paper: 'An O(ND) difference algorithm and its variations (1986)'
- solves: 'Shortest Edit Script'
- input: two strings
- output: 'Shortest Edit Script'
- implemented by: 'diff', 'git' (Linear space variant)
- variants: 'Linear space'

## Patience Diff method
- https://bramcohen.livejournal.com/73318.html
- requires: 'diff algorithm'

## Non-negative matrix factorization
- also called: 'NMF', 'NNMF'
- https://en.wikipedia.org/wiki/Non-negative_matrix_factorization
- applications: 'Collaborative filtering', 'Dimensionality reduction', 'Text mining'

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
- input: 'List of non-negative reals'
- output: 'Optimal binary search tree' (special case)

## Knuth's optimal binary search tree algorithm
- https://en.wikipedia.org/wiki/Optimal_binary_search_tree#Knuth%27s_dynamic_programming_algorithm
- paper: 'Optimum binary search trees'
- uses method: 'dynamic programming'
- output: 'Optimal binary search tree'
- time comlexity: O(n^2)

## Mehlhorn's nearly optimal binary search tree algorithm
- paper 'Nearly optimal binary search trees'
- output approximatly: 'Optimal binary search tree'
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
- solves approximately: 'Travelling salesman problem' (for metric distances)

## Push–relabel maximum flow algorithm
- https://en.wikipedia.org/wiki/Push–relabel_maximum_flow_algorithm
- solves: 'Maximum flow problem', 'Minimum-cost flow problem'
- implemented in: 'boost::graph::push_relabel_max_flow'

## Kruskal's algorithm
- https://en.wikipedia.org/wiki/Kruskal%27s_algorithm
- http://mathworld.wolfram.com/KruskalsAlgorithm.html
- output: 'Minimum spanning tree'
- properties: 'greedy'
- implemented in: 'C++ boost::graph::kruskal_minimum_spanning_tree'
- book: 'Introduction to Algorithms'

## Prim's algorithm
- https://en.wikipedia.org/wiki/Prim%27s_algorithm
- output: 'Minimum spanning tree'
- properties: 'greedy'
- implemented in 'C++ boost::graph::prim_minimum_spanning_tree'
- time complexity depends on used data structures
- book: 'Introduction to Algorithms'

## Hierholzer's algorithm
- https://en.wikipedia.org/wiki/Eulerian_path#Hierholzer's_algorithm
- https://www.geeksforgeeks.org/hierholzers-algorithm-directed-graph/
- input: 'Finite graph'
- output: 'Eulerian path'
- more efficient than: 'Fleury's algorithm'

## Kahn's algorithm
- https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
- applications: 'topological sorting'

## Depth-first search
- https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
- applications: 'topological sorting', 'Strongly connected component'

## HyperLogLog++
- https://en.wikipedia.org/wiki/HyperLogLog
- solves approximately: 'Count-distinct problem'
- implemented by: 'Lucence'

## Hunt–McIlroy algorithm
- https://en.wikipedia.org/wiki/Hunt%E2%80%93McIlroy_algorithm
- solves: 'Longest common subsequence problem'
- implemented by: 'diff'
- input: 'set of sequences'
- output: 'Longest common subsequence'

## Burrows–Wheeler transform
- is a: 'transform'
- properties: 'reversible'
- applications: 'text compression'
- implemented using: 'Suffix array'
- variant: 'Bijective variant'

## Tarjan's strongly connected components algorithm
- https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
- input: 'Directed graph'
- output: 'Strongly connected component'

## Awerbuch-Shiloach algorithm for finding the connected components
- paper: 'New Connectivity and MSF Algorithms for Shuffle-Exchange Network and PRAM (1987)'
- is a: 'Parallel algorithm'

## Tarjan's off-line lowest common ancestors algorithm
- https://en.wikipedia.org/wiki/Tarjan%27s_off-line_lowest_common_ancestors_algorithm
- input: 'pairs of nodes in a tree'
- output: 'Lowest common ancestor'

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

## Chazelle's algorithm for the minimum spanning tree
- paper: 'A minimum spanning tree algorithm with inverse-Ackermann type complexity (2000)'
- output: 'Minimum spanning tree'
- uses: 'soft heaps'

## Chazelle's polygon triangulation algorithm
- paper: 'Triangulating a simple polygon in linear time (1991)'
- very difficult to implement in code

## Risch semi-algorithm
- paper: 'On the Integration of Elementary Functions which are built up using Algebraic Operations (1968)'
- https://en.wikipedia.org/wiki/Risch_algorithm
- http://mathworld.wolfram.com/RischAlgorithm.html
- solves: 'Indefinite integration'
- applications: 'Symbolic computation', 'Computer algebra'
- implemented in: 'Axiom'
- very difficult to implement in code

## Romberg's method
- paper: 'Vereinfachte numerische Integration (1955)'
- https://en.wikipedia.org/wiki/Romberg%27s_method
- applications: 'Numerical integration'
- implemented in: 'Python scipy.integrate.romberg'
- input: 'Function and integration boundaries'
- output: 'Integrated value'
- domain: 'Analysis'

## Brelaz's heuristic algorithm
- paper: 'New methods to color the vertices of a graph (1979)'
- http://mathworld.wolfram.com/BrelazsHeuristicAlgorithm.html
- solves approximately: 'Graph coloring problem'
- implemented in: 'Mathematica BrelazColoring'

## Local-Ratio algorithm
- paper: 'A Local-Ratio Theorem for Approximating the Weighted Vertex Cover Problem'
- implemented in: 'networkx.algorithms.approximation.vertex_cover.min_weighted_vertex_cover'
- solves approximately: 'Vertex cover problem'
- domain: 'Graph theory'

## Hopcroft–Karp algorithm
- paper: 'An $n^{5/2}$ Algorithm for Maximum Matchings in Bipartite Graphs (1973)'
- https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm
- https://brilliant.org/wiki/hopcroft-karp/
- input: 'Bipartite graph'
- output: 'Maximum matching'
- time complexity: O(E sqrt(V)) for E edges and V vertices
- best for sparse, for dense, there are more recent improvements like 'Alt et al. (1991)'
- domain: 'Graph theory'

## Hungarian Maximum Matching algorithm
- also called: 'Kuhn-Munkres algorithm'
- paper: 'The Hungarian method for the assignment problem (1955)'
- https://en.wikipedia.org/wiki/Hungarian_algorithm
- http://mathworld.wolfram.com/HungarianMaximumMatchingAlgorithm.html
- https://brilliant.org/wiki/hungarian-matching/
- input: 'bipartite graph'
- output: 'maximum-weight matching'
- time complexity: O(V^3) for V vertices
- domain: 'Graph theory', 'Combinatorial optimization'

## Blossom algorithm
- paper: 'Paths, trees, and flowers (1965)'
- also called: 'Edmonds' matching algorithm'
- https://en.wikipedia.org/wiki/Blossom_algorithm
- http://mathworld.wolfram.com/BlossomAlgorithm.html
- https://brilliant.org/wiki/blossom-algorithm/
- domain: 'Graph theory'
- input: 'Graph'
- output: 'Maximum matching'
- time complexity: O(E V^2) for E edges and V vertices

## Xiao and Nagamochi's algorithm for the maximum independent set
- paper: 'Exact algorithms for maximum independent set (2017)'
- solves: 'Maximum independent set problem'
- time complexity: O(1.1996^n)
- space complexity: polynomial
- superseeds: 'Robson (1986)'

## Boppana and Halldórsson's approximation algorithm for the maximum independent set
- paper: 'Approximating maximum independent sets by excluding subgraphs (1992)'
- solves approximate: 'Maximum independent set problem'
- implemented in: 'Python networkx.algorithms.approximation.independent_set.maximum_independent_set'
- time complexity: O(n / (log b)^2)

## DPLL algorithm
- paper: 'A machine program for theorem-proving (1962)'
- https://en.wikipedia.org/wiki/DPLL_algorithm
- also called: 'Davis–Putnam–Logemann–Loveland algorithm'
- solves: 'Boolean satisfiability problem'
- applications: 'Automated theorem proving'

## Canny's Roadmap algorithm
- http://planning.cs.uiuc.edu/node298.html
- very difficult to implement in code
- domain: 'Computational algebraic geometry'
- solves: 'Motion planning'

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
- implemented in: 'Python librosa.sequence.viterbi'

## List Viterbi algorithm
- also called: 'LVA'

## BCJR algorithm
- https://en.wikipedia.org/wiki/BCJR_algorithm
- paper: 'Optimal Decoding of Linear Codes for minimizing symbol error rate (1974)'
- decodes: 'Error correction code'
- implemented in: 'C++ Susa'
- related theory: 'maximum a posteriori'

## Seam carving
- https://en.wikipedia.org/wiki/Seam_carving
- uses method: 'Dynamic programming'
- applications: 'Image resizing', 'Image processing'
- domain: 'computer graphics'
- implemented in: 'Adobe Photoshop', 'GIMP'

## Dynamic time warping
- https://en.wikipedia.org/wiki/Dynamic_time_warping
- paper: 'Dynamic programming algorithm optimization for spoken word recognition (1978)'
- applications: 'Time series analysis', 'Speech recognition', 'Speaker recognition', 'Signature recognition', 'Shape matching', 'Correlation power analysis'
- implemented in: 'Python pydtw', 'Python librosa.core.dtw'
- possibly superseded by: 'Connectionist temporal classification'
- input: two sequences
- time complexity: O(n m), where n and m are the lengths of the input sequences
- space complexity: O(n m), where n and m are the lengths of the input sequences

## Synchronized overlap-add method
- also called: 'Time-domain harmonic scaling', 'SOLA'
- paper: 'High quality time-scale modification for speech (1985)'
- applications: 'Audio pitch scaling'

## Phase vocoder
- https://en.wikipedia.org/wiki/Phase_vocoder
- paper: 'Phase vocoder (1966)'
- applications: 'Audio time stretching and pitch scaling'
- uses: 'Short-time Fourier transform'

## De Boor's algorithm
- https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
- domain: 'numerical analysis'
- spline curves in B-spline
- properties: 'numerically stable'
- implemented in: 'Python scipy.interpolate.BSpline'
- generalization of: 'De Casteljau's algorithm'

## Lucas–Kanade method
- https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method
- applications: 'Optical flow estimation', 'aperture problem'
- local, sparse
- implemented in: 'opencv::calcOpticalFlowPyrLK', 'opencv::CalcOpticalFlowLK' (obsolete)
- uses: 'Structure tensor'

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
- properties: 'Embarrassingly parallel'

## Lempel–Ziv–Welch
- https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch
- applications: 'Lossless compression'
- was patented

## General number field sieve
- https://en.wikipedia.org/wiki/General_number_field_sieve
- applications: 'Integer factorization'

## Shor's algorithm
- paper: 'Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer (1994)'
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
- input (variant 1): 'List of bools'
- output (variant 1): 'shortest linear feedback shift register'
- input (variant 2): 'arbitrary field'
- output (variant 2): 'minimal polynomial of a linearly recurrent sequence'

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
- applications: 'Sequence alignment' (global), 'Computer stereo vision'
- time complexity: O(m n)
- uses method: 'Dynamic programming'
- domain: 'bioinformatics'
- implemented in: 'EMBOSS', 'Python Bio.pairwise2.align.globalxx'
- input: 'two random access collections'
- output: 'Optimal global alignment'

## Smith–Waterman algorithm
- https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
- applications: 'Sequence alignment' (local)
- time complexity: O(m n)
- uses method: 'Dynamic programming'
- domain: 'bioinformatics'
- implemented in: 'EMBOSS', 'Python Bio.pairwise2.align.localxx'
- input: 'two random access collections'
- output: 'Optimal local alignment'

## Hirschberg's algorithm
- https://en.wikipedia.org/wiki/Hirschberg%27s_algorithm
- applications: 'Sequence alignment' (global)
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

## Stoer and Wagner's minimum cut algorithm
- paper: 'A Simple Min-Cut Algorithm (1994)'
- input: 'Connected graph'
- output: 'Minimum cut'
- implemented in: 'boost::graph::stoer_wagner_min_cut'
- domain: 'Graph theory'

## Ford–Fulkerson method
- https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm
- https://brilliant.org/wiki/ford-fulkerson-algorithm/
- properties: 'greedy', 'incomplete'
- solves: 'Maximum flow problem'
- implemented by: 'Edmonds–Karp algorithm'
- input: 'Flow network'
- output: 'Maximum flow'

## Edmonds–Karp algorithm
- https://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm
- https://brilliant.org/wiki/edmonds-karp-algorithm/
- implements: 'Ford–Fulkerson method'
- implemented in: 'Python networkx.algorithms.flow.edmonds_karp', 'boost::graph::edmonds_karp_max_flow'
- time complexity: O(v e^2) or O(v^2 e) where v is the number of vertices and e the number of edges
- solves: 'Maximum flow problem'
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

## Work stealing algorithm
- https://en.wikipedia.org/wiki/Work_stealing
- applications: 'Scheduling'
- implemented in: 'Cilk'

## k-nearest neighbors algorithm
- also called: 'k-NN', 'KNN'
- https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
- is a: 'Machine learning algorithm', 'Classification algorithm', 'Regression algorithm'
- properties: 'non-parametric', 'instance-based learning', 'lazy learning'
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
- paper: 'An Algorithm for Vector Quantizer Design' (1980)
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
- also called: 'MST-algorithm'
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
- implemented in: 'Python scipy.cluster.hierarchy.linkage, Bio.Phylo.TreeConstruction.DistanceTreeConstructor'
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

## Brooks–Iyengar algorithm
- https://en.wikipedia.org/wiki/Brooks%E2%80%93Iyengar_algorithm
- is a: 'distributed algorithm'
- applications: 'Distributed sensing', 'Fault tolerance'

## Suzuki–Kasami algorithm
- https://en.wikipedia.org/wiki/Suzuki%E2%80%93Kasami_algorithm
- paper: 'A distributed mutual exclusion algorithm (1985)'
- solves: 'mutual exclusion in distributed systems'
- modificaton of: 'Ricart–Agrawala algorithm'

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
- output: 'Simple graph'

## QR algorithm
- https://en.wikipedia.org/wiki/QR_algorithm
- uses: 'QR decomposition'
- properties: 'numerically stable'
- modern implicit variant called: 'Francis algorithm'
- supersedes: 'LR algorithm' because of better numerical stability
- input: 'Real matrix'
- output: 'Eigenvalues and eigenvectors'

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

## Force-directed graph drawing
- https://en.wikipedia.org/wiki/Force-directed_graph_drawing
- class of algorithms
- used by: 'Graphviz'
- applications: 'Graph drawing'

## Fruchterman–Reingold algorithm
- paper: 'Graph Drawing by Force-Directed Placement (1991)'
- type: 'Force-directed graph drawing'
- implemented in: 'boost::graph::fruchterman_reingold_force_directed_layout'
- input: 'Unweighted, undirected graph'
- time complexity (worst): O(v^2 + e) where v is the number of vertices and e the number of edges

## Kamada–Kawai algorithm
- paper: 'An algorithm for drawing general undirected graphs (1989)'
- type: 'Force-directed graph drawing'
- implemented in: 'boost::graph::kamada_kawai_spring_layout'
- input: 'Connected, undirected graph'

## Fast multipole method
- https://en.wikipedia.org/wiki/Fast_multipole_method
- domain: 'Computational electromagnetism'

## Fast Fourier transform method
- https://en.wikipedia.org/wiki/Fast_Fourier_transform
- output: 'Discrete Fourier transform'
- implemented in: 'FFTW', 'FFTPACK'

## Cooley–Tukey FFT algorithm
- paper: 'An Algorithm for the Machine Calculation of Complex Fourier Series (1965)'
- https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
- variation of: 'Fast Fourier transform'
- implemented in: 'Python numpy.fft, scipy.fftpack.fft'

## Winograd FFT algorithm
- also called: 'Multiplicative Fourier transform algorithm'
- paper: 'On computing the Discrete Fourier Transform (1976)'
- https://www.encyclopediaofmath.org/index.php/Winograd_Fourier_transform_algorithm
- minimal multiplications at the cost of more additions

## Rader's FFT algorithm
- paper: 'Discrete Fourier transforms when the number of data samples is prime (1968)'
- https://en.wikipedia.org/wiki/Rader%27s_FFT_algorithm

## Fast wavelet transform
- also called: 'FWT'
- paper: 'A theory for multiresolution signal decomposition: the wavelet representation (1989)'
- https://en.wikipedia.org/wiki/Fast_wavelet_transform
- input: 'waveform'
- output: 'wavelets'

## Kirkpatrick–Seidel algorithm
- paper: 'The Ultimate Planar Convex Hull Algorithm? (1983)'
- https://en.wikipedia.org/wiki/Kirkpatrick%E2%80%93Seidel_algorithm
- input: 'Collection of 2-d points'
- output: 'Convex hull'
- properties: 'output-sensitive'

## Chan's algorithm
- paper: 'Optimal output-sensitive convex hull algorithms in two and three dimensions (1996)'
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
- paper: 'Percolation and cluster distribution. I. Cluster multiple labeling technique and critical concentration algorithm (1976)'
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
- paper: 'A sweepline algorithm for Voronoi diagrams (1986)'
- https://en.wikipedia.org/wiki/Fortune%27s_algorithm
- is a: 'Sweep line algorithm'
- input: 'Collection of points'
- output: 'Voronoi diagram'
- time complexity: O(n log n)
- space complexity: O(n)

## Bentley–Ottmann algorithm
- paper: 'Algorithms for Reporting and Counting Geometric Intersections (1979)'
- https://en.wikipedia.org/wiki/Bentley%E2%80%93Ottmann_algorithm
- is a: 'Sweep line algorithm'
- solves: 'Line segment intersection'
- input: 'set of line segments'
- output: 'crossings'
- based on: 'Shamos–Hoey algorithm'

## Freeman-Shapira's minimum bounding box
- paper: 'Determining the minimum-area encasing rectangle for an arbitrary closed curve (1975)'
- solves: 'Minimum bounding box'
- input: 'convex polygon'
- output: 'minimum-area enclosing rectangle'
- time complexity: O(n)

## Rotating calipers
- https://en.wikipedia.org/wiki/Rotating_calipers
- 'Solving Geometric Problems with the Rotating Calipers'?

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

## Buchberger's algorithm
- https://en.wikipedia.org/wiki/Buchberger%27s_algorithm
- http://mathworld.wolfram.com/BuchbergersAlgorithm.html
- domain: 'Algebraic geometry', 'Commutative algebra'
- input: 'set of generators for a polynomial ideal'
- output: 'Gröbner basis with respect to some monomial order'

## FGLM algorithm
- https://en.wikipedia.org/wiki/FGLM_algorithm
- input: 'Gröbner basis of a zero-dimensional ideal in the ring of polynomials over a field with respect to a monomial order and a second monomial order'
- output: 'Gröbner basis of the ideal with respect to the second ordering'
- applications: 'Computer algebra system'
- domain: 'Computer algebra'

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

## Knuth's Algorithm X
- also called: 'DLX' with implemented using 'Dancing Links'
- solves: 'Exact cover'
- applications: 'Sudoku', 'Tessellation', 'Eight queens puzzle'

## Badouel intersection algorithm
- https://en.wikipedia.org/wiki/Badouel_intersection_algorithm
- book: 'Graphics Gems I'
- domain: 'Computational geometry'
- input: 'ray and a triangle in three dimensions'

## Rete algorithm
- https://en.wikipedia.org/wiki/Rete_algorithm
- is a: 'Pattern matching algorithm'

## Hilltop algorithm
- https://en.wikipedia.org/wiki/Hilltop_algorithm

## Out-of-kilter algorithm
- https://en.wikipedia.org/wiki/Out-of-kilter_algorithm
- solves: 'Minimum-cost flow problem'
- input: 'Flow network'
- output: 'Minimum-cost flow'

## Network simplex algorithm
- https://en.wikipedia.org/wiki/Network_simplex_algorithm
- solves: 'Minimum-cost flow problem'
- specialisation of: 'Simplex algorithm'

## Expectation–maximization algorithm
- https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm
- is a: 'iterative method'
- applications: 'Parameter estimation'

## Inside–outside algorithm
- https://en.wikipedia.org/wiki/Inside%E2%80%93outside_algorithm
- paper: 'Trainable grammars for speech recognition (1971)'
- uses?: 'Expectation–maximization algorithm'
- special case of?: 'Expectation–maximization algorithm'
- generalization of: 'Forward–backward algorithm'
- re-estimating production probabilities in a probabilistic context-free grammar
- applications: 'PCFG'

## Cocke–Younger–Kasami algorithm
- also called: 'CYK', 'CKY'
- https://en.wikipedia.org/wiki/CYK_algorithm
- is a: 'chart parsing algorithm'
- is a: parsing algorithm for context-free grammars
- uses techniques: 'bottom-up parsing', 'dynamic programming'

## Earley parser
- https://en.wikipedia.org/wiki/Earley_parser
- is a: 'chart parsing algorithm'
- uses technique: 'Dynamic programming', 'Top-down parsing'
- is a: algorithm for parsing strings that belong to a given context-free language
- implemented in: 'Python nltk.parse.earleychart.EarleyChartParser'

## Valiant's algorithm
- https://en.wikipedia.org/wiki/CYK_algorithm#Valiant's_algorithm
- variant of: 'Cocke–Younger–Kasami algorithm'

## Shunting-yard algorithm
- https://en.wikipedia.org/wiki/Shunting-yard_algorithm
- is a: method for parsing mathematical expressions specified in infix notation
- generalization: 'Operator-precedence parser'

## Recursive ascent parser
- https://en.wikipedia.org/wiki/Recursive_ascent_parser

## Recursive descent parser
- https://en.wikipedia.org/wiki/Recursive_descent_parser
- uses technique: 'Top-down parsing'
- implemented in: 'Python nltk.parse.recursivedescent.RecursiveDescentParser'

## LALR parser
- also called: 'Look-Ahead LR parser'
- https://en.wikipedia.org/wiki/LALR_parser

## Shift-reduce parser
- https://en.wikipedia.org/wiki/Shift-reduce_parser
- uses technique: 'Bottom-up parsing'
- implemented in: 'Python nltk.parse.shiftreduce.ShiftReduceParser'

## Baum–Welch algorithm
- https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
- uses: 'Forward–backward algorithm', 'Expectation–maximization algorithm'
- input: 'Hidden Markov model'
- output: 'HMM parameters'

## WINEPI
- https://en.wikipedia.org/wiki/WINEPI
- applications: 'Data mining', 'Time series analysis', 'Association rule learning'

## Bach's algorithm
- paper: 'How to generate factored random numbers (1988)'
- https://en.wikipedia.org/wiki/Bach%27s_algorithm
- applications: 'Random number generation'
- input: 'Seed'
- output: 'Random number and its factorization'

## Bareiss algorithm
- https://en.wikipedia.org/wiki/Bareiss_algorithm
- properties: 'uses only integer arithmetic'
- input: 'matrix with integer entries
- output: 'determinant or the echelon form of metrix'
- variant of: 'Gaussian elimination'

## Odlyzko–Schönhage algorithm
- https://en.wikipedia.org/wiki/Odlyzko%E2%80%93Sch%C3%B6nhage_algorithm
- evaluates: Riemann zeta function
- properties: vectorized
- uses: 'Fast Fourier transform'

## Principal component analysis
- also called: PCA, 'Karhunen–Loève transform', KLT, Hotelling transform,
- https://en.wikipedia.org/wiki/Principal_component_analysis
- applications: 'Dimensionality reduction', 'Exploratory data analysis'
- implemented in: 'LAPACK', 'ARPACK', 'Python sklearn.decomposition.PCA, Bio.Cluster.pca'
- can be implemented using: 'Singular Value Decomposition'

## Kernel principal component analysis
- https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
- is a: 'Machine learning algorithm', 'Kernel method'
- solves: 'Nonlinear dimensionality reduction'
- applications: 'novelty detection', 'image de-noising'
- implemented in: 'Python sklearn.decomposition.KernelPCA'

## Spectral Co-Clustering algorithm
- http://www.kemaleren.com/post/spectral-biclustering-part-1/
- paper: 'Co-clustering documents and words using Bipartite Spectral Graph Partitioning (2001)'
- solves: 'Biclustering'
- implemented in: 'sklearn.cluster.bicluster.SpectralCoclustering'
- input: 'Matrix'

## Pantelides algorithm
- https://en.wikipedia.org/wiki/Pantelides_algorithm
- reduces: 'Differential-algebraic system of equations' to lower index
- implemented in: 'Mathematica NDSolve' (as part of)

## Lemke's algorithm
- https://en.wikipedia.org/wiki/Lemke%27s_algorithm
- domain: 'mathematical optimization'
- solves: 'Linear complementarity problem', 'Mixed linear complementarity problem'
- type: 'Basis-exchange'

## Berlekamp–Zassenhaus algorithm
- https://en.wikipedia.org/wiki/Berlekamp%E2%80%93Zassenhaus_algorithm
- http://mathworld.wolfram.com/Berlekamp-ZassenhausAlgorithm.html
- applications: 'Polynomial factorization'

## Blahut–Arimoto algorithm
- https://en.wikipedia.org/wiki/Blahut%E2%80%93Arimoto_algorithm
- domain: 'Coding theory'

## Block Lanczos algorithm
- https://en.wikipedia.org/wiki/Block_Lanczos_algorithm
- paper: 'A Block Lanczos Algorithm for Finding Dependencies over GF(2)'
- input: 'matrix over a finite field'
- output: 'nullspace'
- domain: 'Linear algebra'

## Kabsch algorithm
- https://en.wikipedia.org/wiki/Kabsch_algorithm
- input: 'two paired sets of points'
- output: 'optimal rotation matrix' (according to 'Root-mean-square deviation')

## Cholesky algorithm
- https://en.wikipedia.org/wiki/Cholesky_decomposition#The_Cholesky_algorithm
- input: 'Hermitian, positive-definite matrix'
- output: 'Cholesky decomposition'

## Cholesky–Banachiewicz algorithm
- variation of: 'Cholesky algorithm'

## Cholesky–Crout algorithm
- variation of: 'Cholesky algorithm'

## Knuth's Simpath algorithm
- https://en.wikipedia.org/wiki/Knuth%27s_Simpath_algorithm
- domain: 'Graph theory'
- input: 'Graph'
- output: 'zero-suppressed decision diagram (ZDD) representing all simple paths between two vertices'

## Zeilberger's algorithm
- paper: 'A fast algorithm for proving terminating hypergeometric identities (1990)'
- http://mathworld.wolfram.com/ZeilbergersAlgorithm.html
- https://archive.lib.msu.edu/crcmath/math/math/z/z020.htm
- input: 'Terminating Hypergeometric Identities of a certain form'
- output: 'Polynomial recurrence'

## Gosper's algorithm
- https://en.wikipedia.org/wiki/Gosper%27s_algorithm
- http://mathworld.wolfram.com/GospersAlgorithm.html
- input: 'hypergeometric terms'
- output: 'sums that are themselves hypergeometric terms'

## PSOS algorithm
- also called: 'Partial sum of squares'
- http://mathworld.wolfram.com/PSOSAlgorithm.html
- is a: 'Integer relation algorithm'

## Ferguson-Forcade algorithm
- http://mathworld.wolfram.com/Ferguson-ForcadeAlgorithm.html
- is a: 'Integer relation algorithm'
- generalization of: 'Euclidean algorithm'

## HJLS algorithm
- http://mathworld.wolfram.com/HJLSAlgorithm.html
- is a: 'Integer relation algorithm'
- uses: 'Gram-Schmidt Orthonormalization'
- properties: 'nummerical unstable'

## Lenstra–Lenstra–Lovász lattice basis reduction algorithm
- also called: 'LLL algorithm'
- https://en.wikipedia.org/wiki/Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm
- http://mathworld.wolfram.com/LLLAlgorithm.html
- is a: 'Integer relation algorithm', 'Lattice basis reduction algorithm'
- input: 'Basis B and lattice L'
- output: 'LLL-reduced lattice basis'
- applications: 'factorizing polynomials with rational coefficients', 'simultaneous rational approximations to real numbers', 'Integer linear programming', 'MIMO detection', 'Cryptanalysis'
- implemented in: 'Mathematica LatticeReduce'

## PSLQ algorithm
- http://mathworld.wolfram.com/PSLQAlgorithm.html
- is a: 'Integer relation algorithm'

## Fleury's algorithm
- https://en.wikipedia.org/wiki/Eulerian_path#Fleury's_algorithm
- http://mathworld.wolfram.com/FleurysAlgorithm.html
- input: 'Graph'
- output: 'Eulerian cycle' or 'Eulerian trail'

## Blankinship algorithm
- http://mathworld.wolfram.com/BlankinshipAlgorithm.html
- finds: 'Greatest common divisor'
- properties: 'vectorized'

## Splitting algorithm
- http://mathworld.wolfram.com/SplittingAlgorithm.html

## Miller's algorithm
- http://mathworld.wolfram.com/MillersAlgorithm.html
- https://crypto.stanford.edu/pbc/notes/ep/miller.html
- domain: 'Cryptography'
- input: 'Weil pairing on an algebraic curve'

## Jacobi eigenvalue algorithm
- https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
- is a: 'Iterative method'
- applications: 'Matrix diagonalization'
- input: 'real symmetric matrix'
- output: 'eigenvalues and eigenvectors'

## Spectral biclustering
- http://www.kemaleren.com/post/spectral-biclustering-part-2/
- paper: 'Spectral Biclustering of Microarray Cancer Data: Co-clustering Genes and Conditions (2003)'
- solves: 'Biclustering'
- implemented in: 'sklearn.cluster.bicluster.SpectralBiclustering'
- input: 'Matrix'

## Graphical lasso
- https://en.wikipedia.org/wiki/Graphical_lasso
- solves: 'Covariance matrix estimation'
- is a: 'Graphical model'
- domain: 'Bayesian statistics'
- implemented in: 'sklearn.covariance.GraphicalLasso'

## Lanczos algorithm
- https://en.wikipedia.org/wiki/Lanczos_algorithm
- http://mathworld.wolfram.com/LanczosAlgorithm.html
- solves: 'Matrix diagonalization'
- applications: 'Latent semantic analysis'
- variation: 'Block Lanczos algorithm'
- implemented in: 'ARPACK'

## Block Wiedemann algorithm
- https://en.wikipedia.org/wiki/Block_Wiedemann_algorithm
- input: 'matrix over a finite field'
- output: 'kernel vectors'

## David Eppstein's algorithm for finding the synchronizing word of a DFA
- paper: 'Reset Sequences for Monotonic Automata (1990)'
- time complexity O(n^3 + k*n^2). 
- does not always find the shortest possible synchronizing word
- input: 'DFA'
- output: 'Synchronizing word'
- based on: 'Breadth-first search'
- applications: 'Parts orienters'
- supersedes: 'Natarajan's algorithm for finding the reset sequence of a monotoic DFA'

## Apriori algorithm
- https://en.wikipedia.org/wiki/Apriori_algorithm
- applications: 'Association rule learning', 'Data mining'
- based on: 'Breadth-first search'

## Random sample consensus
- also called: 'RANSAC'
- https://en.wikipedia.org/wiki/Random_sample_consensus
- paper: 'Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography (1981)'
- applications: 'Computer vision', 'Location determination problem'

## PACBO
- also called: 'Probably Approximately Correct Bayesian Online'
- paper: 'PAC-Bayesian Online Clustering'
- implemented in: 'R PACBO'
- related: 'RJMCMC'
- applications: 'online clustering'
- domain: 'Game theory', 'Computational learning theory'

## RJMCMC
- also called: 'Reversible-jump Markov chain Monte Carlo'
- paper: 'Reversible jump Markov chain Monte Carlo computation and Bayesian model determination (1995)'
- https://en.wikipedia.org/wiki/Reversible-jump_Markov_chain_Monte_Carlo
- exntension of: 'Markov chain Monte Carlo'

## Simple moving average
- also called: 'SMA'
- https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
- applications: 'Time series analysis'

## Exponential moving average
- also called: 'EMA', 'Exponentially weighted moving average', 'EWMA'
- https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
- applications: 'Time series analysis'

## Weighted moving average
- also called: 'WMA'
- https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average
- applications: 'Time series analysis'

## Backpropagation through time
- https://en.wikipedia.org/wiki/Backpropagation_through_time
- domain: 'Optimization'
- applications: training certain types of 'Recurrent neural networks'

## Backpropagation through structure
- https://en.wikipedia.org/wiki/Backpropagation_through_structure
- domain: 'Optimization'
- applications: training certain types of 'Recursive neural networks'

## Franceschini-Muthukrishnan-Pătrașcu algorithm
- paper: 'Radix Sorting With No Extra Space (2007)'
- applications: 'Integer sorting'

## Approximate Link state algorithm
- also called: 'XL'
- paper: 'XL: An Efficient Network Routing Algorithm (2008)'
- applications: 'Network routing'

## Nicholl–Lee–Nicholl algorithm
- https://en.wikipedia.org/wiki/Nicholl%E2%80%93Lee%E2%80%93Nicholl_algorithm
- applications: 'Line clipping'

## Liang–Barsky algorithm
- https://en.wikipedia.org/wiki/Liang%E2%80%93Barsky_algorithm
- applications: 'Line clipping'

## C3 linearization
- paper: 'A Monotonic Superclass Linearization for Dylan'
- https://en.wikipedia.org/wiki/C3_linearization
- applications: 'Multiple inheritance', 'Method Resolution Order'
- used to implement application: 'Python', 'Perl'

## Sethi–Ullman algorithm
- paper: 'The Generation of Optimal Code for Arithmetic Expressions (1970)'
- https://en.wikipedia.org/wiki/Sethi%E2%80%93Ullman_algorithm
- applications: 'Code generation', 'Arithmetic expressions'
- input: 'Abstract syntax tree'
- ouput: 'Machine code'
- domain: 'Graph theory'

## Yarrow algorithm
- https://en.wikipedia.org/wiki/Yarrow_algorithm
- is a: 'Cryptographically secure pseudorandom number generator'
- succeeded by: 'Fortuna'
- domain: 'Cryptography'

## Fortuna
- https://en.wikipedia.org/wiki/Fortuna_(PRNG)
- is a: 'Cryptographically secure pseudorandom number generator'
- domain: 'Cryptography'

## Blum–Micali algorithm
- https://en.wikipedia.org/wiki/Blum%E2%80%93Micali_algorithm
- is a: 'Cryptographically secure pseudorandom number generator'
- domain: 'Cryptography'

## Double dabble
- also called: 'shift-and-add-3 algorithm'
- https://en.wikipedia.org/wiki/Double_dabble
- input: 'binary numbers'
- ouput: 'binary-coded decimals'
- properties: 'hardware friendly'

## BKM algorithm
- paper: 'BKM: a new hardware algorithm for complex elementary functions (1994)'
- https://en.wikipedia.org/wiki/BKM_algorithm
- is a shift-and-add algorithm for computing elementary functions
- properties: 'hardware friendly'

## CORDIC
- also called: 'COordinate Rotation DIgital Computer', 'Volder's algorithm'
- paper: 'The CORDIC Computing Technique (1959)'
- calculate hyperbolic and trigonometric functions
- properties: 'Digit-by-digit algorithm', 'hardware friendly'
- class: 'Shift-and-add algorithm'

## XOR swap algorithm
- https://en.wikipedia.org/wiki/XOR_swap_algorithm
- patent: 'US4197590A' (expired)

## Xulvi-Brunet and Sokolov algorithm
- paper: 'Reshuffling scale-free networks: From random to assortative (2004)'
- https://en.wikipedia.org/wiki/Xulvi-Brunet_-_Sokolov_algorithm
- generates networks with chosen degree correlations

## Hash join
- https://en.wikipedia.org/wiki/Hash_join
- class: 'Relational join algorithm'
- requires: 'Equi-join predicate'
- implemented in: 'T-SQL'

## Sort-merge join
- also called: 'Merge join'
- https://en.wikipedia.org/wiki/Sort-merge_join
- class: 'Relational join algorithm'
- implemented in: 'T-SQL'

## Berkeley algorithm
- paper: 'The accuracy of the clock synchronization achieved by TEMPO in Berkeley UNIX 4.3BSD (1989)'
- https://en.wikipedia.org/wiki/Berkeley_algorithm
- applications: 'Clock synchronization'
- is a: 'Distributed algorithm'

## Cristian's algorithm
- paper: 'Probabilistic clock synchronization (1989)'
- https://en.wikipedia.org/wiki/Cristian%27s_algorithm
- applications: 'Clock synchronization'
- is a: 'Distributed algorithm'

## Marzullo's algorithm
- thesis: 'Maintaining the time in a distributed system: an example of a loosely-coupled distributed service (1984)'
- https://en.wikipedia.org/wiki/Marzullo%27s_algorithm
- superseeded by: 'Intersection algorithm'
- is a: 'Agreement algorithm'
- applications: 'Clock synchronization'

## Intersection algorithm
- also called: 'Clock Select Algorithm'
- paper: 'Improved algorithms for synchronizing computer network clocks (1995)'
- https://en.wikipedia.org/wiki/Intersection_algorithm
- https://www.eecis.udel.edu/~mills/ntp/html/select.html
- is a: 'Agreement algorithm'
- used by: 'Network Time Protocol'
- applications: 'Clock synchronization'

## Nagle's algorithm
- paper: 'Congestion Control in IP/TCP Internetworks' (RFC896)
- https://en.wikipedia.org/wiki/Nagle%27s_algorithm
- used by: 'TCP/IP'
- applications: 'Congestion Control'
- domain: 'Networking'

## Segmented string relative ranking
- book: 'PUBLIC BRAINPOWER: Civil Society and Natural Resource Management'
- ranking algorithm

## Algorithm for Producing Rankings Based on Expert Surveys
- paper: 'Algorithm for Producing Rankings Based on Expert Surveys (2019)'
- based on: 'Segmented string relative ranking'
- compare: 'PageRank'
- applications: 'Link analysis'

## SALSA algorithm
- also called: 'Stochastic Approach for Link-Structure Analysis'
- paper: 'SALSA: the stochastic approach for link-structure analysis'
- https://en.wikipedia.org/wiki/SALSA_algorithm
- applications: 'Link analysis'

## TextRank
- paper: 'TextRank: Bringing Order into Texts (2004)'
- based on: 'PageRank'
- domain: 'Graph Theory'
- applications: 'Keyword extraction', 'Text summarization'

## HITS algorithm
- also called: 'Hyperlink-Induced Topic Search'
- paper: 'Authoritative sources in a hyperlinked environment (1999)'
- https://en.wikipedia.org/wiki/HITS_algorithm
- applications: 'Link analysis', 'Search engines', 'Citation analysis'

## Eigenfactor
- paper: 'Eigenfactor: Measuring the value and prestige of scholarly journals (2007)'
- https://en.wikipedia.org/wiki/Eigenfactor
- is a: 'Citation metric'

## Impact factor
- https://en.wikipedia.org/wiki/Impact_factor
- is a: 'Citation metric'

## PageRank
- paper: 'The PageRank Citation Ranking: Bringing Order to the Web'
- https://en.wikipedia.org/wiki/PageRank
- domain: 'Graph theory'
- applications: 'Link analysis', 'Linear algebra'
- input: 'Google matrix'

## CheiRank
- https://en.wikipedia.org/wiki/CheiRank
- input: 'Google matrix'
- domain: 'Graph theory', 'Linear algebra'

## ObjectRank
- paper: 'ObjectRank: Authority-Based Keyword Search in Databases'
- applications: 'Ranking in graphs'

## PathSim
- paper: 'PathSim: Meta Path-Based Top-K Similarity Search in Heterogeneous Information Networks'
- applications: 'Similarity search', 'Ranking in graphs'

## RankDex
- also called: 'Hyperlink Vector Voting method', 'HVV'
- paper: 'Toward a qualitative search engine (1998)'

## Banker's algorithm
- paper: 'Een algorithme ter voorkoming van de dodelijke omarming (EWD-108) (1964–1967)'
- also called: 'detection algorithm'
- https://en.wikipedia.org/wiki/Banker%27s_algorithm
- http://www.cs.colostate.edu/~cs551/CourseNotes/Bankers.html
- applications: 'Resource allocation', 'Deadlock prevention'/'Deadlock avoidance'

## Finite difference: central difference
- https://en.wikipedia.org/wiki/Finite_difference#Forward,_backward,_and_central_differences
- implemented in: 'scipy.misc.derivative'

## Deutsch–Jozsa algorithm
- paper: 'Rapid solution of problems by quantum computation (1992)'
- https://en.wikipedia.org/wiki/Deutsch%E2%80%93Jozsa_algorithm
- is a: 'Quantum algorithm'
- properties: 'deterministic'

## Grover's algorithm
- paper: 'A fast quantum mechanical algorithm for database search (1996)'
- https://en.wikipedia.org/wiki/Grover%27s_algorithm
- is a: 'Quantum algorithm'
- properties: 'probabilistic', 'asymptotically optimal'

## Kalman filter
- also called: 'linear quadratic estimation'
- https://en.wikipedia.org/wiki/Kalman_filter
- http://mathworld.wolfram.com/KalmanFilter.html
- domain: 'Control theory'
- implemented in: 'Python statsmodels.tsa.kalmanf.kalmanfilter.KalmanFilter'
- applications: 'guidance, navigation, and control', 'time series analysis', 'Trajectory optimization', 'Computer vision', 'Object tracking'
- solves: 'Linear–quadratic–Gaussian control problem'

## Ensemble Kalman filter
- also called: 'EnKF'
- https://en.wikipedia.org/wiki/Ensemble_Kalman_filter
- is a: 'Recursive filter'
- applications: 'Data assimilation', 'Ensemble forecasting'

## Ensemble adjustment Kalman filter
- also called: 'EAKF'
- paper: 'An Ensemble Adjustment Kalman Filter for Data Assimilation (2001)'

## PF-PMC-PHD
- also called: 'Particle Filter–Pairwise Markov Chain–Probability Hypothesis Density'
- paper: 'Particle Probability Hypothesis Density Filter Based on Pairwise Markov Chains (2019)'
- applications: 'multi-target tracking system'

## Multichannel affine projection algorithm
- paper: 'A multichannel affine projection algorithm with applications to multichannel acoustic echo cancellation (1996)'
- applications: 'Echo cancellation'
- generalization of: 'Affine projection algorithm'

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

# Methods, patterns and programming models

## Map
- https://en.wikipedia.org/wiki/Map_(parallel_pattern)
- parallelizes 'Embarrassingly parallel' problems

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

## MapReduce
- https://en.wikipedia.org/wiki/MapReduce
- implemented by: 'Apache Hadoop'

## Ostrich algorithm
- https://en.wikipedia.org/wiki/Ostrich_algorithm
- is not really an algorithm, but a strategy
- just ignore problems
- used by Windows and Linux to handle deadlocks
