# Algorithms

-- definition:
	An algorithm is a strict description of a function or process. They mustn't require additional data
	and free parameters should only be used to directly influence the output (ie. they shouldn't have to be optimized
	or finetuned based on the input). An algorithm which is not fully specified is called 'method'.
	They can be 'totally correct' or 'partially correct'.

## Breadth-first search
- https://en.wikipedia.org/wiki/Breadth-first_search
- input: 'Graph'
- implemented in: 'boost::graph::breadth_first_search'

## Depth-first search
- https://en.wikipedia.org/wiki/Depth-first_search
- input: 'Graph'
- implemented in: 'boost::graph::depth_first_search'

## Branch and bound search
- implemented in (libraries): 'google/or-tools'
- commonly used to solve: '0-1 knapsack problem'

## k-d tree construction algorithm using sliding midpoint rule
- example paper: Maneewongvatana and Mount 1999
- constructs: 'k-d tree'
- implemented in: 'scipy.spatial.KDTree'
- input: 'List of k-dimensional points'
- output: 'k-d tree'

## Relooper algorithm
- paper: 'Emscripten: an LLVM-to-JavaScript compiler' (2011)
- http://mozakai.blogspot.com/2012/05/reloop-all-blocks.html

## Alternating-least-squares with weighted-λ-regularization
- also called: 'ALS-WR', 'Alternating-least-squares', 'ALS'
- paper: 'Large-Scale Parallel Collaborative Filtering for the Netflix Prize' (2008)
- article: 'Matrix Factorization Techniques for Recommender Systems' (2009)
- optimizes: 'Tensor rank decomposition'
- implemented in (libraries): 'Apache Spark MLlib', 'libFM'
- applications: 'Recommender system', 'Collaborative filtering'

## Alternating slice-wise diagonalization
- also called: 'ASD'
- paper: 'Three-way data resolution by alternating slice-wise diagonalization (ASD) method' (2000)
- optimizes: 'Tensor rank decomposition'

## Positive Matrix Factorisation for 3 way arrays
- also called: 'PMF3'
- paper: 'A weighted non-negative least squares algorithm for three-way ‘PARAFAC’ factor analysis' (1997)
- optimizes: 'Tensor rank decomposition'

## Direct trilinear decomposition
- also called: 'DTLD', 'DTD'
- paper: 'Tensorial resolution: A direct trilinear decomposition' (1990)
- optimizes: 'Tensor rank decomposition'

## Generalised Rank Annihilation Method
- also called: 'GRAM'
- paper: 'Generalized rank annihilation factor analysis' (1986)
- optimizes: 'Tensor rank decomposition'

## Multivariate curve resolution-alternating least squares
- also called: 'MCR-ALS'

## Newton's method
- also called: 'Newton–Raphson method'
- https://en.wikipedia.org/wiki/Newton%27s_method
- approximates: 'Root-finding'
- solves (badly): 'System of polynomial equations'

## Aberth method
- also called: 'Aberth–Ehrlich method'
- https://en.wikipedia.org/wiki/Aberth_method
- input: 'univariate polynomial'
- output: 'roots'
- approximates: 'Root-finding'
- implemented in (application): 'MPSolve (Multiprecision Polynomial Solver)'

## Brent's method
- also called: 'van Wijngaarden-Dekker-Brent method'
- https://en.wikipedia.org/wiki/Brent%27s_method
- http://mathworld.wolfram.com/BrentsMethod.html
- book: 'Algorithms for Minimization without Derivatives', 'Chapter 4: An Algorithm with Guaranteed Convergence for Finding a Zero of a Function' (1973)
- uses: 'Bisection method', 'Secant method', 'Inverse quadratic interpolation'
- implemented in (libraries): 'Netlib', 'scipy.optimize.brentq', 'boost::brent_find_minima'
- approximates (globally): 'Root-finding'

## Jenkins–Traub algorithm for polynomial zeros
- https://en.wikipedia.org/wiki/Jenkins%E2%80%93Traub_algorithm
- paper: 'Algorithm 493: Zeros of a Real Polynomial' (1975)

## Homotopy continuation
- https://en.wikipedia.org/wiki/System_of_polynomial_equations#Homotopy_continuation_method
- https://en.wikipedia.org/wiki/Numerical_algebraic_geometry
- optimizes: 'Tensor rank decomposition'
- solves: 'System of polynomial equations'

## Weisfeiler-Lehman algorithm
- original paper: 'A reduction of a graph to a canonical form and an algebra arising during this reduction' (1968)
- analysis paper: 'The Weisfeiler-Lehman Method and Graph Isomorphism Testing' (2011)
- https://blog.smola.org/post/33412570425/the-weisfeiler-lehman-algorithm-and-estimation-on
- solves sometimes: 'Graph isomorphism problem'
- applications: 'Graph classification'

## Harley-Seal algorithm
- book: 'O'Reilly', 'Beautiful Code (2007)
- applications: 'Hamming weight'
- is a: 'Carry-save adder'

## Cluster pruning
- book: 'Cambridge University Press, Introduction to Information Retrieval' (2008)
- algorithmic analysis: 'Finding near neighbors through cluster pruning' (2007)
- properties: 'randomized', 'external io'
- applications: 'Approximate nearest neighbor search'
- solutions for exact version: 'Linear scan'
- cf: 'p-spheres', 'rank aggregation'

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
- solves: 'Maximum flow problem'
- implemented in: 'boost::graph::push_relabel_max_flow', 'google/or-tools::max_flow'

## Successive approximation push-relabel method
- also called: 'Cost scaling', 'Cost-scaling push-relabel algorithm'
- paper: 'Finding Minimum-Cost Circulations by Successive Approximation' (1990)
- solves: 'Minimum-cost circulation problem', 'Minimum-cost flow problem'
- implemented in: 'google/or-tools::min_cost_flow'

## Extension of Push–relabel for minimum cost flows
- paper: 'An Efficient Implementation of a Scaling Minimum-Cost Flow Algorithm' (1997)
- solves: 'Minimum-cost flow problem'

## Cost-scaling push-relabel algorithm for the assignment problem
- paper: 'An efficient cost scaling algorithm for the assignment problem' (1995)
- solves: 'Linear assignment problem'
- implemented in (libraries): 'google/or-tools::linear_assignment'

## Darga–Sakallah–Markov symmetry-discovery algorithm
- paper: 'Faster symmetry discovery using sparsity of symmetries' (2008)
- implemented in (libraries): 'google/or-tools::find_graph_symmetries'
- solves: 'Graph automorphism problem'

## Gale–Shapley algorithm
- also called: 'Deferred-acceptance algorithm'
- https://www.britannica.com/science/Gale-Shapley-algorithm
- solves: 'Stable marriage problem'

## Floyd's cycle-finding algorithm
- also called: 'Floyd's Tortoise and Hare'
- https://en.wikipedia.org/wiki/Cycle_detection#Floyd's_Tortoise_and_Hare
- solves: 'Cycle detection'

## Kruskal's algorithm
- https://en.wikipedia.org/wiki/Kruskal%27s_algorithm
- http://mathworld.wolfram.com/KruskalsAlgorithm.html
- output: 'Minimum spanning tree'
- properties: 'greedy'
- implemented in: 'C++ boost::graph::kruskal_minimum_spanning_tree', 'google/or-tools::minimum_spanning_tree'
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
- applications: 'Text compression'
- implemented using: 'Suffix array'
- variant: 'Bijective variant'

## Move-to-front transform
- also called: 'MTF transform'
- https://en.wikipedia.org/wiki/Move-to-front_transform
- is a: 'transform'
- properties: 'reversible'
- applications: 'Compression'

## Tarjan's strongly connected components algorithm
- https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
- input: 'Directed graph'
- output: 'Strongly connected component'
- solves: 'Strongly connected component finding'
- implemented in (libraries): 'google/or-tools::graph::strongly_connected_components'

## Kosaraju's algorithm
- also called: 'Kosaraju–Sharir algorithm'
- paper: 'A strong-connectivity algorithm and its applications in data flow analysis' (1981)
- https://en.wikipedia.org/wiki/Kosaraju%27s_algorithm
- solves: 'Strongly connected component finding'

## Awerbuch-Shiloach algorithm for finding the connected components
- paper: 'New Connectivity and MSF Algorithms for Shuffle-Exchange Network and PRAM (1987)'
- is a: 'Parallel algorithm'

## Tarjan's off-line lowest common ancestors algorithm
- https://en.wikipedia.org/wiki/Tarjan%27s_off-line_lowest_common_ancestors_algorithm
- input: 'pairs of nodes in a tree'
- output: 'Lowest common ancestor'

## Adam7 algorithm
- https://en.wikipedia.org/wiki/Adam7_algorithm
- applications: 'Image interlacing'
- input: 'Raster image'
- used by (file format): 'Portable Network Graphics'
- properties: '2-dimensional'

## Lloyd–Max quantization
- paper: 'Quantizing for minimum distortion' (1960)
- paper: 'Least squares quantization in PCM' (1982)
- applications: 'Quantization'
- domain: 'Signal processing'

## Pyramid Vector Quantization
- also called: 'PVQ'
- paper: 'A pyramid vector quantizer' (1986)
- applications: 'Quantization'
- domain: 'Signal processing'

## Default algorithm for Huffman Tree
- https://en.wikipedia.org/wiki/Huffman_coding#Compression
- applications: 'Huffman coding'
- properties: 'greedy'
- uses 'priority queue'

## Arithmetic coding
- https://en.wikipedia.org/wiki/Arithmetic_coding
- form of: 'Entropy encoding'
- applications: 'Lossless compression'
- domain: 'Information theory'

## Huffman coding
- paper: 'A Method for the Construction of Minimum-Redundancy Codes' (1952)
- https://en.wikipedia.org/wiki/Huffman_coding
- uses: 'Default algorithm for Huffman Tree'
- related code: 'Huffman code'
- domain: 'Information theory', 'Coding theory'
- form of: 'Entropy encoding'

## Context-adaptive binary arithmetic coding
- also called: 'CABAC'
- paper: 'Context-based adaptive binary arithmetic coding in the H.264/AVC video compression standard' (2003)
- https://en.wikipedia.org/wiki/Context-adaptive_binary_arithmetic_coding
- based on: 'Arithmetic coding'
- applications: 'Lossless compression'

## t-digest
- whitepaper: 'Computing extremely accurate quantiles using t-digests'
- Q-digest
- approximates percentiles
- is a: 'distributed algorithm'

## Block-matching and 3D filtering
- also called: 'BM3D'
- paper: 'Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering' (2007)
- http://www.cs.tut.fi/~foi/GCF-BM3D/
- https://en.wikipedia.org/wiki/Block-matching_and_3D_filtering
- is a: 'Block-matching algorithm'
- applications: 'Digital image noise reduction'
- implemented in: 'gfacciol/bm3d', 'HomeOfVapourSynthEvolution/VapourSynth-BM3D'

## Chazelle's algorithm for the minimum spanning tree
- paper: 'A minimum spanning tree algorithm with inverse-Ackermann type complexity (2000)'
- output: 'Minimum spanning tree'
- uses: 'soft heaps'

## Chazelle's polygon triangulation algorithm
- paper: 'Triangulating a simple polygon in linear time (1991)'
- properties: 'very difficult to implement in code'

## Risch semi-algorithm
- paper: 'On the Integration of Elementary Functions which are built up using Algebraic Operations (1968)'
- https://en.wikipedia.org/wiki/Risch_algorithm
- http://mathworld.wolfram.com/RischAlgorithm.html
- solves: 'Indefinite integration'
- applications: 'Symbolic computation', 'Computer algebra'
- implemented in: 'Axiom'
- properties: 'very difficult to implement in code'

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
- also called: 'Kuhn-Munkres algorithm', 'Hungarian method'
- paper: 'The Hungarian method for the assignment problem (1955)'
- https://en.wikipedia.org/wiki/Hungarian_algorithm
- http://mathworld.wolfram.com/HungarianMaximumMatchingAlgorithm.html
- https://brilliant.org/wiki/hungarian-matching/
- input: 'bipartite graph'
- output: 'maximum-weight matching'
- time complexity: O(V^3) for V vertices
- domain: 'Graph theory', 'Combinatorial optimization'
- implemented in (libraries): 'bmc/munkres'

## Kuhn–Munkres algorithm with backtracking
- also called: 'KM_B algorithm'
- paper: 'Solving the Many to Many assignment problem by improving the Kuhn–Munkres algorithm with backtracking' (2015)
- solves: 'Many-to-many assignment problem'

## Cuthill–McKee algorithm
- paper: 'Reducing the bandwidth of sparse symmetric matrices' (1969)
- https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm
- http://ciprian-zavoianu.blogspot.com/2009/01/project-bandwidth-reduction.html
- domain: 'Numerical linear algebra'
- solves approximately: 'Bandwidth reduction problem'
- is a: 'Heuristic algorithm'
- variant of: 'Breadth-first search'
- implemented in (libraries): 'boost::cuthill_mckee_ordering', 'networkx.utils.rcm.cuthill_mckee_ordering'

## Reverse Cuthill–McKee algorithm
- also called: 'RCM'
- https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm
- domain: 'Numerical linear algebra'
- variant of: 'Cuthill–McKee algorithm'
- implemented in (libraries): 'networkx.utils.rcm.reverse_cuthill_mckee_ordering', 'scipy.sparse.csgraph.reverse_cuthill_mckee'

## GPS algorithm
- also called: 'Gibbs-Poole-Stockmeyer algorithm'
- paper: 'An Algorithm for Reducing the Bandwidth and Profile of a Sparse Matrix' (1976)
- solves approximately: 'Bandwidth reduction problem'

## Jonker-Volgenant algorithm
- paper: 'A shortest augmenting path algorithm for dense and sparse linear assignment problems' (1987)
- solves: 'Linear assignment problem'
- implemented in (libraries): 'src-d/lapjv', 'Python scipy.optimize.linear_sum_assignment' (in new versions)
- improvement of: 'Hungarian Maximum Matching algorithm'

## Vogel's approximation method
- also called: 'VAM'
- book: 'Pearson', 'Introduction to Management Science' (Module B)
- https://www.linearprogramming.info/vogel-approximation-method-transportation-algorithm-in-linear-programming/
- https://businessjargons.com/vogels-approximation-method.html
- solves partly: 'Transportation problem'
- type: 'Initial feasible solution'

## Northwest corner method
- also called: 'NWCM'
- book: 'Pearson', 'Introduction to Management Science' (Module B)
- https://www.linearprogramming.info/northwest-corner-method-transportation-algorithm-in-linear-programming/
- https://businessjargons.com/north-west-corner-rule.html
- solves partly: 'Transportation problem'
- is a: 'heuristic'
- type: 'Initial feasible solution'

## Minimum cell cost method
- also called: 'Least cost method', 'LCM'
- book: 'Pearson', 'Introduction to Management Science' (Module B)
- https://study.com/academy/lesson/using-the-minimum-cost-method-to-solve-transportation-problems.html
- https://businessjargons.com/least-cost-method.html
- solves partly: 'Transportation problem'
- type: 'Initial feasible solution'

## Stepping Stone Method
- book: 'Pearson', 'Introduction to Management Science' (Module B)
- https://businessjargons.com/stepping-stone-method.html
- solves partly: 'Transportation problem'
- type: 'Optimality check'

## Modified Distribution Method
- book: 'Pearson', 'Introduction to Management Science' (Module B)
- also called: 'MODI'
- https://businessjargons.com/modified-distribution-method.html
- solves partly: 'Transportation problem'
- type: 'Optimality check'

## Transportation simplex algorithm
- solves: 'Transportation problem'

## Auction algorithm
- paper: 'A distributed algorithm for the assignment problem' (1979)
- https://en.wikipedia.org/wiki/Auction_algorithm
- domain: 'Combinatorial optimization'
- solves: 'Maximum weight matching'
- implemented in: 'maxdan94/auction'

## Roth-Peranson matching algorithm
- paper: 'The Redesign of the Matching Market for American Physicians: Some Engineering Aspects of Economic Design' (1999)
- implemented in: 'J-DM/Roth-Peranson'
- applications: 'National Resident Matching Program'

## MaxCliqueDyn maximum clique algorithm
- paper: 'An improved branch and bound algorithm for the maximum clique problem' (2007)
- https://en.wikipedia.org/wiki/MaxCliqueDyn_maximum_clique_algorithm
- solves: 'Maximum clique'

## Edmonds' algorithm
- also called: 'Chu–Liu/Edmonds' algorithm'
- paper: 'On the Shortest Arborescence of a Directed Graph' (1965)
- paper: 'Optimum Branchings' (1967)
- https://en.wikipedia.org/wiki/Edmonds%27_algorithm
- solves: 'Optimum branching AKA Minimum spanning arborescence'
- implemented in (libraries): 'networkx.algorithms.tree.branchings.Edmonds'

## Blossom algorithm
- also called: 'Edmonds' matching algorithm'
- paper: 'Paths, trees, and flowers (1965)'
- https://en.wikipedia.org/wiki/Blossom_algorithm
- http://mathworld.wolfram.com/BlossomAlgorithm.html
- https://brilliant.org/wiki/blossom-algorithm/
- domain: 'Graph theory'
- input: 'Graph'
- output: 'Maximum matching'
- time complexity: O(E V^2) for E edges and V vertices

## Micali and Vazirani's matching algorithm
- runtime complexity: O(sqrt(n) m) for n vertices and m edges
- paper: 'An O(sqrt(|v|) |E|) algoithm for finding maximum matching in general graphs' (1980)
- exposition paper: 'The general maximum matching algorithm of micali and vazirani' (1988)

## Xiao and Nagamochi's algorithm for the maximum independent set
- paper: 'Exact algorithms for maximum independent set (2017)'
- solves: 'Maximum independent set problem'
- time complexity: O(1.1996^n)
- space complexity: polynomial
- superseeds: 'Robson (1986)'

## Luby's algorithm
- https://en.wikipedia.org/wiki/Maximal_independent_set
- also called: 'Random-selection parallel algorithm'
- solves: 'Finding a maximal independent set'

## Blelloch's algorithm
- https://en.wikipedia.org/wiki/Maximal_independent_set
- also called: 'Random-permutation parallel algorithm'
- solves: 'Finding a maximal independent set'

## Boppana and Halldórsson's approximation algorithm for the maximum independent set
- paper: 'Approximating maximum independent sets by excluding subgraphs (1992)'
- solves approximate: 'Maximum independent set problem'
- implemented in: 'Python networkx.algorithms.approximation.independent_set.maximum_independent_set'
- time complexity: O(n / (log b)^2)

## Davis–Putnam algorithm
- also called: 'DP algorithm'
- paper: 'A Computing Procedure for Quantification Theory' (1960)
- https://en.wikipedia.org/wiki/Davis%E2%80%93Putnam_algorithm
- solves: 'Boolean satisfiability problem'

## DPLL algorithm
- paper: 'A machine program for theorem-proving (1962)'
- https://en.wikipedia.org/wiki/DPLL_algorithm
- also called: 'Davis–Putnam–Logemann–Loveland algorithm'
- solves: 'Boolean satisfiability problem'
- applications: 'Automated theorem proving'
- extension of: 'Davis–Putnam algorithm'
- properties: 'complete', 'sound'

## Conflict-driven clause learning
- paper: 'GRASP-A new search algorithm for satisfiability'
- https://en.wikipedia.org/wiki/Conflict-driven_clause_learning
- solves: 'Boolean satisfiability problem'
- implemented by: 'MiniSAT', 'Zchaff SAT', 'Z3', 'ManySAT'
- extension of: 'DPLL algorithm'
- properties: 'complete', 'sound'

## Literal Block Distance
- also called: 'LBD'
- paper: 'Predicting Learnt Clauses Quality in Modern SAT Solver' (2009)
- is a: 'heuristic'
- applications: 'Boolean satisfiability problem'
- implemented by: 'Glucose SAT Solver'

## Canny's Roadmap algorithm
- http://planning.cs.uiuc.edu/node298.html
- properties: 'very difficult to implement in code'
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

## Hashlife
- paper: 'Exploiting Regularities in Large Cellular Spaces' (1984)
- https://en.wikipedia.org/wiki/Hashlife
- uses: 'Memoization', 'Quadtree'
- solves: 'Conway's Game of Life'
- implemented in (application): 'Golly'

## Diamond-square algorithm
- also called: 'random midpoint displacement fractal', 'cloud fractal', 'plasma fractal'
- paper: 'Computer rendering of stochastic models' (1982)
- https://en.wikipedia.org/wiki/Diamond-square_algorithm
- variants: 'Lewis algorithm'
- applications: 'Heightmap generation', 'Procedural textures'
- domain: 'Computer graphics'
- implemented in (applications): 'Terragen'
- books: 'The Science of Fractal Images' (1988)

## Lewis algorithm
- paper: 'Generalized stochastic subdivision' (1987)
- variant of: 'Diamond-square algorithm'
- applications: 'Heightmap generation', 'Procedural textures'
- domain: 'Computer graphics'

## Seam carving
- https://en.wikipedia.org/wiki/Seam_carving
- uses method: 'Dynamic programming'
- applications: 'Image retargeting'
- domain: 'computer graphics'
- implemented in: 'Adobe Photoshop', 'GIMP'

## Multi-operator
- paper: 'Multi-operator Media Retargeting' (2009)
- applications: 'Image retargeting'

## Symmetry-Summarization
- paper: 'Resizing by Symmetry-Summarization' (2010)
- applications: 'Image retargeting'
- uses: 'Maximally stable extremal regions', 'Mean shift'

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

## Dinic's algorithm
- also called: 'Dinitz's algorithm'
- https://en.wikipedia.org/wiki/Dinic%27s_algorithm
- solves: 'Maximum flow problem'

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

## Lloyd's algorithm
- https://en.wikipedia.org/wiki/Lloyd%27s_algorithm
- is a: 'Iterative method'
- input: 'Collection of points'
- output: 'Voronoi diagram'
- approximates: 'Centroidal Voronoi tessellation'

## Linde–Buzo–Gray algorithm
- paper: 'An Algorithm for Vector Quantizer Design' (1980)
- https://en.wikipedia.org/wiki/Linde%E2%80%93Buzo%E2%80%93Gray_algorithm
- similar: 'k-means clustering'
- generalization of: 'Lloyd's algorithm'
- applications: 'Vector quantization'

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

## MUSIC
- also called: 'MUltiple SIgnal Classification'
- https://en.wikipedia.org/wiki/MUSIC_(algorithm)
- applications: 'Frequency estimation', 'Direction finding'

## SAMV
- also called: 'Iterative sparse asymptotic minimum variance'
- paper: 'Iterative Sparse Asymptotic Minimum Variance Based Approaches for Array Processing' (2012)
- https://en.wikipedia.org/wiki/SAMV_(algorithm)
- solves: 'Super-resolution imaging'
- applications: 'Synthetic-aperture radar', 'Computed tomography', 'Magnetic resonance imaging'
- properties: 'parameter-free'

## Nearest-neighbor interpolation
- also called: 'Proximal interpolation', 'Point sampling'
- https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
- applications: 'Multivariate interpolation', 'Image scaling'

--- move these to models.md 

## Bilinear interpolation
- https://en.wikipedia.org/wiki/Bilinear_interpolation
- applications: 'Multivariate interpolation', 'Image scaling'
- optimized by: analytically

## Bilinear filtering
- also called: 'Bilinear texture mapping'
- https://en.wikipedia.org/wiki/Bilinear_filtering
- applications: 'Image scaling', 'Texture filtering', 'Texture mapping'
- domain: 'Computer vision', 'Digital image processing'
- uses: 'Bilinear interpolation'

## Trilinear interpolation
- https://en.wikipedia.org/wiki/Trilinear_interpolation
- applications: 'Multivariate interpolation'
- extension of: 'Bilinear interpolation'
- optimized by: analytically

## Trilinear filtering
- extension of: 'Bilinear filtering' (betweens mipmaps, not time)
- https://en.wikipedia.org/wiki/Trilinear_filtering
- uses: 'Mipmap'
- applications: 'Texture filtering', 'Texture mapping'

## Anisotropic filtering
- also called: 'AF'
- https://en.wikipedia.org/wiki/Anisotropic_filtering
- descritpion: 'perspectively weighted sampling'

## Bicubic interpolation
- https://en.wikipedia.org/wiki/Bicubic_interpolation
- extension of: 'Cubic interpolation'
- applications: 'Image scaling', 'Multivariate interpolation'
- domain: 'Digital image processing'

## Tricubic interpolation
- https://en.wikipedia.org/wiki/Tricubic_interpolation
- applications: 'Multivariate interpolation'
- implemented in (libraries): 'DurhamDecLab/ARBInterp'

## Spline interpolation
- https://en.wikipedia.org/wiki/Spline_interpolation
- avoids: 'Runge's phenomenon'
- implemented in (libraries): 'scipy.interpolate', 'FITPACK'

## Lanczos resampling
- https://en.wikipedia.org/wiki/Lanczos_resampling
- applications: 'Multivariate interpolation'
- uses: 'Sinc function'

## Lanczos3
- see: 'Lanczos resampling'
- implemented in (applications): 'GIMP'

## Spline36
- see: 'Spline interpolation'
- implemented in (applications): 'AviSynth'

## Elliptical Weighted Average Filter
- also called: 'EWA'
- paper: 'Creating Raster Omnimax Images from Multiple Perspective Views Using the Elliptical Weighted Average Filter' (1986)
- paper: 'High quality elliptical texture filtering on GPU' (2011)
- applications: 'Texture filtering'

## SSimSuperRes
- also called: 'SSSR'
- https://gist.github.com/igv/2364ffa6e81540f29cb7ab4c9bc05b6b
- applications: 'Super resolution', 'Luma upscaling'

## SSimDownscaler
- applications: 'Luma downscaling', 'Real-time image processing'
- https://gist.github.com/igv/36508af3ffc84410fe39761d6969be10
- Sota: 'Real-time luma downscaling'

## KrigBilateral
- applications: 'Chroma upscaling', 'Chroma downscaling'
- Sota: 'Chroma upscaling', 'Chroma downscaling'
- https://gist.github.com/igv/a015fc885d5c22e6891820ad89555637

## NGU
- http://madvr.com/
- properties: 'proprietary'
- applications: 'Image scaling'
- implemented in (applications): 'madVR'

## waifu2x
- https://en.wikipedia.org/wiki/Waifu2x
- official website: http://waifu2x.udp.jp/
- inspired by: 'SRCNN'
- implemented in (applications): 'nagadomi/waifu2x'
- uses: 'CNN'
- applications: 'Single image super-resolution'

## Anime4K
- https://github.com/bloc97/Anime4K
- applications: 'Single image super-resolution', 'Real-time image processing'
- implemented in (applications): 'bloc97/Anime4K'

## New edge-directed interpolation
- also called: 'NEDI'
- paper: 'New edge-directed interpolation' (2001)

## Edge-guided image interpolation
- also called: 'EGGI'
- paper: 'An edge-guided image interpolation algorithm via directional filtering and data fusion' (2006)

## Iterative Curvature-Based Interpolation
- also called: 'ICBI'
- paper: 'Enlargement of Image Based Upon Interpolation Techniques' (2013)

## Directional cubic convolution interpolation
- also called: 'DCCI'
- paper: 'Image zooming using directional cubic convolution interpolation' (2012)

## hqx
- https://en.wikipedia.org/wiki/Hqx
- applications: 'Pixel-art scaling'
- implemented in (libraries): 'libretro'
- variants: 'hq2x', 'hq3x', 'hq4x'

## scalehq
- applications: 'Pixel-art scaling'
- implemented in (libraries): 'libretro'

## xBR algorithm
- also called: 'scale by rules'
- https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms#xBR_family
- implemented in (libraries): 'libretro'

## Eagle
- https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms#Eagle
- applications: 'Pixel-art scaling'

## 2×SaI
- also called: '2× Scale and Interpolation engine'
- https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms#2.C3.97SaI
- inspired by: 'Eagle'
- implemented in (applications): 'DosBox'
- applications: 'Pixel-art scaling'

## RotSprite
- https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms#RotSprite
- applications: 'Pixel-art scaling'

## Kopf–Lischinski algorithm
- paper: 'Depixelizing pixel art' (2011)
- applications: 'Image tracing'

## Unsharp masking
- also called: 'USM'
- https://en.wikipedia.org/wiki/Unsharp_masking
- applications: 'Image sharpening'
- domain: 'Digital image processing'

## Deep Learning Super Sampling
- also called: 'DLSS'
- properties: 'proprietary'
- applications: 'Image scaling'

## Super sampling anti-aliasing
- also called: 'Supersampling', 'SSAA', 'Full-scene anti-aliasing', 'FSAA'
- https://en.wikipedia.org/wiki/Supersampling
- applications: 'Spatial anti-aliasing'

## Multisample anti-aliasing
- also called: 'MSAA'
- https://en.wikipedia.org/wiki/Multisample_anti-aliasing
- applications: 'Spatial anti-aliasing'
- special case of: 'Supersampling'
- properties: 'doesn't support deferred shading'

## Fast approximate anti-aliasing
- also called: 'FXAA', 'Fast sample anti-aliasing', 'FSAA'
- https://en.wikipedia.org/wiki/Fast_approximate_anti-aliasing
- applications: 'Spatial anti-aliasing'
- implemented in (libraries): 'Unity'

## Intel's MLAA
- also called: 'Morphological anti-aliasing', 'MLAA'
- paper: 'Morphological antialiasing' (2009)
- applications: 'Real-time anti-aliasing'
- properties: 'CPU based', 'supports deferred shading'

## Jimenez's MLAA
- also called: 'Morphological anti-aliasing', 'MLAA'
- book: 'GPU Pro 2'
- http://iryoku.com/mlaa/
- applications: 'Real-time anti-aliasing'
- properties: 'GPU based'

## Enhanced subpixel morphological antialiasing
- also called: 'SMAA'
- http://www.iryoku.com/smaa/
- paper: 'SMAA: Enhanced Morphological Antialiasing' (2012)
- applications: 'Real-time anti-aliasing'
- implemented in: 'iryoku/smaa'
- implemented in (libraries): 'Unity'

## Subpixel reconstruction anti-aliasing
- also called: 'SRAA'
- https://research.nvidia.com/publication/subpixel-reconstruction-antialiasing
- paper: 'Subpixel reconstruction antialiasing for deferred shading' (2011)
- applications: 'Real-time anti-aliasing'

## Directionally localized anti-aliasing
- also called: 'DLAA'

## Temporal Anti-aliasing
- also called: 'TAA'
- implemented in (libraries): 'Unity'

## Screen space ambient occlusion
- also called: 'SSAO'
- paper: 'Finding next gen: CryEngine 2' (2007)
- https://en.wikipedia.org/wiki/Screen_space_ambient_occlusion
- https://learnopengl.com/Advanced-Lighting/SSAO
- approximates: 'Ambient occlusion'

## SSAO+
- approximates: 'Ambient occlusion'

## Horizon-based ambient occlusion
- also called: 'HBAO"
- paper: 'Image-space horizon-based ambient occlusion' (2008)

## HBAO+
- https://www.geforce.com/hardware/technology/hbao-plus
- improvement of: 'Horizon-based ambient occlusion'

## Ray traced ambient occlusion
- also called: 'RTAO'
- approximates: 'Ambient occlusion'

--- models end

## Variable Number of Gradients
- also called: 'VNG'
- implemented in (applications): 'dcraw'
- applications: 'Demosaicing'

## Pixel Grouping
- also called: 'PPG', 'Patterned Pixel Grouping'
- implemented in (applications): 'dcraw'
- applications: 'Demosaicing'

## Adaptive Homogeneity-Directed
- also called: 'AHD'
- implemented in (applications): 'dcraw'
- applications: 'Demosaicing'

## Aliasing Minimization and Zipper Elimination
- also called: 'AMaZE'
- implemented in (applications): 'RawTherapee'
- applications: 'Demosaicing'

## CLEAN
- https://en.wikipedia.org/wiki/CLEAN_(algorithm)
- paper: 'Aperture Synthesis with a Non-Regular Distribution of Interferometer Baselines' (1974)
- applications: 'Radio astronomy', 'Deconvolution'
- domain: 'Digital image processing'

## Richardson–Lucy deconvolution
- also called: 'R–L algorithm'
- https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
- applications: 'Deconvolution'
- domain: 'Digital image processing'
- implemented in (applications): 'RawTherapee'
- uses: 'expectation-maximization algorithm'
- assumes distribution: 'Poisson noise'

## Richardson–Lucy deconvolution (blind variant)
- paper: 'Blind deconvolution by means of the Richardson–Lucy algorithm' (1995)
- based on: 'Richardson–Lucy deconvolution'
- applications: 'Blind deconvolution'

## Ayers–Dainty algorithm

## Davey–Lane–Bates algorithm

## Van-Cittert deconvolution
- also called: 'Van-Cittert iteration'
- https://de.wikipedia.org/wiki/Van-Cittert-Dekonvolution
- applications: 'Deconvolution'
- domain: 'Digital image processing'

## SeDDaRA
- applications: 'Blind deconvolution'

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
- paper: 'Algorithm 457: finding all cliques of an undirected graph' (1973)
- https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
- http://mathworld.wolfram.com/Bron-KerboschAlgorithm.html
- input: 'Undirected graph'
- output: all 'Maximal Clique'
- solves: 'Finding all maximal cliques'
- applications: 'Computational chemistry'
- properties: 'not output-sensitive'
- implemented in (libraries): 'Python networkx.algorithms.clique.find_cliques', 'google/or-tools::graph::cliques::BronKerboschAlgorithm'
- domain: 'Graph theory'

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

## Union-find algorithm
- https://www.algorithmist.com/index.php/Union_Find
- uses: 'Disjoint-set data structure'
- solves: 'Connected-component finding'

## Hoshen–Kopelman algorithm
- paper: 'Percolation and cluster distribution. I. Cluster multiple labeling technique and critical concentration algorithm (1976)'
- https://en.wikipedia.org/wiki/Hoshen%E2%80%93Kopelman_algorithm
- is a: 'Cluster analysis algorithm'
- input: 'regular network of bools'
- solves: 'Connected-component labeling'

## Fiorio's algorithm for connected-component labeling
- paper: 'Two linear time Union-Find strategies for image processing (1996)'
- solves: 'Connected-component labeling'
- implemented in: 'Python skimage.measure.label'
- version of: 'Union-find algorithm'
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

## Ruppert's algorithm
- also called: 'Delaunay refinement'
- paper: 'A Delaunay Refinement Algorithm for Quality 2-Dimensional Mesh Generation' (1995)
- https://en.wikipedia.org/wiki/Ruppert%27s_algorithm
- output: 'Delaunay triangulation'
- applications: 'Computational fluid dynamics', 'Finite element analysis'

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
- paper: 'Canonical regular expressions and minimal state graphs for definite events' (1962)
- https://en.wikipedia.org/wiki/DFA_minimization#Brzozowski's_algorithm
- solves: 'DFA minimization'
- uses: 'Powerset construction'
- input: 'Deterministic finite automaton'
- output: 'Minimal deterministic finite automaton'

## Regular-expression derivatives
- paper: 'Derivatives of Regular Expressions' (1964)
- https://en.wikipedia.org/wiki/Brzozowski_derivative
- input: 'Regular expression'
- output: 'Deterministic finite automaton'

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
- also called: 'DLX' implemented using 'Dancing Links'
- https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X
- solves: 'Exact cover problem'
- applications: 'Sudoku', 'Tessellation', 'Eight queens puzzle'
- properties: 'nondeterministic'

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
- implemented in (libraries): 'networkx.algorithms.flow.min_cost_flow'

## Expectation–maximization algorithm
- also called: 'EM algorithm'
- https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm
- is a: 'iterative method'
- applications: 'Parameter estimation'
- properties: 'deterministic'

## Wake-sleep algorithm
- https://en.wikipedia.org/wiki/Wake-sleep_algorithm
- is a: 'Unsupervised learning algorithm'
- properties: 'convergent'
- trains: 'Helmholtz machine'
- domain: 'Machine learning'

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

## Gaussian elimination
- https://en.wikipedia.org/wiki/Gaussian_elimination
- http://mathworld.wolfram.com/GaussianElimination.html
- solves: 'System of linear equations', 'Determinant', 'Matrix inversion'

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

## Graphical lasso ?model or algorithm?
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

## EigenTrust
- paper: 'The Eigentrust algorithm for reputation management in P2P networks' (2003)
- https://en.wikipedia.org/wiki/EigenTrust
- applications: 'Reputation management', 'Peer-to-peer networking'

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

## Synthetic Minority Over-sampling Technique
- also called: 'SMOTE'
- paper: 'SMOTE: Synthetic Minority Over-sampling Technique' (2002)
- applications: 'Class imbalance problem'
- implemented in: 'imblearn.over_sampling.SMOTE'

## Synthetic Minority Over-sampling Technique for Nominal and Continuous
- also called: 'SMOTE-NC'
- paper: 'SMOTE: Synthetic Minority Over-sampling Technique' (2002)
- variant of: 'Synthetic Minority Over-sampling Technique'
- applications: 'Class imbalance problem'
- implemented in: 'mblearn.over_sampling.SMOTENC'

## Borderline-SMOTE
- paper: 'Borderline-SMOTE: a new over-sampling method in imbalanced data sets learning' (2005)
- variant of: 'Synthetic Minority Over-sampling Technique'
- applications: 'Class imbalance problem'
- implemented in: 'imblearn.over_sampling.BorderlineSMOTE'

## Borderline Over-sampling
- also called: 'BOS'
- paper: 'Borderline over-sampling for imbalanced data classification' (2009)
- variant of: 'Synthetic Minority Over-sampling Technique'
- applications: 'Class imbalance problem'
- implemented in: 'imblearn.over_sampling.SVMSMOTE'

## Adaptive synthetic sampling approach
- also called: 'ADASYN'
- paper: 'ADASYN: Adaptive synthetic sampling approach for imbalanced learning' (2008)
- applications: 'Class imbalance problem'
- implemented in: 'imblearn.over_sampling.ADASYN'

## Condensed nearest neighbours
- paper: 'The condensed nearest neighbor rule' (1968)
- applications: 'Class imbalance problem'
- implemented in: 'imblearn.under_sampling.CondensedNearestNeighbour'

## Edited nearest neighbours
- also called: 'ENN'
- paper: 'Asymptotic Properties of Nearest Neighbor Rules Using Edited Data' (1972)
- applications: 'Class imbalance problem'
- implemented in: 'imblearn.under_sampling.EditedNearestNeighbours'

## Tomek links
- paper: 'Two Modifications of CNN' (1976)
- applications: 'Class imbalance problem'
- implemented in: 'imblearn.under_sampling.TomekLinks'

## Instance hardness threshold
- paper: 'An instance level analysis of data complexity' (2014)
- applications: 'Class imbalance problem'
- implemented in: 'imblearn.under_sampling.InstanceHardnessThreshold'

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

## Maximally stable extremal regions
- paper: 'Robust wide baseline stereo from maximally stable extremal regions' (2002)
- https://en.wikipedia.org/wiki/Maximally_stable_extremal_regions
- applications: 'Computer vision'
- implemented in: 'OpenCV::MSER'

## Mean shift
- also called: 'Adaptive mean shift clustering'
- paper: 'Mean shift: a robust approach toward feature space analysis' (2002)
- implemented in: 'sklearn.cluster.MeanShift'
- is a: adaptive gradient ascent method
- properties: 'centroid based'
- applications: 'Clustering'

## Variational Bayes
- also called: 'VB'
- https://en.wikipedia.org/wiki/Variational_Bayesian_methods
- applications: 'Statistical inference'

## Gibbs sampling
- https://en.wikipedia.org/wiki/Gibbs_sampling
- type of: 'Markov chain Monte Carlo'
- applications: 'Sampling', 'Statistical inference'
- properties: 'randomized'

## Collapsed Gibbs sampling
- https://en.wikipedia.org/wiki/Gibbs_sampling#Collapsed_Gibbs_sampler
- variant of: 'Gibbs sampling'
- samples (eg.): 'Latent Dirichlet allocation'

## Hamiltonian Monte Carlo
- also called: 'HMC', 'Hybrid Monte Carlo'
- https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo
- paper: 'Hybrid Monte Carlo (1987)'
- is a: 'Markov chain Monte Carlo algorithm'
- solves: 'Sampling'
- input: 'probability distribution'
- output: 'random samples'
- applications: 'Lattice QCD'
- implemented in: 'Stan'

## No-U-Turn Sampler
- also called: 'NUTS'
- paper: 'The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo (2011)'
- extends: 'Hamiltonian Monte Carlo'
- implemented in: 'Stan'
- solves: 'Sampling'

## Swendsen–Wang algorithm
- https://en.wikipedia.org/wiki/Swendsen%E2%80%93Wang_algorithm
- is a: 'Monte Carlo method'
- simulates: 'Ising model'

## Wolff algorithm
- https://en.wikipedia.org/wiki/Wolff_algorithm
- is a: 'Monte Carlo method'
- simulates: 'Ising model'

## KH-99 algorithm
- paper: 'RNA secondary structure prediction using stochastic context-free grammars and evolutionary history' (1999)
- applications: 'RNA structure prediction'

## Pfold
- paper: 'Pfold: RNA secondary structure prediction using stochastic context-free grammars' (2003)
- applications: 'RNA structure prediction'

## TextTiling
- paper: 'Multi-paragraph segmentation of expository text' (1994)
- paper: 'Text Tiling: Segmenting Text into Multi-paragraph Subtopic Passages' (1997)
- applications: 'Natural language processing', 'Text Segmentation', 'Topic segmentation'
- implemented in (libraries): 'Python nltk.tokenize.texttiling.TextTilingTokenizer'
- input: 'text'
- output: 'semantic paragraphs'

## TopicTiling
- paper: 'TopicTiling: A Text Segmentation Algorithm based on LDA' (2012)
- based on: 'TextTiling'
- applications: 'Natural language processing', 'Text Segmentation', 'Topic segmentation'
- input: 'text'
- output: 'semantic paragraphs'
- uses: 'Latent Dirichlet Allocation'

## GraphSeg
- paper: 'Unsupervised Text Segmentation Using Semantic Relatedness Graphs' (2016)
- applications: 'Natural language processing', 'Text Segmentation', 'Topic segmentation'

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

## Verhoeff algorithm
- paper: 'Error Detecting Decimal Codes' (1969)
- https://en.wikipedia.org/wiki/Verhoeff_algorithm
- applications: 'Error detection', 'Checksum'

## Damm algorithm
- https://en.wikipedia.org/wiki/Damm_algorithm
- book: 'Introduction to Computer Data Representation. Checksums and Error Control'
- applications: 'Error detection'
- cf: 'Verhoeff algorithm'

## Luhn algorithm
- also called: 'Luhn formula', 'modulus 10', 'mod 10 algorithm'
- https://en.wikipedia.org/wiki/Luhn_algorithm
- https://patents.google.com/patent/US2950048
- applications: 'Error detection', 'Checksum', 'Identification numbers validation'

## Luhn's algorithm
- paper: 'The Automatic Creation of Literature Abstracts' (1958)
- applications: 'Extractive text summarization'

## Neuroevolution of augmenting topologies
- also called: 'NEAT'
- https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies
- http://nn.cs.utexas.edu/?stanley:ec02
- is a: 'Genetic algorithm'
- applications: 'Neuroevolution', 'Machine learning'

## Evolutionary acquisition of neural topologies
- also called: 'EANT', 'EANT2'
- https://en.wikipedia.org/wiki/Evolutionary_acquisition_of_neural_topologies
- applications: 'Neuroevolution', 'Machine learning', 'Reinforcement learning'
- is a: 'Evolutionary algorithm'

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
