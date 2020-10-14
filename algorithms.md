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
- usually implemented with: 'Double-ended queue'

## Depth-first search
- https://en.wikipedia.org/wiki/Depth-first_search
- input: 'Graph'
- implemented in: 'boost::graph::depth_first_search'

## Depth-first search for topological sorting
- paper: 'Edge-disjoint spanning trees and depth-first search' (1976)
- https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
- applications: 'topological sorting', 'Strongly connected component'

## Branch and bound search
- implemented in (libraries): 'google/or-tools'
- commonly used to solve: '0-1 knapsack problem'

## Parzen–Rosenblatt window method
- also called: 'Parzen-window method'
- paper: 'Remarks on Some Nonparametric Estimates of a Density Function' (1956) <https://doi.org/10.1214%2Faoms%2F1177728190>
- https://en.wikipedia.org/wiki/Kernel_density_estimation
- applications: 'Kernel density estimation'
- implemented in: 'Python scipy.signal.parzen'

## Kernel density estimation using Gaussian kernels
- applications: 'Kernel density estimation'
- implemented in: 'Python scipy.stats.gaussian_kde', 'Mathematica SmoothKernelDistribution[ker->"Gaussian"]'

## k-d tree construction algorithm using sliding midpoint rule
- example paper: Maneewongvatana and Mount 1999
- constructs: 'k-d tree'
- implemented in: 'scipy.spatial.KDTree'
- input: 'List of k-dimensional points'
- output: 'k-d tree'

## Relooper algorithm
- paper: 'Emscripten: an LLVM-to-JavaScript compiler' (2011)
- http://mozakai.blogspot.com/2012/05/reloop-all-blocks.html

## Alternating least squares
- also called: 'Alternating-least-squares', 'ALS'
- paper: 'Analysis of individual differences in multidimensional scaling via an n-way generalization of “Eckart-Young” decomposition' (1970) <https://doi.org/10.1007/BF02310791>
- paper: 'Foundations of the PARAFAC procedure : Models and conditions for an "explanatory" multi-mode factor analysis' (1970)
- optimizes: 'Tensor rank decomposition'
- implemented in (libraries): 'tensorly.decomposition.parafac', 'tensortools.cp_als'

## Alternating-least-squares with weighted-λ-regularization
- also called: 'ALS-WR'
- paper: 'Large-Scale Parallel Collaborative Filtering for the Netflix Prize' (2008) <https://doi.org/10.1007/978-3-540-68880-8_32>
- article: 'Matrix Factorization Techniques for Recommender Systems' (2009)
- optimizes: 'Tensor rank decomposition'
- implemented in (libraries): 'Apache Spark MLlib', 'libFM'
- applications: 'Recommender system', 'Collaborative filtering'

## Alternating slice-wise diagonalization
- also called: 'ASD'
- paper: 'Three-way data resolution by alternating slice-wise diagonalization (ASD) method' (2000) <https://doi.org/10.1002/(SICI)1099-128X(200001/02)14:1%3C15::AID-CEM571%3E3.0.CO;2-Z>
- optimizes: 'Tensor rank decomposition'

## Positive Matrix Factorisation for 3 way arrays
- also called: 'PMF3'
- paper: 'A weighted non-negative least squares algorithm for three-way ‘PARAFAC’ factor analysis' (1997) <https://doi.org/10.1016/S0169-7439(97)00031-2>
- optimizes: 'Tensor rank decomposition'

## Direct trilinear decomposition
- also called: 'DTLD', 'DTD'
- paper: 'Tensorial resolution: A direct trilinear decomposition' (1990) <https://doi.org/10.1002/cem.1180040105>
- optimizes: 'Tensor rank decomposition'

## Generalised Rank Annihilation Method
- also called: 'GRAM'
- paper: 'Generalized rank annihilation factor analysis' (1986) <https://doi.org/10.1021/ac00293a054>
- optimizes: 'Tensor rank decomposition'

## Multivariate curve resolution-alternating least squares
- also called: 'MCR-ALS'

## Newton's method
- also called: 'Newton–Raphson method'
- https://en.wikipedia.org/wiki/Newton%27s_method
- approximates: 'Root-finding'
- solves (badly): 'System of polynomial equations'
- implemented in (libraries): 'scipy.optimize.newton'

## Aberth method
- also called: 'Aberth–Ehrlich method'
- https://en.wikipedia.org/wiki/Aberth_method
- input: 'univariate polynomial'
- output: 'roots'
- approximates: 'Root-finding'
- implemented in (application): 'MPSolve (Multiprecision Polynomial Solver)'
- implemented in: 'afoures/aberth-method'

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
- original paper: 'A Three-Stage Algorithm for Real Polynomials Using Quadratic Iteration' (1970) <https://doi.org/10.1137/0707045>
- paper: 'Algorithm 493: Zeros of a Real Polynomial' (1975)
- implemented in (libraries): 'sweeneychris/RpolyPlusPlus'

## Homotopy continuation
- https://en.wikipedia.org/wiki/System_of_polynomial_equations#Homotopy_continuation_method
- https://en.wikipedia.org/wiki/Numerical_algebraic_geometry
- optimizes: 'Tensor rank decomposition'
- solves: 'System of polynomial equations'
- implemented in: 'janverschelde/PHCpack', 'phcpy'

## Weisfeiler-Lehman algorithm
- original paper: 'A reduction of a graph to a canonical form and an algebra arising during this reduction' (1968)
- analysis paper: 'The Weisfeiler-Lehman Method and Graph Isomorphism Testing' (2011)
- https://blog.smola.org/post/33412570425/the-weisfeiler-lehman-algorithm-and-estimation-on
- solves sometimes: 'Graph isomorphism problem'
- applications: 'Graph classification'
- implemented in: 'grakel.WeisfeilerLehman'

## Harley-Seal algorithm
- book: 'O'Reilly', 'Beautiful Code (2007)
- applications: 'Hamming weight'
- is a: 'Carry-save adder'
- implemented in: 'WojciechMula/sse-popcount'

## Cluster pruning
- book: 'Cambridge University Press', 'Introduction to Information Retrieval' (2008)
- algorithmic analysis: 'Finding near neighbors through cluster pruning' (2007)
- properties: 'randomized', 'external io'
- applications: 'Approximate nearest neighbor search'
- solutions for exact version: 'Linear scan'
- cf: 'p-spheres', 'rank aggregation'

## Median of medians
- also called: 'PICK', Blum-Floyd-Pratt-Rivest-Tarjan partition algorithm', 'BFPRT'
- paper: 'Time bounds for selection' (1973) <https://doi.org/10.1016/S0022-0000(73)80033-9>
- https://en.wikipedia.org/wiki/Median_of_medians
- solves: 'Selection problem'
- input: 'random access collection'
- properties: 'deterministic'

## Introselect
- paper: 'Introspective Sorting and Selection Algorithms' (1997) <https://doi.org/10.1002/(SICI)1097-024X(199708)27:8%3C983::AID-SPE117%3E3.0.CO;2-%23>
- https://en.wikipedia.org/wiki/Introselect
- implemented in: 'C++ std::nth_element, 'numpy.partition'
- solves: 'Selection problem'
- input: 'random access collection'

## Floyd–Rivest algorithm
- paper: 'Algorithm 489: the algorithm SELECT—for finding the ith smallest of n elements [M1]' (1975) <https://doi.org/10.1145/360680.360694>
- https://en.wikipedia.org/wiki/Floyd%E2%80%93Rivest_algorithm
- solves: 'Selection problem'
- is a: 'Divide and conquer algorithm'
- input: 'random access collection'

## Quickselect
- also called: 'Hoare's selection algorithm'
- paper: 'Algorithm 65: find' (1961) <https://doi.org/10.1145/366622.366647>
- https://en.wikipedia.org/wiki/Quickselect
- solves: 'Selection problem'
- input: 'random access collection'
- properties: 'parallelizable', 'randomized'

## Dijkstra's algorithm
- paper: 'A note on two problems in connexion with graphs' (1959) <https://doi.org/10.1007/BF01386390>
- book: 'MIT Press', 'Introduction to Algorithms'
- https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
- http://mathworld.wolfram.com/DijkstrasAlgorithm.html
- uses method: 'Dynamic programming'
- solves 'Single-source shortest path problem' for non-negative weights in directed/undirected graphs in O(v^2) where v is the number of vertices
- variant implementation with 'Fibonacci heap' runs in O(e * v*log v) where e and v are the number of edges and vertices resp.
- implemented in (libraries): 'Python scipy.sparse.csgraph.shortest_path(method="D")', 'boost::graph::dijkstra_shortest_paths'
- implemented in (lattice variant): 'skimage.graph.MCP'
- Fibonacci implementation is the asymptotically the fastest known single-source shortest-path algorithm for arbitrary directed graphs with unbounded non-negative weights.
- input: 'Directed graph with non-negative weights'

## Bellman–Ford algorithm
- also called: 'Bellman–Ford–Moore algorithm'
- paper: 'Structure in communication nets' (1953) <https://doi.org/10.1007/BF02476438>
- paper: 'On a routing problem' (1958) <https://doi.org/10.1090/qam/102435>
- book: 'MIT Press', 'Introduction to Algorithms'
- https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm
- solves variant of the 'Shortest path problem' for real-valued edge weights in directed graph in O(v*e) where v and e are the number of vertices and edges respectively.
- negative cycles are detected
- implemented in: 'Python scipy.sparse.csgraph.shortest_path(method="BF")', 'boost:graph::bellman_ford_shortest_paths'
- input: 'Weighted directed graph'

## Johnson's algorithm
- paper: 'Efficient Algorithms for Shortest Paths in Sparse Networks' (1977) <https://doi.org/10.1145/321992.321993>
- https://en.wikipedia.org/wiki/Johnson%27s_algorithm
- solves 'All-pairs shortest paths problem' for real-valued weights in a directed graph in O(v^2 log v + v*e) where v and e are the number of vertices and edges
- implemented in: 'Python scipy.sparse.csgraph.shortest_path(method='J')', 'C++ boost::graph::johnson_all_pairs_shortest_paths'
- combination of 'Bellman–Ford' and 'Dijkstra's algorithm'
- is faster than 'Floyd–Warshall algorithm' for sparse graphs
- input: 'weighted directed graph without negative cycles'
- challenges: 'SPOJ: JHNSN'

## Floyd–Warshall algorithm
- paper: 'Algorithm 97: Shortest path' (1962) <https://doi.org/10.1145/367766.368168>
- https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
- http://mathworld.wolfram.com/Floyd-WarshallAlgorithm.html
- domain: 'Graph theory'
- solves 'All-pairs shortest paths problem' for real-valued weights for directed/undirected graphs in O(v^3) where v is the number of vertices
- negative cycles are not allowed
- uses method: 'dynamic programming'
- implemented in: 'python scipy.sparse.csgraph.shortest_path(method='FW')', 'c++ boost::graph::floyd_warshall_all_pairs_shortest_paths', 'networkx.algorithms.shortest_paths.dense.floyd_warshall'
- is faster than 'Johnson's algorithm' for dense graphs
- operates in: 'weighted directed graph without negative cycles'

## Suurballe's algorithm
- paper: 'Disjoint paths in a network' (1974) <https://doi.org/10.1002/net.3230040204>
- https://en.wikipedia.org/wiki/Suurballe%27s_algorithm
- implemented in: 'Python nildo/suurballe'
- uses: 'Dijkstra's algorithm'
- solves: 'Shortest pair of edge disjoint paths'
- input: 'Directed graph with non-negative weights'

## Edge disjoint shortest pair algorithm
- paper: 'Survivable networks: algorithms for diverse routing' (1998) [978-0-7923-8381-9]
- https://en.wikipedia.org/wiki/Edge_disjoint_shortest_pair_algorithm
- solves: 'Shortest pair of edge disjoint paths'
- superseded by: 'Suurballe's algorithm'
- input: 'Weighted directed graph'

## Reaching algorithm
- also called: 'DAG-Shortest-Paths'
- http://mathworld.wolfram.com/ReachingAlgorithm.html
- solves: 'Shortest path problem'
- book: 'MIT Press', 'Introduction to Algorithms'
- time complexity: O(n), where n is the number of edges
- input: 'Directed acyclic graph'
- uses: 'Breadth-first search', 'Dynamic programming', 'Topological sort'
- implemented in: 'boost::graph::dag_shortest_paths'

## Collaborative diffusion
- also called: 'Dijkstra flow maps'
- paper: 'Collaborative diffusion: programming antiobjects' (2006) <https://doi.org/10.1145/1176617.1176630>
- https://en.wikipedia.org/wiki/Collaborative_diffusion
- applications: 'pathfinding'
- time complexity: constant in the number of agents
- implemented in: 'C glouw/pather'

## Ukkonen's algorithm
- https://en.wikipedia.org/wiki/Ukkonen%27s_algorithm
- paper: 'On-line construction of suffix trees' (1995) <https://doi.org/10.1007/BF01206331>
- book: 'Cambridge', 'Algorithms on Strings, Trees, and Sequences'
- output: 'suffix tree'
- properties: 'online'
- time complexity: O(n), where n is the length of the string
- input: 'List of strings'
- implemented in: 'C++ adamserafini/suffix-tree', 'Python kasramvd/SuffixTree', 'Python mutux/Ukkonen-s-Suffix-Tree-Algorithm'

## Weiner's linear-time suffix tree algorithm
- book: 'Cambridge', 'Algorithms on Strings, Trees, and Sequences'
- output: 'suffix tree'
- superseded by: 'Ukkonen's algorithm'

## McCreight's algorithm
- book: 'Cambridge', 'Algorithms on Strings, Trees, and Sequences'
- output: 'suffix tree'
- superseded by: 'Ukkonen's algorithm'

## A* search algorithm
- paper: 'A Formal Basis for the Heuristic Determination of Minimum Cost Paths' (1968) <https://doi.org/10.1109/TSSC.1968.300136>
- https://en.wikipedia.org/wiki/A*_search_algorithm
- tutorial: https://www.redblobgames.com/pathfinding/a-star/implementation.htmll
- generalization of 'Dijkstra's algorithm'
- heuristic search
- informed search algorithm (best-first search)
- usually implemented using: 'Priority queue'
- applications: 'Pathfinding', 'Parsing using stochastic grammars in NLP'
- uses method: 'Dynamic programming'
- input: 'Weighted graph'
- implemented in: 'networkx.algorithms.shortest_paths.astar.astar_path'

## Linear search
- https://en.wikipedia.org/wiki/Linear_search
- find element in any sequence in O(i) time where i is the index of the element in the sequence
- works on: 'Linked list', 'Array', 'List'
- has an advantage when sequential access is fast compared to random access
- O(1) for list with geometric distributed values
- implemented in: 'C++ std::find (impl. dependent)', 'Python list.index'
- input: 'List'

## Binary search algorithm
- https://en.wikipedia.org/wiki/Binary_search_algorithm
- find element in sorted finite list in O(log n) time where n is the number of elements in list
- requires: 'Random access'
- variants: 'Exponential search'
- implemented in: 'C++ std::binary_search', 'Python bisect'
- input: 'Sorted list'

## Naïve string-search algorithm
- https://en.wikipedia.org/wiki/String-searching_algorithm#Na%C3%AFve_string_search
- find string in string in O(n+m) average time and O(n*m) worst case, where n and m are strings to be search for, resp. in.
- implemented in: 'C++ std::search (impl. dependent)', 'python list.index'
- input: 'Buffered list'

## Exponential search
- also called: 'Algorithm U', 'doubling search', 'galloping search', 'Struzik search'
- paper: 'An almost optimal algorithm for unbounded searching' (1976) <https://doi.org/10.1016/0020-0190(76)90071-5>
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
- thesis: 'The Complexity of Parallel Computations' (1979)
- solves: 'List ranking'
- related: 'Cumulative sum'
- is a: 'Parallel algorithm'
- time complexity: O(log n) on n processors in parallel
- succeeded by: 'Anderson–Miller algorithm'

## Anderson–Miller algorithm
- paper: 'Deterministic parallel list ranking' (1988) <https://doi.org/10.1007/BFb0040376>
- solves: 'List ranking'
- is a: 'Parallel algorithm'
- properties: 'optimal', 'deterministic'
- time complexity: O(log n) on n/log(n) processors in parallel
- abstract machine: 'EREW PRAM'

## Reid-Miller–Blelloch algorithm
- paper: 'List Ranking and List Scan on the CRAYC90' (1996) <https://doi.org/10.1006/jcss.1996.0074>
- solves: 'List ranking'
- is a: 'Parallel algorithm'
- time complexity: O(log^2 n)

## Helman–JáJá algorithm
- paper: 'Designing Practical Efficient Algorithms for Symmetric Multiprocessors' (1999) <https://doi.org/10.1007/3-540-48518-X_3>
- solves: 'List ranking'
- is a: 'Parallel algorithm'

## Funnelsort
- paper: 'Cache-oblivious algorithms' (1999) <https://doi.org/10.1109/SFFCS.1999.814600>
- https://en.wikipedia.org/wiki/Funnelsort
- is a: 'cache-oblivious algorithm', 'external memory algorithm', 'Comparison-based sorting algorithm'
- input: 'Collection'

## Quicksort
- paper: 'Algorithm 64: Quicksort' (1961) <https://doi.org/10.1145/366622.366644>
- https://en.wikipedia.org/wiki/Quicksort
- http://mathworld.wolfram.com/Quicksort.html
- book: 'MIT Press', 'Introduction to Algorithms'
- is a: 'Sorting algorithm', 'Unstable sorting algorithm', 'In-place algorithm', 'Divide and conquer algorithm', 'Comparison-based sorting algorithm'
- time complexity (best): O(n log n)
- time complexity (average): O(n log n)
- time complexity (worst): O(n^2)
- space complexity: O(log n) auxiliary
- input: 'Random access collection'
- properties: easily parallelizable
- implemented in: 'C qsort'

## Radix sort
- also called: 'Bucket sort', 'Digital sort'
- https://en.wikipedia.org/wiki/Radix_sort
- http://opendatastructures.org/ods-cpp/11_2_Counting_Sort_Radix_So.html#SECTION001522000000000000000
- input: 'Collection of integers'

## Bubble sort
- also called: 'Sinking sort'
- https://en.wikipedia.org/wiki/Bubble_sort
- input: 'Bidirectional Collection'
- it's bad, only applicable to almost sorted inputs
- properties: 'stable', 'in-place'

## Gnome sort
- also called: 'Stupid sort'
- article: 'Stupid Sort: A new sorting algorithm' (2000)
- https://en.wikipedia.org/wiki/Gnome_sort
- time complexity (average, worst): O(n^2)
- time complexity (best): O(n)
- requires no nested loops

## Splaysort
- paper: 'Splaysort: Fast, Versatile, Practical' (1996) <https://doi.org/10.1002/(SICI)1097-024X(199607)26:7%3C781::AID-SPE35%3E3.0.CO;2-B>
- https://en.wikipedia.org/wiki/Splaysort
- based on: 'Splay tree'
- properties: 'comparison based'

## Cocktail shaker sort
- book chapter: 'Sorting by Exchanging' in 'Art of Computer Programming. 3. Sorting and Searching' (1973)
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
- thesis: 'Information sorting in the application of electronic digital computers to business operations' (1954) by 'H. H. Seward'
- https://en.wikipedia.org/wiki/Counting_sort
- is a: 'Integer sorting algorithm'
- properties: 'parallelizable'

## Heapsort
- https://en.wikipedia.org/wiki/Heapsort
- http://mathworld.wolfram.com/Heapsort.html
- book: 'MIT Press', 'Introduction to Algorithms'
- is a: 'Sorting algorithm', 'In-place algorithm', 'Unstable sorting algorithm', 'Comparison-based sorting algorithm'
- solves: 'Partial sorting'
- time complexity (average, best, worst): O(n log n)
- space complexity: O(1)
- uses: 'max heap'
- not easily parallelizable
- variant works on 'doubly linked lists'
- input: 'Random access collection'

## Ultimate heapsort
- paper: 'The Ultimate Heapsort' (1998)
- variant of: 'Heapsort'
- comparisons: n log_2 n + O(1)

## Timsort
- https://en.wikipedia.org/wiki/Timsort
- is a: 'Sorting algorithm', 'Stable sorting algorithm', 'Comparison-based sorting algorithm'
- implemented in: 'Python sorted', 'Android Java'
- input: 'Random access collection'

## Introsort
- paper: 'Introspective Sorting and Selection Algorithms' (1997) <https://doi.org/10.1002/(SICI)1097-024X(199708)27:8%3C983::AID-SPE117%3E3.0.CO;2-%23>
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
- paper: 'The pairwise sorting network' (1992) <https://doi.org/10.1142/S0129626492000337>
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
- also called: 'Knuth shuffle', 'Algorithm P (Shuffling)'
- https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
- is a: 'Shuffling algorithm', 'In-place algorithm'
- unbiased
- input: 'Random access collection'
- applications: 'Card dealing'

## Alan Waterman's reservoir sampling algorithm
- also called: 'Algorithm R'
- https://en.wikipedia.org/wiki/Reservoir_sampling
- is a: 'randomized algorithms'
- version of: 'Fisher–Yates shuffle'
- runtime complexity: 'O(N)'

## Sattolo's algorithm
- paper: 'An algorithm to generate a random cyclic permutation' (1986) <https://doi.org/10.1016/0020-0190(86)90073-6>
- https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#Sattolo's_algorithm
- https://danluu.com/sattolo/
- version of: 'Fisher–Yates shuffle'
- output: random cyclic permutations of length n

## Vitter's reservoir sampling algorithm
- also called: 'Algorithm Z'
- https://en.wikipedia.org/wiki/Reservoir_sampling
- paper: 'Random sampling with a reservoir' (1985) <https://doi.org/10.1145/3147.3165>
- https://xlinux.nist.gov/dads/HTML/reservoirSampling.html
- properties: 'online'
- is a: 'randomized algorithms'
- runtime complexity: 'O(n * (1 + log(N/n)))' (using the threshold optimization)

## Reservoir sampling (optimal)
- also called: 'Algorithm L'
- paper: 'Reservoir-sampling algorithms of time complexity O(n(1 + log(N/n)))' (1994) <https://doi.org/10.1145/198429.198435>
- properties: 'online'
- is a: 'randomized algorithms'

## Cache-oblivious distribution sort
- https://en.wikipedia.org/wiki/Cache-oblivious_distribution_sort
- comparison-based sorting algorithm
- cache-oblivious algorithm

## Naive Method for SimRank by Jeh and Widom
- paper: 'SimRank: a measure of structural-context similarity' (2002) <https://doi.org/10.1145/775047.775126>
- calculate: 'SimRank'

## De Casteljau's algorithm
- https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
- paper: 'Système d'aide à la définition et à l'usinage de surfaces de carosserie' (1971)
- evaluate polynomials in Bernstein form or Bézier curves
- properties: 'numerically stable'
- applications: 'Computer aided geometric design'
- implemented in: 'dhermes/bezier'

## Clenshaw algorithm
- also called: 'Clenshaw summation'
- https://en.wikipedia.org/wiki/Clenshaw_algorithm
- evaluate polynomials in Chebyshev form
- implemented in: 'orthopy.clenshaw'

## Wagner–Fischer algorithm
- https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm
- uses method: 'dynamic programming'
- calculates: 'Levenshtein distance'
- O(n*m) complexity where n and m are the respective string lenths
- optimal time complexity for problem proven to be O(n^2), so this algorithm is pretty much optimal
- space complexity of O(n*m) could be reduced to O(n+m)
- input: 'two strings'

## Aho–Corasick algorithm
- paper: 'Efficient string matching: An aid to bibliographic search' (1975) <https://doi.org/10.1145/360825.360855>
- https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm
- https://xlinux.nist.gov/dads/HTML/ahoCorasick.html
- multiple *string searching*
- implemented in (applications): original fgrep
- implemented in (libraries): 'pyahocorasick'
- (pre)constructs 'Finite-state machine' from set of search strings
- applications: virus signature detection
- classification 'constructed search engine', 'match prefix first', 'one-pass'
- shows better results than 'Commentz-Walter' for peptide identification according to 'Commentz-Walter: Any Better Than Aho-Corasick For Peptide Identification?' and for biological sequences according to 'A Comparative Study On String Matching Algorithms Of Biological Sequences'
- input: 'Collection of strings' (construction)
- input: 'List of characters' (searching)

## Commentz-Walter algorithm
- https://en.wikipedia.org/wiki/Commentz-Walter_algorithm
- multiple *string searching*
- classification 'match suffix first'
- implemented in (applications): grep (variant)

## Boyer–Moore string-search algorithm
- paper: 'A Fast String Searching Algorithm' (1977) <https://doi.org/10.1145/359842.359859>
- https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_string-search_algorithm
- single *string searching*
- implemented in (applications): 'grep'
- implemented in (libraries): 'C++ std::boyer_moore_searcher'
- implemented in (languages): 'Python str' (variant)
- better for large alphabets like text than: 'Knuth–Morris–Pratt algorithm'

## Knuth–Morris–Pratt algorithm
- https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm
- book: 'MIT Press', 'Introduction to Algorithms'
- single *string searching*
- implemented in (applications): 'grep'
- better for small alphabets like DNA than: 'Boyer–Moore string-search algorithm'

## Rabin–Karp algorithm
- also called: 'Karp–Rabin algorithm'
- https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm
- book: 'MIT Press', 'Introduction to Algorithms'
- is a: 'String-searching algorithm'
- single/multiple *string searching*
- time complexity (worst): O(m+n)
- space complexity: constant
- applications: 'Plagiarism detection'

## Bitap algorithm
- also called: 'shift-or algorithm', 'shift-and algorithm', 'Baeza-Yates–Gonnet algorithm'
- https://en.wikipedia.org/wiki/Bitap_algorithm
- solves: 'Approximate string matching'
- implemented in (applications): 'agrep'

## Myers' Diff Algorithm
- paper: 'An O(ND) difference algorithm and its variations' (1986) <https://doi.org/10.1007/BF01840446>
- solves: 'Shortest Edit Script'
- input: two strings
- output: 'Shortest Edit Script'
- implemented by (applications): 'diff', 'git' (Linear space variant)
- variants: 'Linear space'

## Patience Diff method
- https://bramcohen.livejournal.com/73318.html
- requires: 'diff algorithm'

## Non-negative matrix factorization
- also called: 'NMF', 'NNMF'
- https://en.wikipedia.org/wiki/Non-negative_matrix_factorization
- applications: 'Collaborative filtering', 'Dimensionality reduction', 'Text mining'
- implemented in (libraries): 'Python sklearn.decomposition.NMF'

## Beam search
- https://en.wikipedia.org/wiki/Beam_search
- is a: 'heuristic search algorithm'
- properties: 'greedy'
- optimization of best-first search
- greedy version of breadth-first search
- applications: 'Machine translation', 'Speech recognition'
- approximate solution

## Hu–Tucker algorithm
- paper: 'Optimal Computer Search Trees and Variable-Length Alphabetical Codes' (1971) <https://doi.org/10.1137/0121057>
- superseded by: 'Garsia–Wachs algorithm'

## Garsia–Wachs algorithm
- paper: 'A New Algorithm for Minimum Cost Binary Trees' (1977) <https://doi.org/10.1137/0206045>
- https://en.wikipedia.org/wiki/Garsia%E2%80%93Wachs_algorithm
- input: 'List of non-negative reals'
- output: 'Optimal binary search tree' (special case)

## Knuth's optimal binary search tree algorithm
- https://en.wikipedia.org/wiki/Optimal_binary_search_tree#Knuth%27s_dynamic_programming_algorithm
- paper: 'Optimum binary search trees' (1971) <https://doi.org/10.1007/BF00264289>
- uses method: 'dynamic programming'
- output: 'Optimal binary search tree'
- time comlexity: O(n^2)

## Mehlhorn's nearly optimal binary search tree algorithm
- paper 'Nearly optimal binary search trees' (1975) <https://doi.org/10.1007/BF00264563>
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
- paper: 'Dynamic Programming Treatment of the Travelling Salesman Problem' (1962) <https://doi.org/10.1145/321105.321111>
- paper: 'A Dynamic Programming Approach to Sequencing Problems' (1962) <https://doi.org/10.1137/0110015>
- https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm
- uses method: 'dynamic programming'
- solves: 'Travelling salesman problem'

## Christofides algorithm
- https://en.wikipedia.org/wiki/Christofides_algorithm
- solves approximately: 'Travelling salesman problem' (for metric distances)
- implemented in: 'google/or-tools::ChristofidesPathSolver'
- report: 'Worst-case analysis of a new heuristic for the travelling salesman problem' (1976)

## Push–relabel maximum flow algorithm
- paper: 'A new approach to the maximum flow problem' (1986) <https://doi.org/10.1145/12130.12144>
- paper: 'A new approach to the maximum-flow problem' (1988) <https://doi.org/10.1145/48014.61051>
- https://en.wikipedia.org/wiki/Push–relabel_maximum_flow_algorithm
- https://cp-algorithms.com/graph/push-relabel.html
- solves: 'Maximum flow problem'
- implemented in (libraries): 'boost::graph::push_relabel_max_flow', 'google/or-tools::max_flow'

## Improved push–relabel maximum flow algorithm
- paper: 'Analysis of preflow push algorithms for maximum network flow' (1989) <https://doi.org/10.1137/0218072>
- https://cp-algorithms.com/graph/push-relabel-faster.html
- solves: 'Maximum flow problem'

## Successive approximation push-relabel method
- also called: 'Cost scaling', 'Cost-scaling push-relabel algorithm'
- paper: 'Finding Minimum-Cost Circulations by Successive Approximation' (1990) <https://doi.org/10.1287/moor.15.3.430>
- solves: 'Minimum-cost circulation problem', 'Minimum-cost flow problem'
- implemented in (libraries): 'google/or-tools::min_cost_flow', 'lemon::CostScaling'

## Cycle canceling
- solves: 'Minimum-cost flow problem'
- paper: 'A Primal Method for Minimal Cost Flows with Applications to the Assignment and Transportation Problems' (1967) <https://doi.org/10.1287/mnsc.14.3.205>
- implemented in (libraries): 'lemon::CycleCanceling(Method=SIMPLE_CYCLE_CANCELING)'

## Minimum mean cycle canceling
- paper: 'Finding minimum-cost circulations by canceling negative cycles' (1989) <https://doi.org/10.1145/76359.76368>
- implemented in (libraries): 'lemon::CycleCanceling(Method=MINIMUM_MEAN_CYCLE_CANCELING)'
- solves: 'Minimum-cost flow problem'

## Cancel-and-tighten algorithm
- solves: 'Minimum-cost flow problem'
- implemented in (libraries): 'lemon::CycleCanceling(Method=CANCEL_AND_TIGHTEN)'
- runtime complexity: O(n^2 e^2 log(n))

## Extension of Push–relabel for minimum cost flows
- paper: 'An Efficient Implementation of a Scaling Minimum-Cost Flow Algorithm' (1997) <https://doi.org/10.1006/jagm.1995.0805>
- solves: 'Minimum-cost flow problem'

## Cost-scaling push-relabel algorithm for the assignment problem
- paper: 'An efficient cost scaling algorithm for the assignment problem' (1995) <https://doi.org/10.1007/BF01585996>
- solves: 'Linear assignment problem'
- implemented in (libraries): 'google/or-tools::linear_assignment'

## Darga–Sakallah–Markov symmetry-discovery algorithm
- paper: 'Faster symmetry discovery using sparsity of symmetries' (2008) <https://doi.org/10.1145/1391469.1391509>
- implemented in (libraries): 'google/or-tools::find_graph_symmetries'
- solves: 'Graph automorphism problem'

## Gale–Shapley algorithm
- also called: 'Deferred-acceptance algorithm'
- paper: 'College Admissions and the Stability of Marriage' (1962) <https://doi.org/10.1080/00029890.1962.11989827> <https://doi.org/10.2307/2312726>
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
- book: 'MIT Press', 'Introduction to Algorithms'

## Prim's algorithm
- https://en.wikipedia.org/wiki/Prim%27s_algorithm
- output: 'Minimum spanning tree'
- properties: 'greedy'
- implemented in 'C++ boost::graph::prim_minimum_spanning_tree'
- time complexity depends on used data structures
- book: 'MIT Press', 'Introduction to Algorithms'

## Hierholzer's algorithm
- https://en.wikipedia.org/wiki/Eulerian_path#Hierholzer's_algorithm
- https://www.geeksforgeeks.org/hierholzers-algorithm-directed-graph/
- input: 'Finite graph'
- output: 'Eulerian path'
- more efficient than: 'Fleury's algorithm'

## Kahn's algorithm
- paper: 'Topological sorting of large networks' (1962) <https://doi.org/10.1145/368996.369025>
- https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
- applications: 'topological sorting'
- time complexity: O(|V| + |E|)
- input: 'Directed acyclic graph'

## HyperLogLog++
- https://en.wikipedia.org/wiki/HyperLogLog
- solves approximately: 'Count-distinct problem'
- implemented by (libraries): 'Lucence'

## Hunt–McIlroy algorithm
- also called: 'Hunt–Szymanski algorithm'
- https://en.wikipedia.org/wiki/Hunt%E2%80%93McIlroy_algorithm
- solves: 'Longest common subsequence problem'
- technical report: 'An Algorithm for Differential File Comparison'
- implemented by (application): 'diff'
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
- original paper: 'Applications of Path Compression on Balanced Trees' (1979)
- paper (refined version): 'A linear-time algorithm for a special case of disjoint set union' (1983)
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
- implemented in: 'Python networkx.algorithms.bipartite.matching.hopcroft_karp_matching', 'sofiatolaosebikan/hopcroftkarp'

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
- implemented in (libraries): 'bmc/munkres', 'scipy.optimize.linear_sum_assignment' (older versions)
- solves: 'Linear assignment problem'
- variants: 'Jonker-Volgenant algorithm'

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
- implemented in: 'networkx.algorithms.matching.max_weight_matching'

## Micali and Vazirani's matching algorithm
- runtime complexity: O(sqrt(n) m) for n vertices and m edges
- original paper: 'An O(sqrt(|v|) |E|) algoithm for finding maximum matching in general graphs' (1980)
- exposition paper: 'The general maximum matching algorithm of micali and vazirani' (1988)

## Xiao and Nagamochi's algorithm for the maximum independent set
- paper: 'Exact algorithms for maximum independent set' (2017)
- solves: 'Maximum independent set problem'
- time complexity: O(1.1996^n)
- space complexity: polynomial
- superseeds: 'Robson (1986)'

## Luby's algorithm
- https://en.wikipedia.org/wiki/Maximal_independent_set#Random-selection_parallel_algorithm_[Luby's_Algorithm]
- also called: 'Random-selection parallel algorithm'
- solves: 'Finding a maximal independent set'

## Blelloch's algorithm
- https://en.wikipedia.org/wiki/Maximal_independent_set#Random-permutation_parallel_algorithm_[Blelloch's_Algorithm]
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

## Ramer–Douglas–Peucker algorithm
- also called: 'Iterative end-point fit algorithm', 'Duda–Hart split-and-merge algorithm'
- paper: 'An iterative procedure for the polygonal approximation of plane curves' (1972) <https://doi.org/10.1016/S0146-664X(72)80017-0>
- https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
- implemented in: 'skimage.measure.approximate_polygon', 'psimpl'
- applications: 'Downsampling'
- runtime complexity: O(n log n)
- runtime complexity (worst case): O(n^2)

## Visvalingam–Whyatt algorithm
- discussion paper: 'Line generalisation by repeated elimination of the smallest area ' (1992) <https://hull-repository.worktribe.com/output/459275>
- applications: 'Downsampling'
- properties: 'area based'

## Reumann–Witkam algorithm
- paper: 'Optimizing curve segmentation in computer graphics' (1974) <>
- implemented in: 'psimpl'
- applications: 'Downsampling'

## Sleeve-fitting polyline simplification algorithm
- also called: 'Zhao–Saalfeld algorithm'
- paper: 'Linear-Time Sleeve-Fitting Polyline Simplification Algorithms' (1997) <>
- applications: 'Downsampling'

## Opheim simplification algorithm
- paper: 'Smoothing a digitized curve by data reduction methods' (1981) <https://doi.org/10.2312/eg.19811012>
- applications: 'Downsampling'
- implemented in: 'psimpl'

## Lang simplification algorithm
- paper: 'Rules for robot draughtsmen' (1969) <>
- applications: 'Downsampling'
- implemented in: 'psimpl'

## Canny's Roadmap algorithm
- http://planning.cs.uiuc.edu/node298.html
- properties: 'very difficult to implement in code'
- domain: 'Computational algebraic geometry'
- solves: 'Motion planning'

## Rapidly-exploring random tree
- also called: 'RRT'
- report: 'Rapidly-exploring random trees: A new tool for path planning' (1998)
- https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree
- solves: 'Motion planning'
- implemented in (libraries): 'PythonRobotics', 'ompl::geometric::RRT'
- domain: 'Robotics'
- properties: 'single-query', 'geometric'

## Probabilistic roadmap
- also called: 'PRM'
- paper: 'Probabilistic roadmaps for path planning in high-dimensional configuration spaces' (1996) <https://doi.org/10.1109/70.508439>
- https://en.wikipedia.org/wiki/Probabilistic_roadmap
- solves: 'Motion planning'
- implemented in (libraries): 'PythonRobotics', 'ompl::geometric::PRM'
- domain: 'Robotics'
- properties: 'multi-query', 'geometric'

## LazyPRM
- variant of: 'Probabilistic roadmap'
- implemented in (libraries): 'ompl::geometric::LazyPRM'
- properties: 'multi-query', 'geometric'

## Fan and Su algortihm for multiple pattern match
- paper: 'An efficient algorithm for matching multiple patterns' (1993)
- "combines the concept of deterministic finite state automata (DFSA) and Boyer-Moore’s algorithm"

## Hu and Shing algortihm for matrix chain products
- paper: 'Computation of Matrix Chain Products. Part I' (1982)
- paper: 'Computation of Matrix Chain Products. Part II' (1984)
- solves: Matrix chain multiplication'
- time complexity: O(n log n) where n is the number of matrices

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

## Harvey and Van Der Hoeven algorithm
- paper: 'Integer multiplication in time O(n log n)' (2019) <https://hal.archives-ouvertes.fr/hal-02070778>
- is a: 'multiplication algorithm'
- applications: 'integer multiplication'
- complexity: O(n log n)
- impractical, large constants

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

## Grubbs's test for outliers
- also called: 'Maximum normalized residual test', 'Extreme studentized deviate test'
- paper: 'Sample Criteria for Testing Outlying Observations' (1950) <https://doi.org/10.1214/aoms/1177729885>
- https://en.wikipedia.org/wiki/Grubbs%27s_test_for_outliers
- applications: 'Outlier detection'

## Generalized ESD test
- also called: 'Generalized extreme Studentized deviate test'
- paper: 'Percentage Points for a Generalized ESD Many-Outlier Procedure' (1983) <https://doi.org/10.2307/1268549>
- https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/esd.htm
- https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
- implemented in (libraries): 'PyAstronomy.pyasl.generalizedESD', 'adtk.detector.GeneralizedESDTestAD', 'PyAstronomy.pyasl.pointDistGESD'
- applications: 'Outlier detection'
- domain: 'Statistics'

## STL decomposition
- paper: 'STL: A Seasonal-Trend Decomposition Procedure Based on Loess' (1990) <>
- implemented in (libraries): 'R stl', 'Python adtk.transformer.STLDecomposition', 'statsmodels.tsa.seasonal.STL'
- applications: 'Decomposition of time series'

## Baxter-King bandpass filter
- paper: 'Measuring Business Cycles Approximate Band-Pass Filters for Economic Time Series' (1995) <https://doi.org/10.3386/w5022>
- applications: 'Time series analysis'
- implemented in (libraries): 'statsmodels.tsa.filters.bk_filter.bkfilter', 'R mFilter/bkfilter'

## Kwiatkowski-Phillips-Schmidt-Shin test
- also called: 'KPSS test', 'Kwiatkowski-Phillips-Schmidt-Shin test for stationarity'
- paper: 'Testing the null hypothesis of stationarity against the alternative of a unit root' (1992) <https://doi.org/10.1016/0304-4076(92)90104-Y>
- https://en.wikipedia.org/wiki/KPSS_test
- applications: 'Time series analysis', 'Trend stationary testing'
- implemented in (libraries): 'statsmodels.tsa.stattools.kpss'
- domain: 'Econometrics'

## Savitzky–Golay filter
- also called: 'LOESS', 'Locally estimated scatterplot smoothing'
- paper: 'Smoothing and Differentiation of Data by Simplified Least Squares Procedures' (1964) <https://doi.org/10.1021/ac60214a047>
- rediscovery paper: 'Robust Locally Weighted Regression and Smoothing Scatterplots' (1979) <https://doi.org/10.2307/2286407>
- https://en.wikipedia.org/wiki/Local_regression
- https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
- implemented in: 'scipy.signal.savgol_filter'
- properties: 'Nonparametric'
- applications: 'Smoothing'

## LOWESS
- paper: 'LOWESS: A Program for Smoothing Scatterplots by Robust Locally Weighted Regression' (1981) <https://doi.org/10.2307/2683591>
- https://en.wikipedia.org/wiki/Local_regression
- also called: 'Locally weighted scatterplot smoothing'
- properties: 'Nonparametric'
- based on: 'Savitzky–Golay filter'
- implemented in: 'statsmodels.nonparametric.smoothers_lowess.lowess'
- applications: 'Smoothing'

## Synchronized overlap-add method
- also called: 'Time-domain harmonic scaling', 'SOLA'
- paper: 'High quality time-scale modification for speech' (1985) <https://doi.org/10.1109/ICASSP.1985.1168381>
- solves: 'Audio time stretching and pitch scaling'

## PSOLA
- also called: 'Pitch Synchronous Overlap and Add'
- paper: 'Diphone synthesis using an overlap-add technique for speech waveforms concatenation' (1986) <https://doi.org/10.1109/ICASSP.1986.1168657>
- https://en.wikipedia.org/wiki/PSOLA
- based on: 'Short-time Fourier transform'
- solves: 'Audio time stretching and pitch scaling'

## TD-PSOLA
- also called: 'Time-domain pitch-synchronous overlap-and-add'
- paper: 'A diphone synthesis system based on time-domain prosodic modifications of speech' (1989) <https://doi.org/10.1109/ICASSP.1989.266409>
- based on: 'PSOLA'
- implemented in: 'diguo2046/psola'
- solves: 'Audio time stretching and pitch scaling'

## ESOLA
- also called: 'Epoch-Synchronous Overlap-Add'
- paper: 'Epoch-Synchronous Overlap-Add (ESOLA) for Time- and Pitch-Scale Modification of Speech Signals' (2018) <https://arxiv.org/abs/1801.06492>
- solves: 'Audio time stretching and pitch scaling'

## Phase vocoder
- https://en.wikipedia.org/wiki/Phase_vocoder
- paper: 'Phase vocoder' (1966) <https://doi.org/10.1002/j.1538-7305.1966.tb01706.x>
- uses: 'Short-time Fourier transform'
- solves: 'Audio time stretching and pitch scaling'

## De Boor's algorithm
- https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
- domain: 'numerical analysis'
- spline curves in B-spline
- properties: 'numerically stable'
- implemented in: 'Python scipy.interpolate.BSpline'
- generalization of: 'De Casteljau's algorithm'

## Lucas–Kanade method
- paper: 'An iterative image registration technique with an application to stereo vision' (1981)
- https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method
- applications: 'Optical flow estimation', 'aperture problem'
- implemented in: 'opencv::calcOpticalFlowPyrLK', 'opencv::CalcOpticalFlowLK' (obsolete)
- uses: 'Structure tensor'
- properties: 'local', 'sparse'

## Horn–Schunck method
- paper: 'Determining Optical Flow' (1981)
- https://en.wikipedia.org/wiki/Horn%E2%80%93Schunck_method
- applications: 'Optical flow estimation', 'aperture problem'
- implemented in: 'opencv::CalcOpticalFlowHS' (obsolete, should be replaced with calcOpticalFlowPyrLK or calcOpticalFlowFarneback according to opencv docs)
- properties: 'global'

## Gunnar-Farneback algorithm
- paper: 'Two-frame motion estimation based on polynomial expansion'
- applications: 'Optical flow estimation'
- implemented in: 'opencv::calcOpticalFlowFarneback'
- properties: 'dense'

## SimpleFlow algorithm
- paper: 'SimpleFlow: A Non-iterative, Sublinear Optical Flow Algorithm' (2012)
- implemented in: 'opencv::calcOpticalFlowSF'
- applications: 'Optical flow estimation'

## TV-L1
- original paper: 'A Duality Based Approach for Realtime TV-L1 Optical Flow' (2007) <https://doi.org/10.1007/978-3-540-74936-3_22>
- applications: 'Optical flow estimation'
- properties: 'dense'

## Improved TV-L1
- paper: 'An Improved Algorithm for TV-L1 Optical Flow' (2009) <https://doi.org/10.1007/978-3-642-03061-1_2>
- implementation paper: 'TV-L1 Optical Flow Estimation' (2013) <https://doi.org/10.5201/ipol.2013.26>
- improved variant of: 'TV-L1'
- applications: 'Optical flow estimation'
- implemented in (libraries): 'cv::DualTVL1OpticalFlow', 'skimage.registration.optical_flow_tvl1'
- properties: 'dense'

## Kadane's algorithm
- https://en.wikipedia.org/wiki/Maximum_subarray_problem#Kadane's_algorithm
- applications: 'Maximum subarray problem'
- time complexity: O(n)
- uses method: 'Dynamic programming'

## Scale-invariant feature transform
- paper: 'Object recognition from local scale-invariant features' (1999)
- https://en.wikipedia.org/wiki/Scale-invariant_feature_transform
- is a: 'feature detection algorithm'
- applications: 'object recognition', 'robotic mapping and navigation', 'image stitching', '3D modeling', 'gesture recognition', 'video tracking'
- patent: US6711293

## Marching cubes
- https://en.wikipedia.org/wiki/Marching_cubes
- paper: 'Marching cubes: A high resolution 3D surface construction algorithm' (1987) <https://doi.org/10.1145/37402.37422>
- domain: 'Computer graphics'
- applications: 'Medical imaging', 'Mesh generation'
- implemented in: 'skimage.measure.marching_cubes_classic'

## Lewiner's marching cubes
- paper: 'Efficient Implementation of Marching Cubes' Cases with Topological Guarantees' (2003) <https://doi.org/10.1080/10867651.2003.10487582>
- improved variant of: 'Marching cubes'
- implemented in: 'skimage.measure.marching_cubes_lewiner'
- domain: 'Computer graphics'

## Naive calculation of image moments
- solves special case of: 'Image moments'
- implemented in: 'skimage.measure.moments'

## Naive calculation of Hu moment invariants
- solves special case of: 'Image moments'
- implemented in: 'skimage.measure.moments_hu'

## Marching squares
- https://en.wikipedia.org/wiki/Marching_squares
- domain: 'Computer graphics', 'Cartography'
- applications: 'contour finding'
- properties: 'Embarrassingly parallel'
- implemented in (libraries): 'skimage.measure.find_contours'
- special case of: 'Marching cubes'

## Lempel–Ziv–Welch
- https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch
- applications: 'Lossless compression'
- was patented

## General number field sieve
- also called: 'GNFS'
- https://en.wikipedia.org/wiki/General_number_field_sieve
- applications: 'Integer factorization'
- implemented in (applications): 'msieve', 'GGNFS'

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
- paper: 'An efficient antialiasing technique' (1991)
- https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm
- is a: Line drawing algorithm, Anti-aliasing algorithm
- domain: computer graphics
- applications: antialiasing
- input: 'Start and end points'
- output: 'List of points with associated graylevel'

## Needleman–Wunsch algorithm
- original paper: 'A general method applicable to the search for similarities in the amino acid sequence of two proteins' (1970)
- https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
- applications: 'Sequence alignment' (global), 'Computer stereo vision'
- time complexity: O(m n)
- uses method: 'Dynamic programming'
- domain: 'bioinformatics'
- implemented in: 'EMBOSS', 'Python Bio.pairwise2.align.globalxx'
- input: 'two random access collections'
- output: 'Optimal global alignment'

## Smith–Waterman algorithm
- original paper: 'Identification of common molecular subsequences' (1981)
- https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
- applications: 'Sequence alignment' (local)
- time complexity: O(m n)
- uses method: 'Dynamic programming'
- domain: 'bioinformatics'
- implemented in: 'EMBOSS', 'Python Bio.pairwise2.align.localxx'
- input: 'two random access collections'
- output: 'Optimal local alignment'

## Hirschberg's algorithm
- paper: 'A linear space algorithm for computing maximal common subsequences' (1975)
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
- paper: 'An experimental comparison of min-cut/max- flow algorithms for energy minimization in vision' (2004)
- implemented in: 'Python networkx.algorithms.flow.boykov_kolmogorov', 'boost::graph::boykov_kolmogorov_max_flow'

## Stoer and Wagner's minimum cut algorithm
- paper: 'A Simple Min-Cut Algorithm (1994)'
- input: 'Connected graph'
- output: 'Minimum cut'
- implemented in: 'boost::graph::stoer_wagner_min_cut'
- domain: 'Graph theory'

## Ford–Fulkerson method
- paper: 'Maximal Flow Through a Network' (1956)
- https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm
- https://brilliant.org/wiki/ford-fulkerson-algorithm/
- properties: 'greedy', 'incomplete'
- solves: 'Maximum flow problem'
- implemented by: 'Edmonds–Karp algorithm'
- input: 'Flow network'
- output: 'Maximum flow'

## Dinic's algorithm
- also called: 'Dinitz's algorithm'
- paper: 'Algorithm for Solution of a Problem of Maximum Flow in a Network with Power Estimation' (1970) <>
- https://en.wikipedia.org/wiki/Dinic%27s_algorithm
- https://cp-algorithms.com/graph/dinic.html
- solves: 'Maximum flow problem'
- similar: 'Edmonds–Karp algorithm'
- runtime complexity: O(v e log(v)) (with dynamic trees)
- runtime complexity: O(v^2 e) (with dynamic trees)

## MPM algorithm
- also called: 'Malhotra, Pramodh-Kumar and Maheshwari algorithm'
- paper: 'An O(|V|3) algorithm for finding maximum flows in networks' (1978) <https://doi.org/10.1016/0020-0190(78)90016-9>
- https://cp-algorithms.com/graph/mpm.html
- solves: 'Maximum flow problem'
- input: 'acyclic flow network'
- runtime complexity: O(v^3)

## Edmonds–Karp algorithm
- paper: 'Theoretical Improvements in Algorithmic Efficiency for Network Flow Problems' (1972)
- https://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm
- https://brilliant.org/wiki/edmonds-karp-algorithm/
- implements: 'Ford–Fulkerson method'
- implemented in: 'Python networkx.algorithms.flow.edmonds_karp', 'boost::graph::edmonds_karp_max_flow'
- time complexity: O(v e^2) [or O(v^2 e)?] where v is the number of vertices and e the number of edges
- solves: 'Maximum flow problem'
- input: 'Flow network'
- output: 'Maximum flow'
- book: 'MIT Press', 'Introduction to Algorithms'
- similar: 'Dinic's algorithm'

## Marr–Hildreth algorithm
- https://en.wikipedia.org/wiki/Marr%E2%80%93Hildreth_algorithm
- paper: 'Theory of edge detection (1980)'
- applications: 'Edge detection'
- domain: 'image processing'
- input: 'Grayscale image'
- output: 'Binary image'

## Otsu's method
- paper: 'A Threshold Selection Method from Gray-Level Histograms' (1979) <https://doi.org/10.1109/TSMC.1979.4310076>
- https://en.wikipedia.org/wiki/Otsu%27s_method
- solves: 'Image binarization'
- domain: 'Image processing'
- implemented in: 'cv::threshold(type=THRESH_OTSU)', 'skimage.filters.threshold_otsu'
- input: 'Grayscale image'
- output: 'Binary image'
- properties: 'global'

## Otsu's method (local variant)
- solves: 'Image binarization'
- domain: 'Image processing'
- implemented in (libraries): 'skimage.filters.rank.otsu'
- input: 'Grayscale image'
- output: 'Binary image'
- properties: 'local'
- variant of: 'Otsu's method'

## Soundex
- https://en.wikipedia.org/wiki/Soundex
- is a: 'Phonetic algorithm'
- applications: 'Indexing', 'Phonetic encoding'

## Match rating approach
- https://en.wikipedia.org/wiki/Match_rating_approach
- is a: 'Phonetic algorithm'
- applications: 'Indexing', 'Phonetic encoding', 'Phonetic comparison'
- is a: 'Similarity measure' # should this go to metrics-and-distances.md?

## Chamfer matching
- paper: 'Parametric correspondence and chamfer matching: two new techniques for image matching'
- uses: 'Chamfer distance'

## Work stealing algorithm
- https://en.wikipedia.org/wiki/Work_stealing
- applications: 'Scheduling'
- implemented in: 'Cilk'

## Louvain algorithm
- paper: 'Fast unfolding of communities in large networks' (2008)
- https://en.wikipedia.org/wiki/Louvain_modularity#Algorithm
- applications: 'Community detection'
- domain: 'Network theory'

## Leiden algorithm
- paper: 'From Louvain to Leiden: guaranteeing well-connected communities' (2018)
- created as improvement of: 'Louvain algorithm'
- applications: 'Community detection'
- domain: 'Network theory'

## k-means clustering
- https://en.wikipedia.org/wiki/K-means_clustering
- is a: 'Clustering algorithm'
- implemented in: 'sklearn.cluster.KMeans, scipy.cluster.vq.kmeans, Bio.Cluster.kcluster', 'cv::kmeans'
- partitions space into: 'Voronoi cells'
- applications: 'Vector quantization', 'Cluster analysis', 'Feature learning'
- input: 'Collection of points' & 'Positive integer k'
- output: 'Collection of cluster indices'

## PQk-means
- also called: 'Product-quantized k-means'
- paper: 'PQk-means: Billion-scale Clustering for Product-quantized Codes' (2017)
- implemented in (libraries): 'Python pqkmeans'
- approximation of: 'k-means clustering'

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
- implemented in (libraries): 'C++ samhocevar/lolremez'

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

## Embedded Zerotrees of Wavelet transforms
- also called: 'EZW'
- paper: 'Embedded image coding using zerotrees of wavelet coefficients' (1993)
- https://en.wikipedia.org/wiki/Embedded_Zerotrees_of_Wavelet_transforms#See_also
- applications: 'Lossy image compression'
- involves subproblems: 'Discrete wavelet transform', 'Zerotree coding', 'Arithmetic coding'

## Set partitioning in hierarchical trees
- also called: 'SPIHT'
- paper: 'A new, fast, and efficient image codec based on set partitioning in hierarchical trees' (1996)
- https://en.wikipedia.org/wiki/Set_partitioning_in_hierarchical_trees
- applications: 'Lossy image compression'
- involves subproblem: 'Discrete wavelet transform'
- superseeds: 'Embedded Zerotrees of Wavelet transforms'
- related evaluation metric: 'PSNR'

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

## Fiorio's algorithm for linear time restricted Union-Find
- paper: 'Two linear time Union-Find strategies for image processing' (1996) <https://doi.org/10.1016/0304-3975(94)00262-2>
- implemented in: 'Python skimage.measure.label'
- solves special case of: 'Disjoint set union problem'
- properties: 'two pass'
- solves: 'Image segmentation'

## Improved variant of Fiorio's algorithm for connected-component labeling
- paper: 'Optimizing connected component labeling algorithms' (2005) <https://doi.org/10.1117/12.596105>
- based on: 'Fiorio's algorithm for linear time restricted Union-Find'
- implemented in (libraries): 'skimage.measure.label'
- solves: 'Connected-component labeling'

## Wu's algorithm for connected-component labeling
- paper: 'Two Strategies to Speed up Connected Component Labeling Algorithms (2005)'
- solves: 'Connected-component labeling'
- implemented in: 'opencv::connectedComponents'

## Fortune's algorithm
- paper: 'A sweepline algorithm for Voronoi diagrams' (1986)
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
- paper: 'Algorithms for Reporting and Counting Geometric Intersections' (1979)
- https://en.wikipedia.org/wiki/Bentley%E2%80%93Ottmann_algorithm
- is a: 'Sweep line algorithm'
- solves: 'Line segment intersection'
- input: 'set of line segments'
- output: 'crossings'
- based on: 'Shamos–Hoey algorithm'
- implemented in: 'ideasman42/isect_segments-bentley_ottmann'

## Freeman-Shapira's minimum bounding box
- paper: 'Determining the minimum-area encasing rectangle for an arbitrary closed curve' (1975)
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
- paper: 'The quickhull algorithm for convex hulls' (1996) (modern version)
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

## Gilbert–Johnson–Keerthi distance algorithm
- also called: 'GJK algorithm'
- paper: 'A fast procedure for computing the distance between complex objects in three-dimensional space' (1988) <https://doi.org/10.1109/56.2083>
- https://en.wikipedia.org/wiki/Gilbert%E2%80%93Johnson%E2%80%93Keerthi_distance_algorithm
- applications: 'Convex collision detection', 'Realtime physics'
- uses: 'Johnson's distance subalgorithm'
- domains: 'Computational geometry', 'Convex geometry'

## Lin-Canny Closest Features Method
- paper: 'A fast algorithm for incremental distance calculation' (1991) <https://doi.org/10.1109/ROBOT.1991.131723>
- applications: 'Collision detection'
- domains: 'Computational geometry'

## H-Walk
- paper: 'H-Walk: hierarchical distance computation for moving convex bodies' (1999) <https://doi.org/10.1145/304893.304979>
- applications: 'Collision detection'
- domains: 'Computational geometry'

## Sequence step algorithm
- https://en.wikipedia.org/wiki/Sequence_step_algorithm
- https://www.planopedia.com/sequence-step-algorithm/
- applications: 'Scheduling'

## Fast folding algorithm
- https://en.wikipedia.org/wiki/Fast_folding_algorithm
- paper: 'Fast folding algorithm for detection of periodic pulse trains' (1969)
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
- paper: 'Decentralized extrema-finding in circular configurations of processors' (1980)
- solves: 'Leader election problem'
- properties: 'distributed'

## Gale–Church alignment algorithm
- https://en.wikipedia.org/wiki/Gale%E2%80%93Church_alignment_algorithm
- paper: 'A Program for Aligning Sentences in Bilingual Corpora' (1993)
- domain: 'Computational linguistics'
- applications: 'Sentence alignment'
- input: 'pair of list of sentences'

## Beier–Neely morphing algorithm
- paper: 'Feature-based image metamorphosis' (1992)
- https://en.wikipedia.org/wiki/Beier%E2%80%93Neely_morphing_algorithm
- applications: 'Image processing', 'Image morphing'
- domain: 'Computer graphics'
- input: 'pair of images'
- implemented in: 'JS blendmaster/reexpress'

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
- article: 'Hilltop: A Search Engine based on Expert Documents'
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
- implemented in (libraries): 'networkx.algorithms.flow.min_cost_flow', 'C++ lemon::NetworkSimplex'

## Expectation–maximization algorithm
- also called: 'EM algorithm'
- paper: 'Maximum Likelihood from Incomplete Data via the EM Algorithm' (1977)
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
- applications: 'Speech recognition', 'Cryptanalysis', 'Copy-number variation'
- implemented in (libraries): 'GHMM'

## WINEPI
- https://en.wikipedia.org/wiki/WINEPI
- applications: 'Data mining', 'Time series analysis', 'Association rule learning'

## Bach's algorithm
- paper: 'How to generate factored random numbers' (1988)
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
- properties: 'vectorized'
- uses: 'Fast Fourier transform'

## Principal component analysis
- also called: 'PCA', 'Karhunen–Loève transform', 'KLT', 'Hotelling transform'
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
- paper: 'A fast algorithm for proving terminating hypergeometric identities' (1990)
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
- paper: 'A New Version of the Euclidean Algorithm' (1963)
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

## Graphical lasso # ?model or algorithm?
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
- paper: 'Reset Sequences for Monotonic Automata' (1990)
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
- paper: 'Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography' (1981)
- applications: 'Computer vision', 'Location determination problem'
- implemented in: 'skimage.measure.ransac'

## PACBO
- also called: 'Probably Approximately Correct Bayesian Online'
- paper: 'PAC-Bayesian Online Clustering' (2016)
- implemented in: 'R PACBO'
- related: 'RJMCMC'
- applications: 'online clustering'
- domain: 'Game theory', 'Computational learning theory'

## RJMCMC
- also called: 'Reversible-jump Markov chain Monte Carlo'
- paper: 'Reversible jump Markov chain Monte Carlo computation and Bayesian model determination' (1995)
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
- paper: 'Radix Sorting With No Extra Space' (2007)
- applications: 'Integer sorting'

## Approximate Link state algorithm
- also called: 'XL'
- paper: 'XL: An Efficient Network Routing Algorithm' (2008)
- applications: 'Network routing'

## Nicholl–Lee–Nicholl algorithm
- https://en.wikipedia.org/wiki/Nicholl%E2%80%93Lee%E2%80%93Nicholl_algorithm
- applications: 'Line clipping'

## Liang–Barsky algorithm
- https://en.wikipedia.org/wiki/Liang%E2%80%93Barsky_algorithm
- applications: 'Line clipping'

## C3 linearization
- paper: 'A Monotonic Superclass Linearization for Dylan' (1996)
- https://en.wikipedia.org/wiki/C3_linearization
- applications: 'Multiple inheritance', 'Method Resolution Order'
- used to implement application: 'Python', 'Perl'

## Sethi–Ullman algorithm
- paper: 'The Generation of Optimal Code for Arithmetic Expressions' (1970)
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
- paper: 'Reshuffling scale-free networks: From random to assortative' (2004)
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
- paper: 'The accuracy of the clock synchronization achieved by TEMPO in Berkeley UNIX 4.3BSD' (1989)
- https://en.wikipedia.org/wiki/Berkeley_algorithm
- applications: 'Clock synchronization'
- is a: 'Distributed algorithm'

## Cristian's algorithm
- paper: 'Probabilistic clock synchronization (1989)'
- https://en.wikipedia.org/wiki/Cristian%27s_algorithm
- applications: 'Clock synchronization'
- is a: 'Distributed algorithm'

## Marzullo's algorithm
- thesis: 'Maintaining the time in a distributed system: an example of a loosely-coupled distributed service' (1984)
- https://en.wikipedia.org/wiki/Marzullo%27s_algorithm
- superseded by: 'Intersection algorithm'
- is a: 'Agreement algorithm'
- applications: 'Clock synchronization'

## Intersection algorithm
- also called: 'Clock Select Algorithm'
- paper: 'Improved algorithms for synchronizing computer network clocks' (1995)
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
- paper: 'Algorithm for Producing Rankings Based on Expert Surveys' (2019)
- based on: 'Segmented string relative ranking'
- compare: 'PageRank'
- applications: 'Link analysis'

## SALSA algorithm
- also called: 'Stochastic Approach for Link-Structure Analysis'
- paper: 'SALSA: the stochastic approach for link-structure analysis' (2000)
- https://en.wikipedia.org/wiki/SALSA_algorithm
- applications: 'Link analysis'

## TextRank
- paper: 'TextRank: Bringing Order into Texts' (2004)
- based on: 'PageRank'
- domain: 'Graph Theory'
- applications: 'Keyword extraction', 'Text summarization'

## HITS algorithm
- also called: 'Hyperlink-Induced Topic Search'
- paper: 'Authoritative sources in a hyperlinked environment' (1999)
- https://en.wikipedia.org/wiki/HITS_algorithm
- applications: 'Link analysis', 'Search engines', 'Citation analysis'

## Eigenfactor
- paper: 'Eigenfactor: Measuring the value and prestige of scholarly journals' (2007)
- https://en.wikipedia.org/wiki/Eigenfactor
- is a: 'Citation metric'

## Impact factor
- https://en.wikipedia.org/wiki/Impact_factor
- is a: 'Citation metric'

## PageRank
- paper: 'The PageRank Citation Ranking: Bringing Order to the Web' (1999)
- https://en.wikipedia.org/wiki/PageRank
- domain: 'Graph theory'
- applications: 'Link analysis', 'Linear algebra'
- input: 'Google matrix'

## CheiRank
- https://en.wikipedia.org/wiki/CheiRank
- input: 'Google matrix'
- domain: 'Graph theory', 'Linear algebra'

## ObjectRank
- paper: 'ObjectRank: Authority-Based Keyword Search in Databases' (2004)
- applications: 'Ranking in graphs'

## PathSim
- paper: 'PathSim: Meta Path-Based Top-K Similarity Search in Heterogeneous Information Networks' (2011)
- applications: 'Similarity search', 'Ranking in graphs'

## RankDex
- also called: 'Hyperlink Vector Voting method', 'HVV'
- paper: 'Toward a qualitative search engine' (1998)

## Banker's algorithm
- paper: 'Een algorithme ter voorkoming van de dodelijke omarming (EWD-108') (1964–1967)
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
- implemented in: 'imblearn.over_sampling.SMOTENC'

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
- paper: 'Rapid solution of problems by quantum computation' (1992)
- https://en.wikipedia.org/wiki/Deutsch%E2%80%93Jozsa_algorithm
- is a: 'Quantum algorithm'
- properties: 'deterministic'

## Grover's algorithm
- paper: 'A fast quantum mechanical algorithm for database search' (1996)
- https://en.wikipedia.org/wiki/Grover%27s_algorithm
- is a: 'Quantum algorithm'
- properties: 'probabilistic', 'asymptotically optimal'

## Maximally stable extremal regions
- paper: 'Robust wide baseline stereo from maximally stable extremal regions' (2002)
- https://en.wikipedia.org/wiki/Maximally_stable_extremal_regions
- applications: 'Computer vision'
- implemented in: 'OpenCV::MSER'

## Mean shift
- original paper: 'The estimation of the gradient of a density function, with applications in pattern recognition' (1975) <https://doi.org/10.1109/TIT.1975.1055330>
- rediscovery paper: 'Mean shift, mode seeking, and clustering' (1995) <https://doi.org/10.1109/34.400568>
- https://en.wikipedia.org/wiki/Mean_shift
- is a: 'mode-seeking algorithm'
- applications: 'Cluster analysis', 'visual tracking', 'image smoothing'
- basis for: 'Camshift'

## Mean shift for feature space analysis
- also called: 'Adaptive mean shift clustering'
- variant of: 'Mean shift'
- paper: 'Mean shift: a robust approach toward feature space analysis' (2002) <https://doi.org/10.1109/34.1000236>
- implemented in: 'sklearn.cluster.MeanShift'
- is a: adaptive gradient ascent method
- properties: 'centroid based'
- applications: 'Clustering'
- domain: 'Computer vision'

## Variational Bayes
- also called: 'VB'
- https://en.wikipedia.org/wiki/Variational_Bayesian_methods
- applications: 'Statistical inference'
- properties: 'deterministic'

## Gibbs sampling
- https://en.wikipedia.org/wiki/Gibbs_sampling
- type of: 'Markov chain Monte Carlo'
- applications: 'Sampling', 'Statistical inference'
- properties: 'randomized'

## Collapsed Gibbs sampling
- https://en.wikipedia.org/wiki/Gibbs_sampling#Collapsed_Gibbs_sampler
- variant of: 'Gibbs sampling'
- samples (examples): 'Latent Dirichlet allocation'
- properties: 'randomized'

## Hamiltonian Monte Carlo
- also called: 'HMC', 'Hybrid Monte Carlo'
- https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo
- paper: 'Hybrid Monte Carlo' (1987)
- is a: 'Markov chain Monte Carlo algorithm'
- solves: 'Sampling'
- input: 'probability distribution'
- output: 'random samples'
- applications: 'Lattice QCD'
- implemented in: 'Stan'

## No-U-Turn Sampler
- also called: 'NUTS'
- paper: 'The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo' (2011)
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
- paper: 'An Ensemble Adjustment Kalman Filter for Data Assimilation' (2001)

## PF-PMC-PHD
- also called: 'Particle Filter–Pairwise Markov Chain–Probability Hypothesis Density'
- paper: 'Particle Probability Hypothesis Density Filter Based on Pairwise Markov Chains' (2019)
- applications: 'multi-target tracking system'

## Multichannel affine projection algorithm
- paper: 'A multichannel affine projection algorithm with applications to multichannel acoustic echo cancellation' (1996)
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
- patent: 'US2950048'
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

## Line integral convolution
- paper: 'Imaging vector fields using line integral convolution' (1993) <https://doi.org/10.1145/166117.166151>
- https://en.wikipedia.org/wiki/Line_integral_convolution#Algorithm
- implemented in: 'lime::LIC'
- domain: 'Scientific visualization'
- visualizes: 'vector field'

## Meyer's flooding algorithm
- https://en.wikipedia.org/wiki/Watershed_(image_processing)#Meyer's_flooding_algorithm
- solves: 'Watershed transformation'
- input: 'Grayscale image'
- applications: 'Image segmentation'
- superseded by: 'Priority-Flood'

## Priority-Flood
- paper: 'Priority-flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models' (2014) <https://doi.org/10.1016/j.cageo.2013.04.024>
- solves: 'Watershed transformation'
- input: 'Grayscale image'
- applications: 'Image segmentation'

## SKImage's watershed algorithm
- implemented in: 'skimage.segmentation.watershed'
- solves: 'Watershed transformation'
- input: 'Grayscale image'
- applications: 'Image segmentation'

## Active contour model
- paper: 'Snakes: Active contour models' (1988) <https://doi.org/10.1007/BF00133570>
- implemented in: 'skimage.segmentation.active_contour'
- applications: 'Image segmentation'

## Chan-Vese segmentation algorithm
- paper: 'An Active Contour Model without Edges' (1999) <https://doi.org/10.1007/3-540-48236-9_13>
- implemented in: 'skimage.segmentation.chan_vese'
- applications: 'Image segmentation'

## Morphological geodesic active contours
- also called: 'MorphGAC'
- paper: 'A Morphological Approach to Curvature-Based Evolution of Curves and Surfaces' (2013) <https://doi.org/10.1109/TPAMI.2013.106>
- implemented in: 'skimage.segmentation.morphological_geodesic_active_contour'

## Morphological Active Contours without Edges
- also called: 'MorphACWE'
- paper: 'A Morphological Approach to Curvature-Based Evolution of Curves and Surfaces' (2013) <https://doi.org/10.1109/TPAMI.2013.106>
- implemented in: 'skimage.segmentation.morphological_chan_vese'

## Anisotropic diffusion
- also called: 'Perona–Malik diffusion'
- paper: 'Scale-space and edge detection using anisotropic diffusion' (1990) <https://doi.org/10.1109/34.56205>
- https://en.wikipedia.org/wiki/Anisotropic_diffusion
- applications: 'Noise reduction', 'Edge-preserving smoothing'
- implemented in: 'cv::ximgproc::anisotropicDiffusion'

## Reich's edge-preserving filter
- also called: 'EPF'
- paper: 'A Real-Time Edge-Preserving Denoising Filter' (2018) <https://doi.org/10.5220/0006509000850094>
- implemented in: 'cv::ximgproc::edgePreservingFilter'

## Bilateral filter
- paper: 'Bilateral filtering for gray and color images' (1998) <https://doi.org/10.1109/ICCV.1998.710815>
- https://en.wikipedia.org/wiki/Bilateral_filter
- applications: 'Noise reduction', 'Edge-preserving smoothing'
- implemented in: 'skimage.restoration.denoise_bilateral', 'Avisynth TBilateral'
- properties: 'non-linear'

## SPTWO
- original paper: 'Patch-Based Video Denoising With Optical Flow Estimation' (2016) <https://doi.org/10.1109/TIP.2016.2551639>
- implementation paper: 'Video Denoising with Optical Flow Estimation' (2018) <https://doi.org/10.5201/ipol.2018.224>
- applications: 'Video denoising'
- properties: 'patch based'

## Video non-local Bayes
- also called: 'VNLB'
- http://dev.ipol.im/~pariasm/video_nlbayes/
- paper: 'Video Denoising via Empirical Bayesian Estimation of Space-Time Patches' (2018) <https://doi.org/10.1007/s10851-017-0742-4>
- applications: 'Video denoising'
- properties: 'patch based'
- implemented in: 'pariasm/vnlb'

## Pei-Lin normalization
- paper: 'Image normalization for pattern recognition' (1995) <https://doi.org/10.1016/0262-8856(95)98753-G>
- implemented in: 'cv.ximgproc.PeiLinNormalization'

## Structured forests edge detection
- paper: 'Structured Forests for Fast Edge Detection' (2013) <https://doi.org/10.1109/ICCV.2013.231>
- applications: 'Edge detection'
- implemented in: 'cv::ximgproc::StructuredEdgeDetection'

## Zhang-Suen thinning algorithm
- paper: 'A fast parallel algorithm for thinning digital patterns' (1984) <https://doi.org/10.1145/357994.358023>
- implemented in: 'cv::ximgproc::thinning'
- input: 'Binary image'
- output: 'Binary image'

## Edge Boxes
- paper: 'Edge Boxes: Locating Object Proposals from Edges' (2014) <https://doi.org/10.1007/978-3-319-10602-1_26>
- implemented in (libraries): 'cv::ximgproc::EdgeBoxes'
- applications: 'Object detection'
- domain: 'Computer vision'

## Contrast Limited Adaptive Histogram Equalization
- also called: 'CLAHE'
- book: 'Morgan Kaufmann', 'Graphics Gems IV' (1994) <https://doi.org/10.1016/C2013-0-07360-4>
- https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE
- implemented in (libraries): 'skimage.exposure.equalize_adapthist'

## Niblack's binarization
- book: 'Prentice Hall', 'An introduction to digital image processing' (1986)
- implemented in: 'cv::ximgproc::niBlackThreshold', 'skimage.filters.threshold_niblack'
- solves: 'Image binarization'
- domain: 'Image processing'
- properties: 'local'

## Sauvola binarization
- paper: 'Adaptive document image binarization' (2000) <https://doi.org/10.1016/S0031-3203(99)00055-2>
- based on: 'Niblack's binarization'
- implemented in: 'skimage.filters.threshold_sauvola'
- solves: 'Image binarization'
- domain: 'Image processing'
- properties: 'local'

## Phansalkar binarization
- paper: 'Adaptive local thresholding for detection of nuclei in diversity stained cytology images' (2011) <https://doi.org/10.1109/ICCSP.2011.5739305>
- based on: 'Sauvola binarization'
- solves: 'Image binarization'
- domain: 'Image processing'
- properties: 'local'

## Bernsen binarization
- paper: 'Dynamic Thresholding of Grey-Level Images' (1986) <>
- solves: 'Image binarization'
- domain: 'Image processing'
- properties: 'local'

## ISODATA thresholding
- also called: 'Ridler-Calvard method', 'Inter-means'
- paper: 'Picture Thresholding Using an Iterative Selection Method' (1978) <https://doi.org/10.1109/TSMC.1978.4310039>
- solves: 'Image binarization'
- domain: 'Image processing'
- properties: 'histogram-based', 'global'
- implemented in (libraries): 'skimage.filters.threshold_isodata'
- implamented in (applications): 'ImageJ'

## Li's iterative minimum cross entropy method
- paper: 'An iterative algorithm for minimum cross entropy thresholding' (1998) <https://doi.org/10.1016/S0167-8655(98)00057-9>
- solves: 'Image binarization'
- domain: 'Image processing'
- implemented in (libraries): 'skimage.filters.threshold_li'
- properties: 'global'

## Mean thresholding
- paper: 'An Analysis of Histogram-Based Thresholding Algorithms' (1993) <https://doi.org/10.1006/cgip.1993.1040>
- solves: 'Image binarization'
- domain: 'Image processing'
- implemented in (libraries): 'skimage.filters.threshold_mean'
- properties: 'global'

## Minimum thresholding
- paper: 'The Analysis of Cell Images' (1966) <https://doi.org/10.1111/j.1749-6632.1965.tb11715.x>
- solves: 'Image binarization'
- domain: 'Image processing'
- implemented in (libraries): 'skimage.filters.threshold_minimum'
- properties: 'global'

## Triangle thresholding
- paper: 'Automatic measurement of sister chromatid exchange frequency' (1977) <https://doi.org/10.1177/25.7.70454>
- solves: 'Image binarization'
- domain: 'Image processing'
- implemented in: 'skimage.filters.threshold_triangle'
- properties: 'global'

## Yen's thresholding method
- paper: 'A new criterion for automatic multilevel thresholding' (1995) <https://doi.org/10.1109/83.366472>
- solves: 'Image binarization'
- domain: 'Image processing'
- implemented in: 'skimage.filters.threshold_yen'
- properties: 'global'

## Felzenszwalb-Huttenlocher
- paper: 'Efficient graph-based image segmentation' (2004) <https://doi.org/10.1023/B:VISI.0000022288.19776.77>
- applications: 'Image segmentation'
- implemented in: 'skimage.segmentation.felzenszwalb'

## Quickshift
- paper: 'Quick Shift and Kernel Methods for Mode Seeking' (2008) <https://doi.org/10.1007/978-3-540-88693-8_52>
- implemented in: 'skimage.segmentation.quickshift'
- applications: 'Image segmentation'

## Simple linear iterative clustering
- also called: 'SLIC'
- paper: 'SLIC Superpixels Compared to State-of-the-Art Superpixel Methods' (2012) <https://doi.org/10.1109/TPAMI.2012.120>
- is a: 'superpixel algorithm'
- implemented in: 'skimage.segmentation.slic', 'cv::ximgproc::SuperpixelSLIC'
- uses: 'k-means clustering'

## Linear spectral clustering
- also called: 'LSC'
- paper: 'Superpixel segmentation using Linear Spectral Clustering' (2015) <https://doi.org/10.1109/CVPR.2015.7298741>
- implemented in: 'cv::ximgproc::SuperpixelLSC'

## SEEDS
- also called: 'Superpixels extracted via energy-driven sampling'
- paper: 'SEEDS: Superpixels Extracted Via Energy-Driven Sampling' (2015) <https://doi.org/10.1007/s11263-014-0744-2>
- implemented in: 'cv::ximgproc::SuperpixelSEEDS'

## Random walker algorithm
- paper: 'Random Walks for Image Segmentation' (2006) <https://doi.org/10.1109/TPAMI.2006.233>
- https://en.wikipedia.org/wiki/Random_walker_algorithm
- applications: 'Image segmentation'
- solves: 'Combinatorial Dirichlet problem'
- implemented in: 'skimage.segmentation.random_walker'

## Difference of Gaussians
- also called: 'DoG'
- https://en.wikipedia.org/wiki/Difference_of_Gaussians
- domain: 'Imaging science'
- implemented in (libraries): 'lime::edgeDoG', 'skimage.feature.blob_dog'
- is a: 'Band-pass filter'
- applications: 'Edge detection', 'Blob detection'

## Determinant of Hessian method
- also called: 'DoH'
- https://en.wikipedia.org/wiki/Blob_detection#The_determinant_of_the_Hessian
- implemented in (libraries): 'skimage.feature.blob_doh'
- applications: 'Blob detection'

## Laplacian of Gaussian
- also called: 'LoG'
- https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian
- applications: 'Blob detection'
- implemented in (libraries): 'skimage.feature.blob_log'

## Difference of Box
- also called: 'DoB'

## Horn's algorithm
- paper: 'Determining lightness from an image' (1974) <https://doi.org/10.1016/0146-664X(74)90022-7>
- implemented in: 'lime::colorConstancy(tpye=CONSTANCY_HORN)'
- applications: 'Color constancy'

## Rahman's algorithm
- patent: 'Method of improving a digital image' (1999) <US5991456A>
- implemented in: 'lime::colorConstancy(tpye=CONSTANCY_RAHMAN)'
- applications: 'Color constancy'
- properties: 'patent expired'

## Faugeras's algorithm
- paper: 'Digital color image processing within the framework of a human visual model' (1979) <https://doi.org/10.1109/TASSP.1979.1163262>
- implemented in: 'lime::colorConstancy(tpye=CONSTANCY_FAUGERAS)'
- applications: 'Color constancy'

## Image inpainting by biharmonic functions
- paper: 'On surface completion and image inpainting by biharmonic functions: Numerical aspects' (2017) <https://arxiv.org/abs/1707.06567>
- implemented in: 'skimage.restoration.inpaint_biharmonic'
- solves: 'Inpainting'

## Canny edge detector
- also called: 'Hysteresis thresholding'
- paper: 'A Computational Approach to Edge Detection' (1986) <https://doi.org/10.1109/TPAMI.1986.4767851>
- https://en.wikipedia.org/wiki/Canny_edge_detector
- domain: 'image processing'
- applications: 'Edge detection'
- implemented in (libraries): 'cv::Canny', 'skimage.feature.canny'
- uses: 'Hysteresis thresholding'

## Moravec corner detection algorithm
- https://en.wikipedia.org/wiki/Corner_detection#Moravec_corner_detection_algorithm
- thesis: 'Obstacle avoidance and navigation in the real world by a seeing robot rover' (1980) <https://www.ri.cmu.edu/publications/obstacle-avoidance-and-navigation-in-the-real-world-by-a-seeing-robot-rover/>
- implemented in (libraries): 'skimage.feature.corner_moravec'
- applications: 'Corner detection'

## Förstner corner detector
- paper: 'A Fast Operator for Detection and Precise Location of Distinct Points, Corners and Centers of Circular Features' (1987) <>
- https://en.wikipedia.org/wiki/Corner_detection#The_F%C3%B6rstner_corner_detector
- implemented in (libraries): 'skimage.feature.corner_foerstner'
- applications: 'Corner detection'

## Harris-Stephens corner detector
- paper: 'A combined corner and edge detector' (1988) <>
- https://en.wikipedia.org/wiki/Corner_detection#The_Harris_&_Stephens_/_Plessey_/_Shi%E2%80%93Tomasi_corner_detection_algorithms
- applications: 'Corner detection'
- implemented in (libraries): 'skimage.feature.corner_harris', 'cv::cornerHarris'

## Shi-Tomasi corner detector
- also called: 'Kanade-Tomasi corner detector'
- paper: 'Good features to track' (1994) <https://doi.org/10.1109/CVPR.1994.323794>
- https://en.wikipedia.org/wiki/Corner_detection#The_Harris_&_Stephens_/_Plessey_/_Shi%E2%80%93Tomasi_corner_detection_algorithms
- implemented in (libraries): 'skimage.feature.corner_shi_tomasi', 'cv::goodFeaturesToTrack'
- based on: 'Harris-Stephens corner detector'

## Trajkovic-Hedley corner detector
- paper: 'Fast corner detection' (1998) <https://doi.org/10.1016/S0262-8856(97)00056-5>
- https://en.wikipedia.org/wiki/Corner_detection#The_Trajkovic_and_Hedley_corner_detector
- applications: 'Corner detection'
- domain: 'Computer vision'

## Kitchen-Rosenfeld corner detector
- paper: 'Gray-level corner detection' (1982) <https://doi.org/10.1016/0167-8655(82)90020-4>
- implemented in (libraries): 'skimage.feature.corner_kitchen_rosenfeld'
- applications: 'Corner detection'
- domain: 'Computer vision'

## Features from accelerated segment test
- also called: 'FAST'
- paper: 'Machine Learning for High-Speed Corner Detection' (2006) <https://doi.org/10.1007/11744023_34>
- https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test
- implemented in (libraries): 'skimage.feature.corner_fast'
- applications: 'Corner detection'

## DAISY descriptor
- paper: 'A fast local descriptor for dense matching' (2008) <https://doi.org/10.1109/CVPR.2008.4587673>
- paper: 'DAISY: An Efficient Dense Descriptor Applied to Wide-Baseline Stereo' (2009) <https://doi.org/10.1109/TPAMI.2009.77>
- implemented in (libraries): 'skimage.feature.daisy'
- properties: 'local'
- applications: 'Feature detection'

## Haar-like feature
- paper: 'Rapid object detection using a boosted cascade of simple features' (2001) <https://doi.org/10.1109/CVPR.2001.990517>
- https://scikit-image.org/docs/dev/api/skimage.feature.html#ra19a7aed16ca-1
- implemented in (libraries): 'skimage.feature.haar_like_feature'
- domain: 'Computer vision'
- applications: 'Feature detection'

## Histogram of oriented gradients
- also called: 'HOG'
- patent: 'Method of and apparatus for pattern recognition' (1982) <https://patents.google.com/patent/US4567610>
- https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
- domain: 'Computer vision'
- implemented in (libraries): 'skimage.feature.hog'
- applications: 'Feature detection'
- properties: 'patent expired'

## Scale-invariant feature transform
- also called: 'SIFT'
- patent: 'Method and apparatus for identifying scale invariant features in an image and use of same for locating an object in an image' (2000) <https://patents.google.com/patent/US6711293>
- https://en.wikipedia.org/wiki/Scale-invariant_feature_transform
- applications: 'Feature detection'
- domain: 'Computer vision'

## GLOH
- also called: 'Gradient Location and Orientation Histogram'
- paper: 'A performance evaluation of local descriptors' (2005) <https://doi.org/10.1109/TPAMI.2005.188>
- https://en.wikipedia.org/wiki/GLOH
- applications: 'Feature detection'
- domain: 'Computer vision'

## Local binary patterns
- also called: 'LBP'
- paper: 'A comparative study of texture measures with classification based on featured distributions' (1996) <https://doi.org/10.1016/0031-3203(95)00067-4>
- https://en.wikipedia.org/wiki/Local_binary_patterns
- http://www.scholarpedia.org/article/Local_Binary_Patterns
- implemented in (libraries): 'skimage.feature.local_binary_pattern', 'mahotas.features.lbp.lbp'
- applications: 'Feature detection'
- domain: 'Computer vision'

## Multi-block local binary pattern
- also called: 'MB-LBP'
- paper: 'Face Detection Based on Multi-Block LBP Representation' (2007) <https://doi.org/10.1007/978-3-540-74549-5_2>
- implemented in (libraries): 'skimage.feature.multiblock_lbp'
- applications: 'Feature detection'
- domain: 'Computer vision'

## Speeded up robust features
- also called: 'SURF'
- patent: 'Robust interest point detector and descriptor ' (2006) <https://patents.google.com/patent/US20090238460A1/en>
- https://en.wikipedia.org/wiki/Speeded_up_robust_features
- applications: 'Feature detection'
- domain: 'Computer vision'
- implemented in (libraries): 'mahotas.features.surf.surf'

## Binary Robust Independent Elementary Features
- also called: 'BRIEF'
- paper: 'BRIEF: Binary Robust Independent Elementary Features' (2010) <https://doi.org/10.1007/978-3-642-15561-1_56>
- implemented in (libraries): 'skimage.feature.BRIEF'
- applications: 'Feature detection'
- domain: 'Computer vision'

## ORB
- also called: 'Oriented FAST and rotated BRIEF'
- paper: 'ORB: An efficient alternative to SIFT or SURF' (2011) <https://doi.org/10.1109/ICCV.2011.6126544>
- implemented in (libraries): 'skimage.feature.ORB'
- applications: 'Feature detection'
- domain: 'Computer vision'
- based on: 'Features from accelerated segment test', 'Binary Robust Independent Elementary Features'

## CenSurE keypoint detector
- paper: 'CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching' (2008) <https://doi.org/10.1007/978-3-540-88693-8_8>
- implemented in (libraries): 'skimage.feature.CENSURE'
- applications: 'Feature detection'
- domain: 'Computer vision'

## Fast normalized cross-correlation
- solves: 'Normalized cross-correlation'
- paper: 'Fast Normalized Cross-Correlation' (1995) <>
- implemented in: 'Matlab normxcorr2'

## Fast normalized cross-correlation
- solves: 'Normalized cross-correlation'
- paper: 'Template matching using fast normalized cross correlation' (2001) <https://doi.org/10.1117/12.421129>

## Image registration by cross-correlation
- paper: 'Efficient subpixel image registration algorithms' (2008) <https://doi.org/10.1364/OL.33.000156>
- implemented in: 'skimage.feature.register_translation'
- applications: 'Image registration'
- domain: 'Computer vision'

## Image registration by masked normalized cross-correlation
- paper: 'Masked Object Registration in the Fourier Domain' (2011) <https://doi.org/10.1109/TIP.2011.2181402>
- implemented in: 'skimage.feature.masked_register_translation'
- applications: 'Image registration'
- domain: 'Computer vision'
- uses: 'Masked normalized cross-correlation'

## Template Matching using Fast Normalized Cross Correlation
- paper: 'Template matching using fast normalized cross correlation' (2001) <https://doi.org/10.1117/12.421129>
- solves: 'Template matching'
- domain: 'Computer vision'
- implemented in: 'skimage.feature.match_template'

## Normalized cuts
- also called: 'Normalized graph cuts'
- paper: 'Normalized cuts and image segmentation' (2000) <https://doi.org/10.1109/34.868688>
- https://en.wikipedia.org/wiki/Segmentation-based_object_categorization#Normalized_cuts
- implemented in: 'skimage.future.graph.cut_normalized'
- applications: 'Image segmentation', 'Medical imaging'
- domain: 'Computer vision', 'Graph theory'
- properties: 'block based', 'region based'

## ncut algorithm
- https://en.wikipedia.org/wiki/Segmentation-based_object_categorization#The_ncut_algorithm
- paper: 'Normalized cuts and image segmentation' (2000) <https://doi.org/10.1109/34.868688>
- implemented in (libraries): 'skimage.future.graph.ncut'
- domain: 'Computer vision'
- applications: 'Image segmentation'

## OBJ CUT
- paper: 'OBJ CUT' (2005) <https://doi.org/10.1109/CVPR.2005.249>
- https://en.wikipedia.org/wiki/Segmentation-based_object_categorization#OBJ_CUT
- domain: 'Computer vision'
- applications: 'Image segmentation'

## LOBPCG
- also called: 'Locally Optimal Block Preconditioned Conjugate Gradient'
- properties: 'Matrix-free'
- solves (partly): 'Generalized eigenvalue problem'
- implemented in (libraries): 'scipy.sparse.linalg.lobpcg'

## Hysteresis thresholding
- paper: 'A Computational Approach to Edge Detection' (1986) <https://doi.org/10.1109/TPAMI.1986.4767851>
- implemented in (libraries): 'skimage.filters.apply_hysteresis_threshold'
- domain: 'Image processing'

## Wave Function Collapse algorithm
- also called: 'WFC algorithm'
- related papers: 'WaveFunctionCollapse is constraint solving in the wild' (2007) <https://doi.org/10.1145/3102071.3110566>
- implemented in (libraries): 'mxgmn/WaveFunctionCollapse'
- applications: 'Locally similar bitmap generation', 'Level generation', 'image generation'

## Fuse algorithm
- http://draves.org/fuse/
- applications: 'texture synthesis', 'image fusion', 'associative image reconstruction'
- type: 'Full neighbourhood search'
- domain: 'Computer graphics'

## Efros--Leung texture synthesis algorithm
- paper: 'Texture synthesis by non-parametric sampling' (1999) <https://doi.org/10.1109/ICCV.1999.790383>
- applications: 'texture synthesis'
- type: 'Full neighbourhood search'
- domain: 'Computer graphics'

## Wei-Levoy texture synthesis algorithm
- also called: 'WL algorithm'
- paper: 'Fast texture synthesis using tree-structured vector quantization' (2000) <https://doi.org/10.1145/344779.345009>
- applications: 'texture synthesis'
- type: 'Full neighbourhood search'
- domain: 'Computer graphics'

## Ashikhmin texture synthesis algorithm
- paper: 'Synthesizing natural textures' (2001) <https://doi.org/10.1145/364338.364405>
- based on: 'Wei-Levoy texture synthesis algorithm'
- applications: 'texture synthesis'
- domain: 'Computer graphics'

## k-coherent search
- paper: 'Synthesis of bidirectional texture functions on arbitrary surfaces' (2002) <https://doi.org/10.1145/566654.566634>
- applications: 'texture synthesis'
- domain: 'Computer graphics'
- uses: 'bidirectional texture function'

## Resynthesis algorithm
- thesis: 'Image Texture Tools: Texture Synthesis, Texture Transfer, and Plausible Restoration' by 'Paul Francis Harrison' (2005)
- applications: 'texture synthesis'
- domain: 'Computer graphics'

## Winnow algorithm
- https://en.wikipedia.org/wiki/Winnow_(algorithm)
- paper: 'Learning Quickly When Irrelevant Attributes Abound: A New Linear-Threshold Algorithm' (1988) <https://doi.org/10.1023/A:1022869011914>
- domain: 'Machine learning'
- solves: 'Binary classification'
- is a: 'Linear classifier'
- type: 'Online learning'

## Manacher's algorithm
- paper: 'A New Linear-Time "On-Line" Algorithm for Finding the Smallest Initial Palindrome of a String' (1975) <https://doi.org/10.1145/321892.321896>
- https://en.wikipedia.org/wiki/Longest_palindromic_substring#Manacher's_algorithm
- solves: 'Longest palindromic substring problem'
- runtime complexity: O(n)

## Jeuring's algorithm
- paper: 'The derivation of on-line algorithms, with an application to finding palindromes' (1994) <https://doi.org/10.1007/BF01182773>
- solves: 'Longest palindromic substring problem'
- runtime complexity: O(n)

## Takaoka's multiset permutations algorithm
- paper: 'A Two-level Algorithm for Generating Multiset Permutations' (2009)
- solves: 'Multiset permutation'

# Filters

## Discrete Gaussian kernel
- https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel
- also called: 'Gaussian blur', 'Gaussian smoothing'
- is a: 'Low-pass filter'
- implemented in: 'scipy.ndimage.gaussian_filter', 'skimage.filters.gaussian'

## Laplace filter
- also called: 'Discrete Laplace operator', 'Laplacian matrix'
- https://en.wikipedia.org/wiki/Discrete_Laplace_operator
- implemented in: 'scipy.ndimage.laplace', 'skimage.filters.laplace'

## Median filter
- https://en.wikipedia.org/wiki/Median_filter
- implemented in: 'scipy.ndimage.median_filter', 'skimage.filters.median', 'skimage.filters.rank.median'
- domain: 'Image processing'
- properties: 'non-separable'
- applications: 'Salt-and-pepper noise reduction', 'Speckle reduction'

## Lulu smoothing
- paper: 'Idempotent one-sided approximation of median smoothers' (1989) <https://doi.org/10.1016/0021-9045(89)90017-8>
- https://en.wikipedia.org/wiki/Lulu_smoothing
- properties: 'idempotent'
- applications: 'Time series smoothing'

## Meijering neuriteness filter
- paper: 'Design and validation of a tool for neurite tracing and analysis in fluorescence microscopy images' (2004) <https://doi.org/10.1002/cyto.a.20022>
- implemented in: 'skimage.filters.meijering'
- applications: 'Neurite tracing'

## Kirsch operator
- paper: 'Computer determination of the constituent structure of biological images' (1971) <https://doi.org/10.1016/0010-4809(71)90034-6>
- https://en.wikipedia.org/wiki/Kirsch_operator
- applications: 'Edge detection'
- properties: 'non-linear'

## Roberts cross operator
- thesis: 'Machine perception of three-dimensional solids' (1963) <http://hdl.handle.net/1721.1/11589>
- https://en.wikipedia.org/wiki/Roberts_cross
- implemented in: 'skimage.filters.roberts'
- domain: 'Image processing'
- applications: 'Edge detection'
- is a: 'difference operator'

## Sato tubeness filter
- paper: 'Three-dimensional multi-scale line filter for segmentation and visualization of curvilinear structures in medical images' (1998) <https://doi.org/10.1016/S1361-8415(98)80009-1>
- implemented in: 'skimage.filters.sato'
- domain: 'Image processing', 'Medical imaging'

## Wiener filter
- https://en.wikipedia.org/wiki/Wiener_filter
- implemented in: 'scipy.signal.wiener'
- domain: 'Signal processing'
- applications: 'System identification', 'Deconvolution', 'Noise reduction', 'Signal detection'
- is a: 'Linear filter'

## Gabor filter
- https://en.wikipedia.org/wiki/Gabor_filter
- is a: 'Linear filter'
- domain: 'Image processing'
- applications: 'localize and extract text-only regions', 'facial expression recognition', 'pattern analysis'
- implemented in: 'skimage.filters.gabor', 'skimage.filters.gabor_kernel'

## Frangi vesselness filter
- paper: 'Multiscale vessel enhancement filtering' (1998) <https://doi.org/10.1007/BFb0056195>
- implemented in: 'skimage.filters.frangi'

## Sobel operator
- https://en.wikipedia.org/wiki/Sobel_operator
- is a: 'Discrete differentiation operator'
- domain: 'Image processing'
- applications: 'Edge detection'
- implemented in: 'scipy.ndimage.sobel', 'skimage.filters.sobel'

## Prewitt operator
- https://en.wikipedia.org/wiki/Prewitt_operator
- implemented in: 'scipy.ndimage.prewitt', 'skimage.filters.prewitt'
- applications: 'Edge detection'
- domain: 'Image processing'
- is a: 'Discrete differentiation operator'

## Scharr operator
- thesis: 'Optimale Operatoren in der Digitalen Bildverarbeitung' (2000) <http://doi.org/10.11588/heidok.00000962>
- domain: 'Image processing'
- applications: 'Edge detection'
- implemented in: 'skimage.filters.scharr'
- better rotational invariance than 'Sobel' or 'Prewitt'

## Kuwahara filter
- https://en.wikipedia.org/wiki/Kuwahara_filter
- https://reference.wolfram.com/language/ref/KuwaharaFilter.html
- is a: 'Non-linear filter'
- applications: 'adaptive noise reduction', 'medical imaging', 'fine-art photography'
- disadvantages: 'create block artifacts'
- domain: 'Image processing'
- implemented in: 'lime::kuwaharaFilter'

## Erosion
- https://en.wikipedia.org/wiki/Erosion_(morphology)
- implemented in: 'lime::morphFilter(type=MORPH_ERODE)', 'cv::erode'
- domain: 'Mathematical morphology'
- input: 'Binary image'
- output: 'Binary image'

## Dilation
- https://en.wikipedia.org/wiki/Dilation_(morphology)
- implemented in: 'lime::morphFilter(type=MORPH_DILATE)', 'cv::dilate'
- domain: 'Mathematical morphology'
- input: 'Binary image'
- output: 'Binary image'

## Opening
- https://en.wikipedia.org/wiki/Opening_(morphology)
- implemented in: 'lime::morphFilter(type=MORPH_OPEN)', 'cv::morphologyEx(op=MORPH_OPEN)'
- domain: 'Mathematical morphology'
- input: 'Binary image'
- output: 'Binary image'
- applications: 'Noise reduction'

## Closing
- https://en.wikipedia.org/wiki/Closing_(morphology)
- implemented in: 'lime::morphFilter(type=MORPH_CLOSE)', 'cv::morphologyEx(op=MORPH_CLOSE)'
- domain: 'Mathematical morphology'
- input: 'Binary image'
- output: 'Binary image'
- applications: 'Noise reduction'

# Methods, patterns and programming models

## Map
- also called: 'parallel for loop'
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
