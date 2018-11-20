# Problems

## Biclustering
- https://en.wikipedia.org/wiki/Biclustering

## Covariance matrix estimation
- https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices

## Polynomial factorization
- http://mathworld.wolfram.com/PolynomialFactorization.html
- https://en.wikipedia.org/wiki/Factorization_of_polynomials
- domain: 'Algebra'

## Singular value decomposition
- https://en.wikipedia.org/wiki/Singular_value_decomposition
- implemented in: 'scipy.linalg.svd'
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

## Maximal matching
- https://en.wikipedia.org/wiki/Matching_(graph_theory)#Maximal_matchings
- https://brilliant.org/wiki/matching/#definitions-and-terminology

## Maximum matching
- the 'Maximal matching' with the maximum number of edges
- https://brilliant.org/wiki/matching/#definitions-and-terminology
- solves: 'Assignment problem' on 'weighted bipartite graphs'

## Stable marriage problem
- https://en.wikipedia.org/wiki/Stable_marriage_problem
- is a: 'matching problem'

# Algorithms and methods

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
- paper: 'Co-clustering documents and words using Bipartite Spectral Graph Partitioning (2001)'
- solves: 'Biclustering'
- implemented in: 'sklearn.cluster.bicluster.SpectralCoclustering'
- input: 'Matrix'

## Spectral biclustering (Kluger, 2003
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

## Canonical-correlation analysis
- also called: 'CCA'
- https://en.wikipedia.org/wiki/Canonical_correlation
- implemented in: 'Python sklearn.cross_decomposition.CCA', 'SPSS', 'R cancor'
- domain: 'Multivariate statistics'

## Minimum Covariance Determinant
- also called: 'MCD'
- paper: 'Least Median of Squares Regression (1984)'
- improvement: 'Fast Minimum Covariance Determinant'

## Fast Minimum Covariance Determinant
- also called: 'FAST-MCD'
- paper: 'A Fast Algorithm for the Minimum Covariance Determinant Estimator (1998)'
- implemented in: 'sklearn.covariance.MinCovDet'
- input: 'Normal distributed data'
- robust to outliers
- solves: 'Covariance matrix estimation'

## Sample covariance matrix
- https://en.wikipedia.org/wiki/Sample_mean_and_covariance
- sensitive to outliers
- solves: 'Covariance matrix estimation'

## Maximum likelihood covariance estimator
- https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices#Maximum-likelihood_estimation_for_the_multivariate_normal_distribution
- implemented in: 'sklearn.covariance.EmpiricalCovariance'
- solves: 'Covariance matrix estimation'

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

## Runge–Kutta methods
- https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
- applications: 'Numerical analysis', 'Ordinary differential equation'
- implemented in: 'scipy.integrate.RK45, scipy.integrate.RK23'

## Newton–Cotes formulas
- https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas
- applications: 'Numerical integration'
- implemented in: 'scipy.integrate.newton_cotes'

## Kabsch algorithm
- https://en.wikipedia.org/wiki/Kabsch_algorithm
- input: 'two paired sets of points'
- output: 'optimal rotation matrix' (according to 'Root-mean-square deviation')

## Jacobi eigenvalue algorithm
- https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
- is a: 'Iterative method'
- applications: 'Matrix diagonalization'
- input: 'real symmetric matrix'
- output: 'eigenvalues and eigenvectors'

## Finite difference: central difference
- https://en.wikipedia.org/wiki/Finite_difference#Forward,_backward,_and_central_differences
- implemented in: 'scipy.misc.derivative'

## Cholesky algorithm
- https://en.wikipedia.org/wiki/Cholesky_decomposition#The_Cholesky_algorithm
- input: 'Hermitian, positive-definite matrix'
- output: 'Cholesky decomposition'

## Cholesky–Banachiewicz algorithm
- variation of: 'Cholesky algorithm'

## Cholesky–Crout algorithm
- variation of: 'Cholesky algorithm'

## Romberg's method
- https://en.wikipedia.org/wiki/Romberg%27s_method
- applications: 'Numerical integration'
- implemented in: 'scipy.integrate.romberg'
- input: 'Function and integration boundaries'
- output: 'Integrated value'

## Knuth's Simpath algorithm
- https://en.wikipedia.org/wiki/Knuth%27s_Simpath_algorithm
- domain: 'Graph theory'
- input: 'Graph'
- output: 'zero-suppressed decision diagram (ZDD) representing all simple paths between two vertices'

## Zeilberger's Algorithm
- http://mathworld.wolfram.com/ZeilbergersAlgorithm.html
- input: 'Terminating Hypergeometric Identities of a certain form'
- output: 'Polynomial recurrence'

## Gosper's algorithm
- https://en.wikipedia.org/wiki/Gosper%27s_algorithm
- http://mathworld.wolfram.com/GospersAlgorithm.html
- input: 'hypergeometric terms'
- output: 'sums that are themselves hypergeometric terms'

## PSOS Algorithm
- also called: 'Partial sum of squares'
- http://mathworld.wolfram.com/PSOSAlgorithm.html
- is a: 'Integer relation algorithm'

## Ferguson-Forcade Algorithm
- http://mathworld.wolfram.com/Ferguson-ForcadeAlgorithm.html
- is a: 'Integer relation algorithm'
- generalization of: 'Euclidean algorithm'

## HJLS Algorithm
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

## PSLQ Algorithm
- http://mathworld.wolfram.com/PSLQAlgorithm.html
- is a: 'Integer relation algorithm'

## Fleury's Algorithm
- https://en.wikipedia.org/wiki/Eulerian_path#Fleury's_algorithm
- http://mathworld.wolfram.com/FleurysAlgorithm.html
- input: 'Graph'
- output: 'Eulerian cycle' or 'Eulerian trail'

## Blankinship Algorithm
- http://mathworld.wolfram.com/BlankinshipAlgorithm.html

## Splitting Algorithm
- http://mathworld.wolfram.com/SplittingAlgorithm.html

## Recursive Monotone Stable Quadrature
- http://mathworld.wolfram.com/RecursiveMonotoneStableQuadrature.html

## Lin's Method
- http://mathworld.wolfram.com/LinsMethod.html

## Hungarian Maximum Matching algorithm
- https://en.wikipedia.org/wiki/Hungarian_algorithm
- http://mathworld.wolfram.com/HungarianMaximumMatchingAlgorithm.html
- https://brilliant.org/wiki/hungarian-matching/
- also called: 'Kuhn-Munkres algorithm'
- input: 'bipartite graph'
- output: 'maximum-weight matching'
- time complexity: O(V^3) for V vertices

## Hopcroft–Karp algorithm
- paper: 'An $n^{5/2}$ Algorithm for Maximum Matchings in Bipartite Graphs (1973)'
- https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm
- https://brilliant.org/wiki/hopcroft-karp/
- input: 'Bipartite graph'
- output: 'Maximum matching'
- time complexity: O(E sqrt(V)) for E edges and V vertices
- best for sparse, for dense, there are more recent improvements like 'Alt et al. (1991)'

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

## Miller's Algorithm
- http://mathworld.wolfram.com/MillersAlgorithm.html
- https://crypto.stanford.edu/pbc/notes/ep/miller.html
- domain: 'Cryptography'
