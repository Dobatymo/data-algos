# Books
- 'Clever Algorithms: Nature-Inspired Programming Recipes'

# Problems

## Unconstrained nonlinear optimization without derivatives
- https://en.wikipedia.org/wiki/Derivative-free_optimization
- https://en.wikipedia.org/wiki/Nonlinear_programming
- domain: 'Numerical optimization'
- mainly used for: 'functions which are not continuous or differentiable'
- also called: 'black box', 'direct search'
- domain: 'Mathematical optimization'

## Unconstrained nonlinear optimization with first derivative
- https://en.wikipedia.org/wiki/Nonlinear_programming

## Constrained nonlinear optimization
- https://en.wikipedia.org/wiki/Nonlinear_programming

## Convex optimization
- https://en.wikipedia.org/wiki/Convex_optimization

## Nonlinear dimensionality reduction
- https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction

# Numerical optimization algorithms

## Powell's method
- https://en.wikipedia.org/wiki/Powell%27s_method
- paper: 'An efficient method for finding the minimum of a function of several variables without calculating derivatives'
- type: 'Unconstrained nonlinear without derivatives'
- is a: 'Optimization method'
- implemented in: 'Python scipy.optimize.minimize(method="Powell")' (modified variant)

## Luus–Jaakola
- https://en.wikipedia.org/wiki/Luus–Jaakola
- paper: 'Optimization by direct search and systematic reduction of the size of search region'
- is a: 'Heuristic'
- type: 'Unconstrained nonlinear without derivatives'
- applied to: 'global real valued optimization'
- becomes 'Iterative method' for 'twice continuously differentiable functions', however for this problem 'Newton's method' is usually used
- applications: 'optimal control', 'transformer design', 'metallurgical processes', 'chemical engineering'
- recommended for: 'neither convex nor differentiable nor locally Lipschitz type functions'
- implemented in: 'Python osol.extremum'

## Pattern search
- https://en.wikipedia.org/wiki/Pattern_search_(optimization)
- paper: '"Direct Search" Solution of Numerical and Statistical Problems'
- type: 'Unconstrained nonlinear without derivatives'
- is a: 'Optimization method'
- is a: 'Heuristic' or 'Iterative method', depending on the class of function WHICH CLASSES?

## Golden-section search
- https://en.wikipedia.org/wiki/Golden-section_search
- paper: 'Sequential minimax search for a maximum'
- used for: 'strictly unimodal functions'
- related: 'Fibonacci search'
- type: 'Unconstrained nonlinear without derivatives'

## Successive parabolic interpolation
- https://en.wikipedia.org/wiki/Successive_parabolic_interpolation
- paper: 'An iterative method for locating turning points (1966)'
- used for: 'continuous unimodal functions'
- type: 'Unconstrained nonlinear without derivatives'
- global or local convergence is not guaranteed

## Line search
- https://en.wikipedia.org/wiki/Line_search
- https://www.mathworks.com/help/optim/ug/unconstrained-nonlinear-optimization-algorithms.html#f15468
- general strategy only, employed by many optimization methods

## Differential evolution
- https://en.wikipedia.org/wiki/Differential_evolution
- paper: 'Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces'
- type: 'Unconstrained nonlinear without derivatives'
- implemented in: 'Mathematica NMinimize(Method->"DifferentialEvolution")', 'scipy.optimize.differential_evolution'
- is a: 'Metaheuristic', 'Stochastic algorithm'

## Random search
- https://en.wikipedia.org/wiki/Random_search
- paper: 'The convergence of the random search method in the extremal control of a many parameter system'
- is a: 'Randomized algorithm'
- implemented in: 'Mathematica NMinimize(Method->"RandomSearch")'
- type: 'Unconstrained nonlinear without derivatives'
- related: 'Random optimization'

## Random optimization
- https://en.wikipedia.org/wiki/Random_optimization
- paper: 'Random optimization'
- type: 'Unconstrained nonlinear without derivatives'
- is a: 'Randomized algorithm'
- related: 'Random search'

## Nelder–Mead method
- https://en.wikipedia.org/wiki/Nelder–Mead_method
- http://mathworld.wolfram.com/Nelder-MeadMethod.html
- http://www.scholarpedia.org/article/Nelder-Mead_algorithm
- paper: 'A simplex method for function minimization'
- type: 'Unconstrained nonlinear without derivatives'
- implemented in: 'Mathematica NMinimize(Method->"NelderMead")', 'Python scipy.optimize.minimize(method="Nelder-Mead")'
- is a: "heuristic method"

## Genetic algorithm
- https://en.wikipedia.org/wiki/Genetic_algorithm
- book: 'Adaptation in natural and artificial systems (1975)'
- type: 'Unconstrained nonlinear without derivatives'
- is a: 'Metaheuristic', 'Randomized search method'

## Principal Axis Method
- https://reference.wolfram.com/language/tutorial/UnconstrainedOptimizationPrincipalAxisMethod.html
- also called: 'PRincipal Axis (PRAXIS) algorithm', 'Brent's algorithm'
- book: 'Algorithms for Minimization without Derivatives (1972/1973)'
- uses: 'SVD', 'Adaptive coordinate descent'
- implemented in: 'Mathematica FindMinimum(Method->"PrincipalAxis")'

## Newton's method
- https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
- type: 'Unconstrained nonlinear optimization with Hessian'
- implemented in: 'Mathematica FindMinimum(Method->"Newton")'
- is a: 'Anytime algorithm'

## Broyden–Fletcher–Goldfarb–Shanno algorithm
- https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm
- is a: 'Quasi-Newton method', 'Iterative method'
- implemented in: 'Python scipy.optimize.minimize(method="BFGS")', 'Mathematica FindMinimum(Method->"QuasiNewton")'
- type: 'Unconstrained nonlinear with first derivative'

## Limited-memory BFGS
- https://en.wikipedia.org/wiki/Limited-memory_BFGS
- is a: 'Quasi-Newton method'
- variant of: 'BFGS' (for large systems with memory optimizations)
- implemented in: 'Mathematica FindMinimum(Method->"QuasiNewton")', 'Python scipy.minimize(method="L-BFGS-B")' (L-BFGS-B variant)
- applications: 'Maximum entropy models', 'CRF'

## Basin-hopping algorithm
- https://en.wikipedia.org/wiki/Basin-hopping
- paper: 'Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms'
- implemented in: 'scipy.optimize.basinhopping'
- global optimization of a smooth scalar function of one or more variables
- applications: 'Physical Chemistry'
- works good if there are lots of local extrema
- is a: 'stochastic algorithm', 'sampling algorithm'
- type: 'Unconstrained nonlinear without derivatives'
- depends on any: 'local optimization algorithm'

## Levenberg–Marquardt algorithm
- https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
- http://mathworld.wolfram.com/Levenberg-MarquardtMethod.html
- paper: 'A Method for the Solution of Certain Non-Linear Problems in Least Squares (1944)'
- type: 'Unconstrained nonlinear with first derivative'
- solves: 'Non-linear least squares', 'Orthogonal non-linear least squares'
- is a: 'iterative procedure'
- applications: 'generic curve-fitting problems'
- finds local minimum
- implemented in: 'Mathematica FindMinimum(Method->"LevenbergMarquardt")', 'Python scipy.optimize.least_squares(method="lm"), scipy.optimize.root(method="lm")', 'MINPACK', 'R ONLS'
- variant of: 'Gauss–Newton'

## Davidon–Fletcher–Powell formula
- https://en.wikipedia.org/wiki/Davidon%E2%80%93Fletcher%E2%80%93Powell_formula
- type: 'Unconstrained nonlinear with first derivative'
- is a: 'Quasi-Newton method'
- superseded by: 'Broyden–Fletcher–Goldfarb–Shanno algorithm'

## Symmetric rank-one
- https://en.wikipedia.org/wiki/Symmetric_rank-one
- also called: 'SR1'
- is a: 'Quasi-Newton method'
- advantages for sparse or partially separable problems
- implemented in: 'scipy.optimize.SR1'
- type: 'Unconstrained nonlinear with first derivative'

## Berndt–Hall–Hall–Hausman algorithm
- also called: 'BHHH'
- https://en.wikipedia.org/wiki/Berndt%E2%80%93Hall%E2%80%93Hall%E2%80%93Hausman_algorithm
- type: 'Unconstrained nonlinear with first derivative'
- similar: 'Gauss–Newton algorithm'

## Nonlinear conjugate gradient method
- https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
- type: 'Unconstrained nonlinear with first derivative'
- implemented in: 'Mathematica FindMinimum(Method->"ConjugateGradient")'
- implemented in: 'Python scipy.optimize.minimize(method="Newton-CG")'

## Truncated Newton method
- https://en.wikipedia.org/wiki/Truncated_Newton_method
- type: 'Unconstrained nonlinear with first derivative'
- implemented in: 'Python scipy.optimize.minimize(method="TNC")'

## Gauss–Newton algorithm
- https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
- applications: 'Non-linear least squares'
- type: 'Unconstrained nonlinear with first derivative'

## Trust region method
- https://en.wikipedia.org/wiki/Trust_region
- https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods
- http://www.applied-mathematics.net/optimization/optimizationIntro.html
- type: 'Unconstrained nonlinear with first derivative'

## Gradient descent
- https://en.wikipedia.org/wiki/Gradient_descent
- type: 'Unconstrained nonlinear with first derivative'
- also called: 'Steepest descent'
- will converge to a global minimum if the function is convex
- stochastic approximation: 'Stochastic gradient descent'
- applications: 'Convex optimization'

## Stochastic gradient descent
- also called: 'SGD'
- https://en.wikipedia.org/wiki/Stochastic_gradient_descent
- applications: 'Machine learning'
- stochastic approximation of: 'Gradient descent'
- variants: momentum, Averaging, AdaGrad, RMSProp, Adam (many implemented in Keras and/or Tensorflow)

## Interior-point method
- also called: 'Barrier method'
- https://en.wikipedia.org/wiki/Interior-point_method
- http://mathworld.wolfram.com/InteriorPointMethod.html
- implemented in: 'Mathematica FindMinimum(Method->'InteriorPoint')'
- applications: 'Convex optimization'
- type: 'Constrained convex optimization'

## Sequential Least SQuares Programming
- also called: SLSQP
- implemented in: 'Python scipy.optimize.minimize(method="SLSQP")'
- type of: 'Sequential quadratic programming'

## Sequential quadratic programming
- also called: SQP
- https://en.wikipedia.org/wiki/Sequential_quadratic_programming
- type: 'Constrained non-linear' for 'twice continuously differentiable functions'

## Successive linear programming
- also called: SLP, 'Sequential Linear Programming'
- https://en.wikipedia.org/wiki/Successive_linear_programming
- type: 'Constrained non-linear'
- applications: 'Petrochemical industry'

## Augmented Lagrangian method
- https://en.wikipedia.org/wiki/Augmented_Lagrangian_method
- type: 'Constrained non-linear'
- applications: 'Total variation denoising', 'Compressed sensing'
- variant: 'Alternating direction method of multipliers'

## Penalty method
- https://en.wikipedia.org/wiki/Penalty_method
- type: 'Constrained non-linear'
- applications: 'Image compression'

-- Metaheuristics (randomized search methods)
  * Evolutionary algorithm
    - for one variant see Genetic algorithm
  * Genetic algorithm (GA)
  * Local search
    - variants: Hill climbing, Tabu search, Simulated annealing
    - applications: vertex cover problem, traveling salesman problem, boolean satisfiability problem, nurse scheduling problem, k-medoid
  * Simulated annealing (SA)
    - applications: combinatorial optimization problems
    - approximate global optimization
	- implemented in: 'Mathematica NMinimize(Method->"SimulatedAnnealing")'
	- http://mathworld.wolfram.com/SimulatedAnnealing.html
  * Tabu search (TS)
    - combinatorial optimization problems
  * Ant colony optimization algorithms
    - applications: combinatorial optimization problems
    - see also: Swarm intelligence
    - works well on graphs with changing topologies

-- Combinatorial optimization (https://en.wikipedia.org/wiki/Combinatorial_optimization)
  * see also Metaheuristics

-- Stochastic optimization
  * Stochastic approximation
  * see also Metaheuristics

-- nonlinear with NOT ENOUGH INFO
  * Newton conjugate gradient
    - implemented in scipy.optimize.minimize(method="trust-ncg")
  * Constrained Optimization BY Linear Approximation (COBYLA) algorithm
    - implemented in scipy.optimize.minimize(method="COBYLA")
  * Trust-region dogleg
    - implemented in scipy.optimize.minimize(method="dogleg")
  * Linear Programming
    - implemented in Mathematica FindMinimum(Method->'LinearProgramming')
    - is this SLP?

-- Linear least squares
  * "Direct" and "IterativeRefinement", and for sparse arrays "Direct" and "Krylov"
  * implemented in: 'minimize(method="trust-krylov")'

-- important practical theorems
  * no free lunch theorem

# Machine learning algorithms

## Self-organizing map
- https://en.wikipedia.org/wiki/Self-organizing_map
- type of: 'Artificial neural network'
- unsupervised
- solves: 'Nonlinear dimensionality reduction'
- applications: 'Visualization'
- implemented in: 'Python mvpa2.mappers.som.SimpleSOMMapper'

## Kernel principal component analysis
- https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
- is a: 'Machine learning algorithm', 'Kernel method'
- solves: 'Nonlinear dimensionality reduction'
- applications: 'novelty detection', 'image de-noising'
- implemented in: 'Python sklearn.decomposition.KernelPCA'

## Isomap
- https://en.wikipedia.org/wiki/Isomap
- paper: 'A Global Geometric Framework for Nonlinear Dimensionality Reduction (2000)'
- solves: 'Nonlinear dimensionality reduction'

## Autoencoder
- https://en.wikipedia.org/wiki/Autoencoder
- type of: 'Artificial neural network'
- solves: 'Nonlinear dimensionality reduction'
- applications: 'Generative model', 'Feature learning'
- variants: 'Variational autoencoder', 'Contractive autoencoder'
- unsupervised
