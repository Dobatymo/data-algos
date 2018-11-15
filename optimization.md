# Books

- 'Nonlinear Programming: 3rd Edition' by 'Dimitri P. Bertsekas'
- 'Clever Algorithms: Nature-Inspired Programming Recipes' by 'Jason Brownlee'
- 'Convex Optimization' by 'Stephen Boyd' and 'Lieven Vandenberghe'

# Problems

## Linear programming
- also called: 'Linear optimization', 'Constrained linear optimization'
- https://en.wikipedia.org/wiki/Linear_programming
- implemented in: 'Mathematica FindMinimum(Method->"LinearProgramming")' (what is the algorithm here?)

## Convex optimization
- https://en.wikipedia.org/wiki/Convex_optimization

## Unconstrained nonlinear optimization without derivatives
- also called: 'black box', 'direct search'
- https://en.wikipedia.org/wiki/Derivative-free_optimization
- https://en.wikipedia.org/wiki/Nonlinear_programming
- domain: 'Numerical optimization'
- mainly used for: 'functions which are not continuous or differentiable'
- domain: 'Mathematical optimization'

## Unconstrained nonlinear optimization with first derivative
- https://en.wikipedia.org/wiki/Nonlinear_programming

## Nonlinear programming
- also called: 'Constrained nonlinear optimization'
- https://en.wikipedia.org/wiki/Nonlinear_programming

## Combinatorial optimization
- https://en.wikipedia.org/wiki/Combinatorial_optimization

## Nonlinear dimensionality reduction
- https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction

## Linear least squares
- https://en.wikipedia.org/wiki/Linear_least_squares
- implemented in: 'Mathematica LeastSquares(Method->"Direct"), LeastSquares(Method->"IterativeRefinement"), LeastSquares(Method->"Krylov")'
- implemented in: 'Python scipy.optimize.minimize(method="trust-krylov")'

## Total least squares
- https://en.wikipedia.org/wiki/Total_least_squares
- generalization of: 'Deming regression', 'Orthogonal regression'
- is a: 'Errors-in-variables model'

## Non-linear least squares
- https://en.wikipedia.org/wiki/Non-linear_least_squares
- applications: 'Nonlinear regression'

## Orthogonal regression
- https://en.wikipedia.org/wiki/Deming_regression#Orthogonal_regression
- http://www.nlreg.com/orthogonal.htm
- also called: 'Orthogonal distance regression', 'Orthogonal non-linear least squares'
- implemented by: 'scipy.odr', 'ODRPACK', 'R ONLS'

## Vertex cover problem
- https://en.wikipedia.org/wiki/Vertex_cover
- solved approximatly by: 'Approximate global optimization'
- hardness: NP-hard

## Boolean satisfiability problem
- https://en.wikipedia.org/wiki/Boolean_satisfiability_problem
- solved approximatly by: 'Approximate global optimization'
- hardness: NP-complete

## Nurse scheduling problem
- https://en.wikipedia.org/wiki/Nurse_scheduling_problem
- solved approximatly by: 'Approximate global optimization'
- hardness: NP-hard

# Numerical optimization algorithms

## Ordinary least squares method
- https://en.wikipedia.org/wiki/Ordinary_least_squares
- solves: 'Linear least squares'

## Karmarkar's algorithm
- https://en.wikipedia.org/wiki/Karmarkar%27s_algorithm
- paper: 'A new polynomial-time algorithm for linear programming'
- type: 'Linear programming'
- is a: 'Interior point method'

## Simplex algorithm
- https://en.wikipedia.org/wiki/Simplex_algorithm
- http://mathworld.wolfram.com/SimplexMethod.html
- also called: 'Dantzig's simplex algorithm'
- type: 'Linear programming'
- implemented in: 'Mathematica LinearProgramming(Method->"Simplex")'
- implemented in: 'scipy.optimize.linprog(method="simplex")'

## Revised simplex method
- https://en.wikipedia.org/wiki/Revised_simplex_method
- implemented in: 'Mathematica LinearProgramming(Method->"RevisedSimplex")'
- type: 'Linear programming'
- variant of: 'Simplex algorithm'

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
- also called: 'downhill simplex method'
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
- is a: 'Metaheuristic', 'Randomized search method', 'Evolutionary algorithm'

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
- implemented in: 'Python scipy.optimize.minimize(method="trust-ncg")' (Trust region variant)

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
- implemented in: 'Mathematica FindMinimum(Method->"InteriorPoint"), LinearProgramming(Method->"InteriorPoint")'
- implemented in: 'scipy.optimize.linprog(method="interior-point")'
- applications: 'Convex optimization'
- type: 'Constrained convex optimization', 'Linear programming'

## Sequential Least SQuares Programming
- also called: SLSQP
- implemented in: 'Python scipy.optimize.minimize(method="SLSQP")'
- type of: 'Sequential quadratic programming'

## Sequential quadratic programming
- also called: SQP
- https://en.wikipedia.org/wiki/Sequential_quadratic_programming
- type: 'Nonlinear programming' for 'twice continuously differentiable functions'

## Successive linear programming
- also called: SLP, 'Sequential Linear Programming'
- https://en.wikipedia.org/wiki/Successive_linear_programming
- type: 'Nonlinear programming'
- applications: 'Petrochemical industry'

## Augmented Lagrangian method
- https://en.wikipedia.org/wiki/Augmented_Lagrangian_method
- type: 'Nonlinear programming'
- applications: 'Total variation denoising', 'Compressed sensing'
- variant: 'Alternating direction method of multipliers'

## Penalty method
- https://en.wikipedia.org/wiki/Penalty_method
- type: 'Nonlinear programming'
- applications: 'Image compression'

## Local search
- https://en.wikipedia.org/wiki/Local_search_(optimization)

## Hill climbing
- https://en.wikipedia.org/wiki/Hill_climbing
- type: 'Local search', 'Heuristic search'
- is a: 'Iterative method', 'Anytime algorithm', 'Metaheuristic'?
- applications: 'Artificial intelligence'
- finds: 'Local extremum'

## Tabu search
- https://en.wikipedia.org/wiki/Tabu_search
- paper: 'Future Paths for Integer Programming and Links to Artificial Intelligence (1986)'
- is a: 'Metaheuristic'
- type: 'Local search'
- applications: 'Combinatorial optimization', 'Travelling salesman problem'
- domain: 'Approximate global optimization'

## Simulated annealing
- https://en.wikipedia.org/wiki/Simulated_annealing
- http://mathworld.wolfram.com/SimulatedAnnealing.html
- applications: 'Combinatorial optimization'
- domain: 'Approximate global optimization'
- implemented in: 'Mathematica NMinimize(Method->"SimulatedAnnealing")'
- is a: 'Metaheuristic', 'Stochastic optimization method'

## Cuckoo search
- https://en.wikipedia.org/wiki/Cuckoo_search
- is a: 'Metaheuristic'
- applications: 'Operations research'

## Ant colony optimization algorithms
- https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms
- applications: 'Combinatorial optimization', 'Scheduling', 'Routing', 'Assignment problem', 'Edge detection'
- see also: 'Swarm intelligence'
- works well on graphs with changing topologies
- is a: 'Metaheuristic'

## Estimation of distribution algorithms
- https://en.wikipedia.org/wiki/Estimation_of_distribution_algorithm
- is a: 'Stochastic optimization method', 'Evolutionary algorithm'
- class of algorithms

## Constrained Optimization BY Linear Approximation
- also called: 'COBYLA'
- https://en.wikipedia.org/wiki/COBYLA
- implemented in scipy.optimize.minimize(method="COBYLA")
- type: 'Constrained optimization without derivatives'

## Dogleg Method
- also called: 'Trust-region dogleg'
- https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods#Dogleg_Method
- implemented in: 'Python scipy.optimize.minimize(method="dogleg")'
- is a: 'Trust region method'

# Practical theorems

## No free lunch theorem
- https://en.wikipedia.org/wiki/No_free_lunch_theorem
- "any two optimization algorithms are equivalent when their performance is averaged across all possible problems"

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
