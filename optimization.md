# Books

- 'Nonlinear Programming: 3rd Edition' by 'Dimitri P. Bertsekas'
- 'Clever Algorithms: Nature-Inspired Programming Recipes' by 'Jason Brownlee'
- 'Convex Optimization' by 'Stephen Boyd' and 'Lieven Vandenberghe'

# Problems

## Linear programming
- also called: 'LP', 'Linear optimization', 'Constrained linear optimization'
- https://en.wikipedia.org/wiki/Linear_programming
- implemented by: 'Mathematica FindMinimum(Method->"LinearProgramming")' (what is the algorithm here?)
- implemented by: 'Artelys Knitro'

## Quadratic programming
- also called: 'QP'
- https://en.wikipedia.org/wiki/Quadratic_programming
- implemented by: 'APOPT'

## Quadratically constrained quadratic program
- also called: 'QCQP'
- hardness: 'NP-hard'
- https://en.wikipedia.org/wiki/Quadratically_constrained_quadratic_program
- implemented by: 'APOPT', 'Artelys Knitro'

## Integer linear programming
- also called: 'ILP'
- https://en.wikipedia.org/wiki/Integer_programming
- hardness: NP-complete
- applications: 'Production planning', 'Scheduling', 'Network design'

## Mixed Integer Linear Programming
- also called: 'MILP'
- implemented by: 'lp_solve', 'APOPT'

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
- also called: 'NLP', 'Constrained nonlinear optimization'
- https://en.wikipedia.org/wiki/Nonlinear_programming
- implemented in: 'APOPT', 'Artelys Knitro'

## Integer Nonlinear Programming
- also called: 'INLP'

## Mixed Integer Nonlinear Programming
- also called: 'MINLP'
- https://neos-guide.org/content/mixed-integer-nonlinear-programming
- implemented in: 'Artelys Knitro'

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

## Cutting-plane method
- https://en.wikipedia.org/wiki/Cutting-plane_method
- solves: 'Mixed Integer Linear Programming', 'Convex optimization'

## Branch and bound
- https://en.wikipedia.org/wiki/Branch_and_bound
- paper: 'An Automatic Method of Solving Discrete Programming Problems (1960)'
- solves: 'Mixed Integer Nonlinear Programming'
- implemented in: 'Artelys Knitro'

## Branch and cut
- https://en.wikipedia.org/wiki/Branch_and_cut
- solves: 'Integer linear programming', 'Mixed Integer Nonlinear Programming'

## Branch and reduce
- paper: 'A branch-and-reduce approach to global optimization (1996)'

## Quesada Grossman algorithm
- solves: 'Mixed Integer Nonlinear Programming'
- implemented in: 'Artelys Knitro', 'AIMMS'

## Outer approximation algorithm
- solves: 'Mixed Integer Nonlinear Programming'

## Ordinary least squares method
- https://en.wikipedia.org/wiki/Ordinary_least_squares
- solves: 'Linear least squares'
- input: 'Parameterized linear function'

## Karmarkar's algorithm
- https://en.wikipedia.org/wiki/Karmarkar%27s_algorithm
- paper: 'A new polynomial-time algorithm for linear programming'
- type: 'Linear programming'
- is a: 'Interior point method'
- input: 'Linear program'

## Simplex algorithm
- https://en.wikipedia.org/wiki/Simplex_algorithm
- http://mathworld.wolfram.com/SimplexMethod.html
- also called: 'Dantzig's simplex algorithm'
- type: 'Linear programming'
- implemented in: 'Mathematica LinearProgramming(Method->"Simplex")'
- implemented in: 'scipy.optimize.linprog(method="simplex")'
- implemented in: 'IBM ILOG CPLEX'
- input: 'Linear program'

## Revised simplex method
- https://en.wikipedia.org/wiki/Revised_simplex_method
- implemented in: 'Mathematica LinearProgramming(Method->"RevisedSimplex")'
- type: 'Linear programming'
- variant of: 'Simplex algorithm'
- input: 'Linear program'

## Powell's method
- https://en.wikipedia.org/wiki/Powell%27s_method
- paper: 'An efficient method for finding the minimum of a function of several variables without calculating derivatives (1964)'
- type: 'Unconstrained nonlinear without derivatives'
- is a: 'Optimization method'
- implemented in: 'Python scipy.optimize.minimize(method="Powell")' (modified variant)
- input: 'Real-valued function of several real variables'

## Luus–Jaakola
- https://en.wikipedia.org/wiki/Luus–Jaakola
- paper: 'Optimization by direct search and systematic reduction of the size of search region'
- is a: 'Heuristic'
- type: 'Unconstrained nonlinear without derivatives'
- becomes 'Iterative method' for 'twice continuously differentiable functions', however for this problem 'Newton's method' is usually used
- applications: 'optimal control', 'transformer design', 'metallurgical processes', 'chemical engineering'
- input: 'Real-valued function of several real variables'
- implemented in: 'Python osol.extremum'
- approximates: 'Global extremum'

## Pattern search
- https://en.wikipedia.org/wiki/Pattern_search_(optimization)
- paper: '"Direct Search" Solution of Numerical and Statistical Problems'
- type: 'Unconstrained nonlinear without derivatives'
- is a: 'Optimization method'
- is a: 'Heuristic' or 'Iterative method', depending on the class of function WHICH CLASSES?
- input: 'Function'

## Golden-section search
- https://en.wikipedia.org/wiki/Golden-section_search
- paper: 'Sequential minimax search for a maximum'
- input: 'Strictly unimodal function'
- related: 'Fibonacci search'
- type: 'Unconstrained nonlinear without derivatives'

## Successive parabolic interpolation
- https://en.wikipedia.org/wiki/Successive_parabolic_interpolation
- paper: 'An iterative method for locating turning points (1966)'
- type: 'Unconstrained nonlinear without derivatives'
- input: 'continuous unimodal function'
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
- input: 'Real-valued function of several real variables'

## Random search
- https://en.wikipedia.org/wiki/Random_search
- paper: 'The convergence of the random search method in the extremal control of a many parameter system'
- is a: 'Randomized algorithm'
- implemented in: 'Mathematica NMinimize(Method->"RandomSearch")'
- type: 'Unconstrained nonlinear without derivatives'
- related: 'Random optimization'
- input: 'Real-valued function of several real variables'

## Random optimization
- https://en.wikipedia.org/wiki/Random_optimization
- paper: 'Random optimization (1965)'
- type: 'Unconstrained nonlinear without derivatives'
- is a: 'Randomized algorithm'
- related: 'Random search'
- input: 'Real-valued function of several real variables'

## Nelder–Mead method
- also called: 'downhill simplex method'
- https://en.wikipedia.org/wiki/Nelder–Mead_method
- http://mathworld.wolfram.com/Nelder-MeadMethod.html
- http://www.scholarpedia.org/article/Nelder-Mead_algorithm
- paper: 'A simplex method for function minimization'
- type: 'Unconstrained nonlinear without derivatives'
- implemented in: 'Mathematica NMinimize(Method->"NelderMead")', 'Python scipy.optimize.minimize(method="Nelder-Mead")'
- implemented in: 'gsl_multimin_fminimizer_nmsimplex2'
- is a: "Heuristic method"
- input: 'Real-valued function of several real variables'

## Genetic algorithm
- https://en.wikipedia.org/wiki/Genetic_algorithm
- book: 'Adaptation in natural and artificial systems (1975)'
- type: 'Unconstrained nonlinear without derivatives'
- is a: 'Metaheuristic', 'Randomized search method', 'Evolutionary algorithm'
- used for: 'Integer Nonlinear Programming'
- input: 'Genetic representation', 'Fitness function'

## Principal Axis Method
- https://reference.wolfram.com/language/tutorial/UnconstrainedOptimizationPrincipalAxisMethod.html
- also called: 'PRincipal Axis (PRAXIS) algorithm', 'Brent's algorithm'
- book: 'Algorithms for Minimization without Derivatives (1972/1973)'
- uses: 'SVD', 'Adaptive coordinate descent'
- implemented in: 'Mathematica FindMinimum(Method->"PrincipalAxis")'
- input: 'Real-valued function of several real variables'

## Newton's method
- https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
- type: 'Unconstrained nonlinear optimization with Hessian'
- implemented in: 'Mathematica FindMinimum(Method->"Newton")'
- is a: 'Anytime algorithm'
- input: 'twice-differentiable function'

## Broyden–Fletcher–Goldfarb–Shanno algorithm
- https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm
- is a: 'Quasi-Newton method', 'Iterative method'
- implemented in: 'Python scipy.optimize.minimize(method="BFGS")', 'Mathematica FindMinimum(Method->"QuasiNewton")'
- implemented in: 'gsl_multimin_fdfminimizer_vector_bfgs2'
- type: 'Unconstrained nonlinear with first derivative'
- input: 'Differentiable real-valued function of several real variables and its gradient'

## Limited-memory BFGS
- https://en.wikipedia.org/wiki/Limited-memory_BFGS
- is a: 'Quasi-Newton method'
- variant of: 'BFGS' (for large systems with memory optimizations)
- implemented in: 'Mathematica FindMinimum(Method->"QuasiNewton")', 'Python scipy.minimize(method="L-BFGS-B")' (L-BFGS-B variant)
- applications: 'Maximum entropy models', 'CRF'
- input: 'Differentiable real-valued function of several real variables and its gradient'

## Basin-hopping algorithm
- https://en.wikipedia.org/wiki/Basin-hopping
- paper: 'Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms'
- implemented in: 'scipy.optimize.basinhopping'
- input: 'smooth scalar function of one or more variables'
- applications: 'Physical Chemistry'
- works good if there are lots of local extrema
- is a: 'stochastic algorithm', 'sampling algorithm'
- type: 'Unconstrained nonlinear without derivatives'
- depends on any: 'local optimization algorithm'
- approximates: 'Global extremum'

## Levenberg–Marquardt algorithm
- also called: 'LMA'
- https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
- http://mathworld.wolfram.com/Levenberg-MarquardtMethod.html
- paper: 'A Method for the Solution of Certain Non-Linear Problems in Least Squares (1944)'
- type: 'Unconstrained nonlinear with first derivative'
- solves: 'Non-linear least squares', 'Orthogonal non-linear least squares'
- is a: 'iterative method'
- applications: 'generic curve-fitting problems'
- output: 'local extermum'
- implemented in: 'Mathematica FindMinimum(Method->"LevenbergMarquardt")', 'Python scipy.optimize.least_squares(method="lm"), scipy.optimize.root(method="lm")', 'MINPACK', 'R ONLS'
- input: 'Sum of squares function and data pairs'
- variant of: 'Gauss–Newton'

## Davidon–Fletcher–Powell formula
- also called: 'DFP'
- https://en.wikipedia.org/wiki/Davidon%E2%80%93Fletcher%E2%80%93Powell_formula
- type: 'Unconstrained nonlinear with first derivative'
- is a: 'Quasi-Newton method'
- superseded by: 'Broyden–Fletcher–Goldfarb–Shanno algorithm'
- input: 'Function and its gradient'

## Symmetric rank-one
- https://en.wikipedia.org/wiki/Symmetric_rank-one
- also called: 'SR1'
- is a: 'Quasi-Newton method'
- advantages for sparse or partially separable problems
- implemented in: 'Python scipy.optimize.SR1'
- type: 'Unconstrained nonlinear with first derivative'
- input: 'Function and its gradient'

## Berndt–Hall–Hall–Hausman algorithm
- also called: 'BHHH'
- https://en.wikipedia.org/wiki/Berndt%E2%80%93Hall%E2%80%93Hall%E2%80%93Hausman_algorithm
- type: 'Unconstrained nonlinear with first derivative'
- similar: 'Gauss–Newton algorithm'
- input: 'log likelihood function'?

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
- input: 'Differentiable real-valued function of several real variables and its gradient'

## Gauss–Newton algorithm
- https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
- applications: 'Non-linear least squares'
- type: 'Unconstrained nonlinear with first derivative'
- input: 'sum of squares function'

## Trust region method
- also called: 'restricted-step method'
- https://en.wikipedia.org/wiki/Trust_region
- https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods
- http://www.applied-mathematics.net/optimization/optimizationIntro.html
- type: 'Unconstrained nonlinear with first derivative'

## Gradient descent
- also called: 'Steepest descent'
- https://en.wikipedia.org/wiki/Gradient_descent
- type: 'Unconstrained nonlinear with first derivative'
- will converge to a global minimum if the function is convex
- stochastic approximation: 'Stochastic gradient descent'
- applications: 'Convex optimization'
- implemented in: 'gsl_multimin_fdfminimizer_steepest_descent'
- input: 'Differentiable real-valued function of several real variables and its gradient'

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
- input: 'Convex real-valued function of several real variables'

## Sequential Least SQuares Programming
- also called: 'SLSQP'
- implemented in: 'Python scipy.optimize.minimize(method="SLSQP")'
- type of: 'Sequential quadratic programming'

## Sequential quadratic programming
- also called: 'SQP'
- https://en.wikipedia.org/wiki/Sequential_quadratic_programming
- type: 'Nonlinear programming' for 'twice continuously differentiable functions'
- implemented in: 'Artelys Knitro'

## Successive linear programming
- also called: 'SLP', 'Sequential Linear Programming'
- https://en.wikipedia.org/wiki/Successive_linear_programming
- type: 'Nonlinear programming'
- applications: 'Petrochemical industry'
- input: 'Nonlinear program'

## Augmented Lagrangian method
- https://en.wikipedia.org/wiki/Augmented_Lagrangian_method
- type: 'Nonlinear programming'
- applications: 'Total variation denoising', 'Compressed sensing'
- variant: 'Alternating direction method of multipliers'
- input: 'Function and constraints'

## Penalty method
- https://en.wikipedia.org/wiki/Penalty_method
- type: 'Nonlinear programming'
- applications: 'Image compression'
- input: 'Function and constraints'

## Local search
- https://en.wikipedia.org/wiki/Local_search_(optimization)

## Hill climbing
- https://en.wikipedia.org/wiki/Hill_climbing
- type: 'Local search', 'Heuristic search'
- is a: 'Iterative method', 'Anytime algorithm', 'Metaheuristic'?
- applications: 'Artificial intelligence'
- output: 'Local extremum'

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
- paper: 'Cuckoo Search via Levy Flights (2010)'
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
- paper: 'A Direct Search Optimization Method That Models the Objective and Constraint Functions by Linear Interpolation (1994)'
- implemented in: 'Python scipy.optimize.minimize(method="COBYLA")'
- type: 'Constrained optimization without derivatives'
- input: 'Real-valued function of several real variables'

## Dogleg Method
- also called: 'Trust-region dogleg'
- https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods#Dogleg_Method
- implemented in: 'Python scipy.optimize.minimize(method="dogleg")'
- is a: 'Trust region method'

# Practical theorems

## No free lunch theorem
- https://en.wikipedia.org/wiki/No_free_lunch_theorem
- "any two optimization algorithms are equivalent when their performance is averaged across all possible problems"
