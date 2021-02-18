# Problems 

## Sparse approximation
- https://en.wikipedia.org/wiki/Sparse_approximation
- domain: 'Linear algebra', 'Signal processing'
- applications: 'Machine learning'

## Logic optimization
- also called: 'Circuit minimization', 'Minimization of boolean functions'
- https://en.wikipedia.org/wiki/Logic_optimization
- solved by: 'Mathematica BooleanMinimize'
- domain: 'Optimization'

## Convex optimization
- https://en.wikipedia.org/wiki/Convex_optimization
- domain: 'Optimization'

## Unconstrained nonlinear optimization without derivatives
- also called: 'black box', 'direct search'
- https://en.wikipedia.org/wiki/Derivative-free_optimization
- https://en.wikipedia.org/wiki/Nonlinear_programming
- mainly used for: 'functions which are not continuous or differentiable'
- domain: 'Optimization'

## Unconstrained nonlinear optimization with first derivative
- https://en.wikipedia.org/wiki/Nonlinear_programming
- domain: 'Optimization'

## Combinatorial optimization
- https://en.wikipedia.org/wiki/Combinatorial_optimization
- example problems: 'Travelling salesman problem', 'Minimum spanning tree', 'Set cover problem'
- domain: 'Optimization'

## Nonlinear dimensionality reduction
- https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction

-- Model + Loss

## Ordinary least squares
- also called: 'Linear least squares', 'OLS', 'LLS'
- https://en.wikipedia.org/wiki/Ordinary_least_squares
- https://en.wikipedia.org/wiki/Linear_least_squares
- http://mlwiki.org/index.php/OLS_Regression
- solves: 'Linear least squares'
- input: 'Parameterized linear function'
- domain: 'Optimization'
- solved by (libraries): 'Python numpy.linalg.lstsq', 'Python scipy.optimize.minimize(method="trust-krylov")'
- solved by (applications): 'Mathematica LeastSquares(Method->"Direct"), LeastSquares(Method->"IterativeRefinement"), LeastSquares(Method->"Krylov")'

## Constrained linear least squares
- solved by (libraries): 'scipy.optimize.lsq_linear'

## Non-negative linear least-squares problem
- also called: 'NNLS'
- https://en.wikipedia.org/wiki/Non-negative_least_squares
- solved by: 'Matlab lsqnonneg', 'Python scipy.optimize.nnls'
- variant of: 'Ordinary least squares'
- specialization of: 'Quadratic programming'

## Bounded-variable least squares
- also called: 'BVLS'
- variant of: 'Ordinary least squares'
- specialization of: 'Quadratic programming'

## Orthogonal regression
- https://en.wikipedia.org/wiki/Deming_regression#Orthogonal_regression
- http://www.nlreg.com/orthogonal.htm
- also called: 'Orthogonal distance regression', 'Orthogonal non-linear least squares'
- implemented by: 'scipy.odr', 'ODRPACK', 'R ONLS'

## Total least squares
- https://en.wikipedia.org/wiki/Total_least_squares
- generalization of: 'Deming regression', 'Orthogonal regression'
- is a: 'Errors-in-variables model'
- similar: 'PCA'
- solved by (algorithms): 'SVD'
- solved by (libraries): 'netlib/vanhuffel'

## Non-linear least squares
- https://en.wikipedia.org/wiki/Non-linear_least_squares
- applications: 'Nonlinear regression'
- solved by (libraries): 'scipy.optimize.least_squares', 'Ceres Solver'

## Linear programming
- also called: 'LP', 'Linear optimization', 'Constrained linear optimization'
- https://en.wikipedia.org/wiki/Linear_programming
- implemented by: 'Mathematica LinearProgramming'
- implemented by: 'Mathematica FindMinimum(Method->"LinearProgramming")' (what is the algorithm here?)
- implemented by: 'Artelys Knitro', 'GNU Linear Programming Kit'

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
- implemented by: 'lp_solve', 'APOPT', 'GNU Linear Programming Kit'

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

# Optimization algorithms

## Iteratively reweighted least squares
- also called: 'IRLS'
- https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares
- domain: 'Optimization'

## Sequential Minimal Optimization
- also called: 'SMO'
- technical report: 'Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines' (1998) <https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/>
- https://en.wikipedia.org/wiki/Sequential_minimal_optimization
- solves: 'SVM type quadratic programming problems'

## SMO-type decomposition method
- based on: 'Sequential Minimal Optimization'
- paper: 'Working Set Selection Using Second Order Information for Training Support Vector Machines' (2005) <https://dl.acm.org/doi/10.5555/1046920.1194907>

## Cutting-plane method
- https://en.wikipedia.org/wiki/Cutting-plane_method
- solves: 'Mixed Integer Linear Programming', 'Convex optimization'
- domain: 'Optimization'

## Branch and bound
- https://en.wikipedia.org/wiki/Branch_and_bound
- paper: 'An Automatic Method of Solving Discrete Programming Problems (1960)'
- solves: 'Mixed Integer Nonlinear Programming'
- implemented in: 'Artelys Knitro'
- domain: 'Optimization'

## Branch and cut
- https://en.wikipedia.org/wiki/Branch_and_cut
- solves: 'Integer linear programming', 'Mixed Integer Nonlinear Programming'
- domain: 'Optimization'

## Branch and reduce
- paper: 'A branch-and-reduce approach to global optimization (1996)'
- domain: 'Optimization'

## Quesada Grossman algorithm
- solves: 'Mixed Integer Nonlinear Programming'
- implemented in: 'Artelys Knitro', 'AIMMS'
- domain: 'Optimization'

## Outer approximation algorithm
- solves: 'Mixed Integer Nonlinear Programming'
- domain: 'Optimization'

## Karmarkar's algorithm
- https://en.wikipedia.org/wiki/Karmarkar%27s_algorithm
- paper: 'A new polynomial-time algorithm for linear programming'
- type: 'Linear programming'
- is a: 'Interior point method'
- input: 'Linear program'
- domain: 'Optimization'

## Simplex algorithm
- https://en.wikipedia.org/wiki/Simplex_algorithm
- http://mathworld.wolfram.com/SimplexMethod.html
- book: 'Introduction to Algorithms'
- also called: 'Dantzig's simplex algorithm'
- type: 'Linear programming'
- implemented in: 'Mathematica LinearProgramming(Method->"Simplex")'
- implemented in: 'scipy.optimize.linprog(method="simplex")'
- implemented in: 'IBM ILOG CPLEX'
- input: 'Linear program'
- domain: 'Optimization'

## Revised simplex method
- https://en.wikipedia.org/wiki/Revised_simplex_method
- implemented in: 'Mathematica LinearProgramming(Method->"RevisedSimplex")'
- type: 'Linear programming'
- variant of: 'Simplex algorithm'
- input: 'Linear program'
- domain: 'Optimization'

## Powell's method
- https://en.wikipedia.org/wiki/Powell%27s_method
- paper: 'An efficient method for finding the minimum of a function of several variables without calculating derivatives (1964)'
- type: 'Unconstrained nonlinear without derivatives'
- is a: 'Optimization method'
- implemented in: 'Python scipy.optimize.minimize(method="Powell")' (modified variant)
- input: 'Real-valued function of several real variables'
- domain: 'Optimization'

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
- domain: 'Optimization'

## Pattern search
- https://en.wikipedia.org/wiki/Pattern_search_(optimization)
- paper: '"Direct Search" Solution of Numerical and Statistical Problems'
- type: 'Unconstrained nonlinear without derivatives'
- is a: 'Optimization method'
- is a: 'Heuristic' or 'Iterative method', depending on the class of function WHICH CLASSES?
- input: 'Function'
- domain: 'Optimization'

## Golden-section search
- https://en.wikipedia.org/wiki/Golden-section_search
- paper: 'Sequential minimax search for a maximum'
- input: 'Strictly unimodal function'
- related: 'Fibonacci search'
- type: 'Unconstrained nonlinear without derivatives'
- domain: 'Optimization'

## Successive parabolic interpolation
- https://en.wikipedia.org/wiki/Successive_parabolic_interpolation
- paper: 'An iterative method for locating turning points (1966)'
- type: 'Unconstrained nonlinear without derivatives'
- input: 'continuous unimodal function'
- global or local convergence is not guaranteed
- domain: 'Optimization'

## Line search
- https://en.wikipedia.org/wiki/Line_search
- https://www.mathworks.com/help/optim/ug/unconstrained-nonlinear-optimization-algorithms.html#f15468
- general strategy only, employed by many optimization methods
- domain: 'Optimization'

## Differential evolution
- https://en.wikipedia.org/wiki/Differential_evolution
- paper: 'Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces'
- type: 'Unconstrained nonlinear without derivatives'
- implemented in: 'Mathematica NMinimize(Method->"DifferentialEvolution")', 'scipy.optimize.differential_evolution'
- is a: 'Metaheuristic', 'Stochastic algorithm'
- input: 'Real-valued function of several real variables'
- domain: 'Optimization'

## Random search
- https://en.wikipedia.org/wiki/Random_search
- paper: 'The convergence of the random search method in the extremal control of a many parameter system'
- is a: 'Randomized algorithm'
- implemented in: 'Mathematica NMinimize(Method->"RandomSearch")'
- type: 'Unconstrained nonlinear without derivatives'
- related: 'Random optimization'
- input: 'Real-valued function of several real variables'
- domain: 'Optimization'

## Random optimization
- https://en.wikipedia.org/wiki/Random_optimization
- paper: 'Random optimization (1965)'
- type: 'Unconstrained nonlinear without derivatives'
- is a: 'Randomized algorithm'
- related: 'Random search'
- input: 'Real-valued function of several real variables'
- domain: 'Optimization'

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
- domain: 'Optimization'

## Genetic algorithm
- https://en.wikipedia.org/wiki/Genetic_algorithm
- book: 'Adaptation in natural and artificial systems (1975)'
- type: 'Unconstrained nonlinear without derivatives'
- is a: 'Metaheuristic', 'Randomized search method', 'Evolutionary algorithm'
- used for: 'Integer Nonlinear Programming'
- input: 'Genetic representation', 'Fitness function'
- domain: 'Optimization'

## Principal Axis Method
- https://reference.wolfram.com/language/tutorial/UnconstrainedOptimizationPrincipalAxisMethod.html
- also called: 'PRincipal Axis (PRAXIS) algorithm', 'Brent's algorithm'
- book: 'Algorithms for Minimization without Derivatives (1972/1973)'
- uses: 'SVD', 'Adaptive coordinate descent'
- implemented in: 'Mathematica FindMinimum(Method->"PrincipalAxis")'
- input: 'Real-valued function of several real variables'
- domain: 'Optimization'

## Newton's method
- https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
- type: 'Unconstrained nonlinear optimization with Hessian'
- implemented in: 'Mathematica FindMinimum(Method->"Newton")'
- is a: 'Anytime algorithm'
- input: 'twice-differentiable function'
- domain: 'Optimization'

## Broyden–Fletcher–Goldfarb–Shanno algorithm
- https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm
- is a: 'Quasi-Newton method', 'Iterative method'
- implemented in: 'Python scipy.optimize.minimize(method="BFGS")', 'Mathematica FindMinimum(Method->"QuasiNewton")'
- implemented in: 'gsl_multimin_fdfminimizer_vector_bfgs2', 'Ceres Solver'
- type: 'Unconstrained nonlinear with first derivative'
- input: 'Differentiable real-valued function of several real variables and its gradient'
- domain: 'Optimization'

## Limited-memory BFGS
- https://en.wikipedia.org/wiki/Limited-memory_BFGS
- is a: 'Quasi-Newton method'
- variant of: 'BFGS' (for large systems with memory optimizations)
- implemented in: 'Mathematica FindMinimum(Method->"QuasiNewton")', 'Python scipy.minimize(method="L-BFGS-B")' (L-BFGS-B variant)
- implemented in: 'Ceres Solver'
- applications: 'Maximum entropy models', 'CRF'
- input: 'Differentiable real-valued function of several real variables and its gradient'
- domain: 'Optimization'

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
- domain: 'Optimization'

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
- implemented in: 'Mathematica FindMinimum(Method->"LevenbergMarquardt")', 'Python scipy.optimize.least_squares(method="lm"), scipy.optimize.root(method="lm")', 'scipy.optimize.leastsq', 'MINPACK', 'R ONLS'
- implemented in: 'Ceres Solver'
- input: 'Sum of squares function and data pairs'
- variant of: 'Gauss–Newton'
- domain: 'Optimization'

## Davidon–Fletcher–Powell formula
- also called: 'DFP'
- https://en.wikipedia.org/wiki/Davidon%E2%80%93Fletcher%E2%80%93Powell_formula
- type: 'Unconstrained nonlinear with first derivative'
- is a: 'Quasi-Newton method'
- superseded by: 'Broyden–Fletcher–Goldfarb–Shanno algorithm'
- input: 'Function and its gradient'
- domain: 'Optimization'

## Symmetric rank-one
- https://en.wikipedia.org/wiki/Symmetric_rank-one
- also called: 'SR1'
- is a: 'Quasi-Newton method'
- advantages for sparse or partially separable problems
- implemented in: 'Python scipy.optimize.SR1'
- type: 'Unconstrained nonlinear with first derivative'
- input: 'Function and its gradient'
- domain: 'Optimization'

## Berndt–Hall–Hall–Hausman algorithm
- also called: 'BHHH'
- https://en.wikipedia.org/wiki/Berndt%E2%80%93Hall%E2%80%93Hall%E2%80%93Hausman_algorithm
- type: 'Unconstrained nonlinear with first derivative'
- similar: 'Gauss–Newton algorithm'
- input: 'log likelihood function'?
- domain: 'Optimization'

## Nonlinear conjugate gradient method
- https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
- type: 'Unconstrained nonlinear with first derivative'
- implemented in: 'Mathematica FindMinimum(Method->"ConjugateGradient")'
- implemented in: 'Python scipy.optimize.minimize(method="Newton-CG")'
- implemented in: 'Python scipy.optimize.minimize(method="trust-ncg")' (Trust region variant)
- domain: 'Optimization'

## Truncated Newton method
- https://en.wikipedia.org/wiki/Truncated_Newton_method
- type: 'Unconstrained nonlinear with first derivative'
- implemented in: 'Python scipy.optimize.minimize(method="TNC")'
- input: 'Differentiable real-valued function of several real variables and its gradient'
- domain: 'Optimization'

## Gauss–Newton algorithm
- https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
- applications: 'Non-linear least squares'
- solves: 'Unconstrained nonlinear with first derivative'
- input: 'sum of squares function'
- domain: 'Optimization'

## Trust region method
- also called: 'restricted-step method'
- https://en.wikipedia.org/wiki/Trust_region
- https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods
- http://www.applied-mathematics.net/optimization/optimizationIntro.html
- solves: 'Unconstrained nonlinear with first derivative'
- domain: 'Optimization'

## Trust Region Reflective
- paper: 'A Subspace, Interior, and Conjugate Gradient Method for Large-Scale Bound-Constrained Minimization Problems' (1999)
- https://www.mathworks.com/help/optim/ug/least-squares-model-fitting-algorithms.html
- implemented in: 'Python scipy.optimize.least_squares(method="trf")', 'scipy.optimize.lsq_linear(method="trf")'
- solves: 'Non-linear least squares', ''Linear least squares''

## Bounded-variable least-squares algorithm
- paper: 'Bounded-Variable Least-Squares: an Algorithm and Applications' (1995)
- implemented in: 'scipy.optimize.lsq_linear(method="bvls")'
- solves: 'Linear least squares'

## Proximal gradient method
- https://en.wikipedia.org/wiki/Proximal_gradient_method
- solves: 'Convex optimization'
- domain: 'Optimization'

## Coordinate descent
- also called: 'CDN'
- https://en.wikipedia.org/wiki/Coordinate_descent
- application: 'L1-regularized classification'
- input: 'Real-valued function of several real variables'
- input: 'Differentiable real-valued function of several real variables and its gradient' (variant)
- domain: 'Optimization'

## Gradient descent
- also called: 'Steepest descent'
- https://en.wikipedia.org/wiki/Gradient_descent
- type: 'Unconstrained nonlinear with first derivative'
- will converge to a global minimum if the function is convex
- stochastic approximation: 'Stochastic gradient descent'
- applications: 'Convex optimization'
- implemented in: 'gsl_multimin_fdfminimizer_steepest_descent'
- input: 'Differentiable real-valued function of several real variables and its gradient'
- domain: 'Optimization'

## Stochastic gradient descent
- also called: 'SGD'
- https://en.wikipedia.org/wiki/Stochastic_gradient_descent
- applications: 'Machine learning'
- stochastic approximation of: 'Gradient descent'
- variants: momentum, Averaging, 'AdaGrad', 'RMSProp', 'Adam'
- domain: 'Optimization'

## Momentum method for stochastic gradient descent
- paper: 'Learning representations by back-propagating errors (1986)'
- implemented in: 'tf.train.MomentumOptimizer, keras.optimizers.SGD'
- domain: 'Optimization'

## Adam
- also called: 'Adaptive moment estimation optimization algorithm'
- paper: 'Adam: A Method for Stochastic Optimization (2014)'
- implemented in: 'tf.train.AdamOptimizer, keras.optimizers.Adam'
- variant of: 'Stochastic gradient descent'
- domain: 'Optimization'

## RAdam
- also called: 'Rectified Adam'
- paper: 'On the Variance of the Adaptive Learning Rate and Beyond' (2019)
- variant of: 'Adam'
- domain: 'Optimization'

## RMSProp
- also called: 'Root Mean Square Propagation'
- implemented in: 'tf.train.RMSPropOptimizer, keras.optimizers.RMSprop'
- domain: 'Optimization'

## ADADELTA
- paper: 'ADADELTA: An Adaptive Learning Rate Method (2012)'
- implemented in: 'tf.train.AdadeltaOptimizer, keras.optimizers.Adadelta'
- domain: 'Optimization'

## AdaGrad
- also called: 'Adaptive gradient algorithm'
- paper: 'Adaptive Subgradient Methods for Online Learning and Stochastic Optimization (2011)'
- implemented in: 'tf.train.AdagradOptimizer, keras.optimizers.Adagrad'
- domain: 'Optimization'

## Interior-point method
- also called: 'Barrier method'
- https://en.wikipedia.org/wiki/Interior-point_method
- http://mathworld.wolfram.com/InteriorPointMethod.html
- implemented in: 'Mathematica FindMinimum(Method->"InteriorPoint"), LinearProgramming(Method->"InteriorPoint")'
- implemented in: 'scipy.optimize.linprog(method="interior-point")'
- applications: 'Convex optimization'
- type: 'Constrained convex optimization', 'Linear programming'
- input: 'Convex real-valued function of several real variables'
- domain: 'Optimization'

## Successive linear programming
- also called: 'SLP', 'Sequential Linear Programming'
- https://en.wikipedia.org/wiki/Successive_linear_programming
- solves: 'Nonlinear programming'
- applications: 'Petrochemical industry'
- input: 'Nonlinear program'
- domain: 'Optimization'

## Sequential quadratic programming
- also called: 'SQP'
- https://en.wikipedia.org/wiki/Sequential_quadratic_programming
- solves: 'Nonlinear programming' for 'twice continuously differentiable functions'
- implemented in: 'Artelys Knitro'
- domain: 'Optimization'

## Han–Powell
- variant of: 'Sequential quadratic programming'
- domain: 'Optimization'

## NLPQL
- paper: 'NLPQL: A fortran subroutine solving constrained nonlinear programming problems' (1986)
- variant of: 'Sequential quadratic programming'
- implemented in: 'Fortran NLPQL'
- domain: 'Optimization'

## NLPQLP
- https://en.wikipedia.org/wiki/NLPQLP
- report: 'NLPQLP: A Fortran implementation of a sequential quadratic programming algorithm with distributed and non-monotone line search'
- variant of: 'Sequential quadratic programming'
- solves: 'Nonlinear programming' for 'twice continuously differentiable functions'
- implemented in: 'Fortran NLPQLP'
- domain: 'Optimization'

## Sequential Least SQuares Programming
- also called: 'SLSQP'
- implemented in: 'Python scipy.optimize.minimize(method="SLSQP")'
- domain: 'Optimization'
- uses: 'Han–Powell'
- variant of: 'Sequential quadratic programming'

## Frank–Wolfe algorithm
- also called: 'conditional gradient method', 'reduced gradient algorithm', 'convex combination algorithm'
- paper: 'An algorithm for quadratic programming (1956)'
- https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm
- solves: 'Constrained convex optimization'
- iterative first-order optimization algorithm 
- domain: 'Optimization'

## Augmented Lagrangian method
- https://en.wikipedia.org/wiki/Augmented_Lagrangian_method
- type: 'Nonlinear programming'
- applications: 'Total variation denoising', 'Compressed sensing'
- variant: 'Alternating direction method of multipliers' (ADMM)
- input: 'Function and constraints'
- domain: 'Optimization'

## Penalty method
- https://en.wikipedia.org/wiki/Penalty_method
- type: 'Nonlinear programming'
- applications: 'Image compression'
- input: 'Function and constraints'
- domain: 'Optimization'

## Local search
- https://en.wikipedia.org/wiki/Local_search_(optimization)
- domain: 'Optimization'

## Hill climbing
- https://en.wikipedia.org/wiki/Hill_climbing
- type: 'Local search', 'Heuristic search'
- is a: 'Iterative method', 'Anytime algorithm', 'Metaheuristic'?
- applications: 'Artificial intelligence'
- output: 'Local extremum'
- domain: 'Optimization'

## Tabu search
- https://en.wikipedia.org/wiki/Tabu_search
- paper: 'Future Paths for Integer Programming and Links to Artificial Intelligence (1986)'
- is a: 'Metaheuristic'
- type: 'Local search'
- applications: 'Combinatorial optimization', 'Travelling salesman problem'
- domain: 'Optimization', 'Approximate global optimization'

## Simulated annealing
- https://en.wikipedia.org/wiki/Simulated_annealing
- http://mathworld.wolfram.com/SimulatedAnnealing.html
- applications: 'Combinatorial optimization'
- domain: 'Optimization', 'Approximate global optimization'
- implemented in: 'Mathematica NMinimize(Method->"SimulatedAnnealing")'
- is a: 'Metaheuristic', 'Stochastic optimization method'

## Cuckoo search
- https://en.wikipedia.org/wiki/Cuckoo_search
- paper: 'Cuckoo Search via Levy Flights (2010)'
- is a: 'Metaheuristic'
- applications: 'Operations research'
- domain: 'Optimization'

## Ant colony optimization algorithms
- https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms
- applications: 'Combinatorial optimization', 'Scheduling', 'Routing', 'Assignment problem', 'Edge detection'
- see also: 'Swarm intelligence'
- works well on graphs with changing topologies
- is a: 'Metaheuristic'
- domain: 'Optimization'

## Fireworks algorithm
- also called: 'FWA', 'FA'
- paper: 'Fireworks Algorithm for Optimization' (2010)
- https://en.wikipedia.org/wiki/Fireworks_algorithm
- see: 'Swarm intelligence'
- domain: 'mathematical optimization'
- input: 'Function'

## Estimation of distribution algorithms
- https://en.wikipedia.org/wiki/Estimation_of_distribution_algorithm
- is a: 'Stochastic optimization method', 'Evolutionary algorithm'
- class of algorithms
- domain: 'Optimization'

## Constrained Optimization BY Linear Approximation
- also called: 'COBYLA'
- https://en.wikipedia.org/wiki/COBYLA
- paper: 'A Direct Search Optimization Method That Models the Objective and Constraint Functions by Linear Interpolation (1994)'
- implemented in: 'Python scipy.optimize.minimize(method="COBYLA")'
- type: 'Constrained optimization without derivatives'
- input: 'Real-valued function of several real variables'
- domain: 'Optimization'

## Dogleg Method
- also called: 'Trust-region dogleg'
- https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods#Dogleg_Method
- implemented in: 'Python scipy.optimize.minimize(method="dogleg")', 'scipy.optimize.least_squares(method="dogbox")', 'Ceres Solver'
- is a: 'Trust region method'
- domain: 'Optimization'

## Matching pursuit
- paper: 'Matching pursuits with time-frequency dictionaries (1993)'
- https://en.wikipedia.org/wiki/Matching_pursuit
- solves: 'Sparse approximation'
- properties: 'greedy'
- variants: 'Orthogonal Matching Pursuit'
- domain: 'Optimization'

## Orthogonal Matching Pursuit
- paper: 'Orthogonal Matching Pursuit: recursive function approximation with application to wavelet decomposition (1993)'
- implemented in: 'sklearn.linear_model.OrthogonalMatchingPursuit' (Orthogonal variant)
- variant of: 'Matching pursuit'
- domain: 'Optimization'

## Karnaugh map
- also called: 'K-map', 'Karnaugh–Veitch map'
- https://en.wikipedia.org/wiki/Karnaugh_map
- solves: 'Logic optimization'
- is a: 'algorithm for humans'
- domain: 'Optimization'

## Quine–McCluskey algorithm
- also called: 'method of prime implicants'
- https://en.wikipedia.org/wiki/Quine%E2%80%93McCluskey_algorithm
- solves: 'Logic optimization'
- time complexity: exponential
- domain: 'Optimization'

## Espresso algorithm
- paper: 'Logic Minimization Algorithms for VLSI Synthesis (1984)'
- https://en.wikipedia.org/wiki/Espresso_heuristic_logic_minimizer
- solves approximately: 'Logic optimization'
- is a: 'heuristic algorithm'
- implemented by: 'Espresso', 'Minilog'
- domain: 'Optimization'

## GLMNET
- paper: 'Regularization Paths for Generalized Linear Models via Coordinate Descent' (2010)
- applications: 'L1-regularized logistic regression'
- Newton-type optimizer
- implemented in: 'R glmnet'

## newGLMNET
- paper: 'An Improved GLMNET for L1-regularized LogisticRegression' (2012)
- applications: 'L1-regularized logistic regression'
- implemented in: 'LIBLINEAR'
- improved variant of: 'GLMNET'

# Practical theorems

## No free lunch theorem
- https://en.wikipedia.org/wiki/No_free_lunch_theorem
- "any two optimization algorithms are equivalent when their performance is averaged across all possible problems"
