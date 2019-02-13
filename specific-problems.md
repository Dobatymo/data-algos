# Specific problems

## Knight's tour
- https://en.wikipedia.org/wiki/Knight%27s_tour
- http://mathworld.wolfram.com/KnightGraph.html
- its graph is a: 'bipartite graph'
- solved by 'depth first search' on graph of legal moves, using 'Warnsdorff's Rule' (heuristic) for improved performance
- version of: 'Hamiltonian path problem'

## Tower of Hanoi
- https://en.wikipedia.org/wiki/Tower_of_Hanoi
- http://mathworld.wolfram.com/TowerofHanoi.html
- applications: 'Backup rotation scheme'
- isomorphic to finding a 'Hamiltonian path' on an n-hypercube

## Optimal solutions for Rubik's Cube
- calculate maximal turn numbers by: diameters of the corresponding Cayley graphs of the Rubik's Cube group
- use 'Korf's algorithm' to calculate optimal solution
- implementation: http://www.cflmath.com/~reid/Rubik/optimal_solver.html

## Matching parentheses
- also called: 'Balanced parentheses'
- properties: 'parallelizable'
- usually solved using: 'Stack'

## Mandelbrot set
- properties: 'Embarrassingly parallel'
