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

## Sudoku
- its graph is a: 'maximally connected bipartite directed graph'
- https://opensourc.es/blog/sudoku/
- "Calculate the strongly connected components of the graph. Then remove all the edges that bridge different strongly connected components"

## One-way privacy-preserving distance calculation
- also called: 'Location proximity'
- prevents C from learning about the exact values of S
- used for: 'Social networking'
- methods: 'Spatial cloaking'

## Two-way privacy-preserving distance calculation
- prevents C from learning about the exact values of S and vice versa
- used for: 'Biometric identification', 'Biometric authentication'
- methods:
	- '(additive) Homomorphic encryption (HE)'
	- 'Yao's garbled circuits protocol'
	- 'oblivious transfer (OT)'
	- 'GMW protocol'
- paper: 'GSHADE: faster privacy-preserving distance computation and biometric identification' (2014) <https://doi.org/10.1145/2600918.2600922>
- paper: 'Privacy-preserving Edit Distance on Genomic Data' (2017) <https://arxiv.org/abs/1711.06234>
- problem: 'Secure two-party computation'
- applications: 'wireless sensor networks'
- solved by (protocols): 'GSHADE'

## Yao's Millionaires' Problem
- https://en.wikipedia.org/wiki/Yao%27s_Millionaires%27_Problem
- solved by (protocols): 'Garbled circuit'
- domain: 'cryptography'
- problem: 'Secure multi-party computation'

## Socialist millionaire problem
- https://en.wikipedia.org/wiki/Socialist_millionaire_problem
- domain: 'cryptography'
- variant of: 'Yao's Millionaires' Problem'
