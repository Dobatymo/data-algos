# Models

## Relevance model
- paper: 'Relevance based language models (2001)'

## RM3
- paper: 'UMass at TREC 2004: Novelty and HARD'
- is a: 'Language model'
- based on: 'Relevance model'

# Problems

## Wahba's problem
- https://en.wikipedia.org/wiki/Wahba%27s_problem
- solved by: 'Singular Value Decomposition'
- domain: 'Linear algebra'
- applications: 'Spacecraft control', 'Physics simulations'
- similar: 'Orthogonal Procrustes problem'
- related: 'Shape Matching', 'Co-rotational Finite Element Method'

## Orthogonal Procrustes problem
- https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
- similar: 'Wahba's problem'

# Methods

## Co-rotational Finite Element Method
- also called: 'Corotational FEM', 'CLFEM', 'Warped stiffness model'
- paper: 'Stable real-time deformations (2002)'
- book: 'Graphical Simulation of Deformable Models'
- extension papers: 'Fast Corotated FEM using Operator Splitting'
- applications: 'Physics simulations', 'Simulation of deformable objects'
- see also: 'Wahba's problem'
- is a: 'Physically-Based Deformable Model', 'Linear FEM'

## Shape Matching (for deformations)
- paper: 'Meshless Deformations Based on Shape Matching (2005)'
- book: 'Graphical Simulation of Deformable Models'
- applications: 'Physics simulations', 'Simulation of deformable objects'
- see also: 'Wahba's problem'
- is a: 'Geometrically-Based Method', 'Position-Based Simulation Method'

## Lattice Shape Matching
- also called: 'LSM'
- paper: 'FastLSM: Fast Lattice Shape Matching for Robust Real-Time Deformation (2007)'
- variant of: 'Shape Matching'

# Algorithms

## Davenport's Q-method
- paper: 'A vector approach to the algebra of rotations with applications (1968)'
- solves: 'Wahba's problem'

## QUaternion ESTimator
- also called: 'QUEST'
- paper: 'Three-axis attitude determination from vector observations (1981)'
- solves: 'Wahba's problem'

## EStimator of the Optimal Quaternion
- also called: 'ESOQ'
- paper: 'ESOQ: A Closed-Form Solution to the Wahba Problem (1997)'
- solves: 'Wahba's problem'

## Fast Optimal Attitude Matrix
- also called: 'FOAM'
- paper: 'Attitude determination using vector observations: A fast optimal matrix algorithm (1993)'
- solves: 'Wahba's problem'

## Kabsch algorithm
- https://en.wikipedia.org/wiki/Kabsch_algorithm
- calculates optimal rotation matrix that minimizes the RMSD

## Irving's rotation matrix extraction algorithm
- paper: 'Invertible finite elements for robust simulation of large deformation (2004)'
- solves: 'Wahba's problem'
- input: '3x3 matrix'
- uses: 'Polar decomposition'

## Müller's rotation matrix extraction algorithm
- paper: 'A robust method to extract the rotational part of deformations (2016)'
- solves: 'Wahba's problem'
- input: '3x3 matrix'
- domain: 'Linear algebra'
- is a: 'iterative algorithm'
- properties: 'warm startable', 'stable', 'no branching'
