# Data structures

## Sparse voxel octree
- also called: 'SVO'
- paper: 'Efficient Sparse Voxel Octrees' (2010)
- https://en.wikipedia.org/wiki/Sparse_voxel_octree
- https://code.google.com/archive/p/efficient-sparse-voxel-octrees/
- implemented in: 'tunabrain/sparse-voxel-octrees'
- applications: 'Rendering'

# Models

## Relevance model
- paper: 'Relevance based language models (2001)'

## RM3
- paper: 'UMass at TREC 2004: Novelty and HARD'
- is a: 'Language model'
- based on: 'Relevance model'

# Problems

## Mesh generation
- also called: 'Meshing'
- https://en.wikipedia.org/wiki/Mesh_generation
- domain: '3D computer graphics'

## Visibility problem
- https://en.wikipedia.org/wiki/Visibility_(geometry)
- related problem: 'Hidden-surface determination'

## Hidden-surface determination
- also called: 'Hidden-surface removal', 'HSR', 'Visible-surface determination', 'VSD'
- https://en.wikipedia.org/wiki/Hidden-surface_determination

## View-Frustum culling
- https://en.wikipedia.org/wiki/Hidden-surface_determination#Viewing-frustum_culling
- special case of: 'Hidden-surface determination'
- domain: '3D computer graphics'

## Occlusion culling
- also called: 'OC'
- https://en.wikipedia.org/wiki/Hidden-surface_determination#Occlusion_culling
- special case of: 'Hidden-surface determination'
- domain: '3D computer graphics'

## Rendering
- also called: 'Image synthesis'
- https://en.wikipedia.org/wiki/Rendering_(computer_graphics)
- approximated by (applications): 'POV-Ray'

## Rendering equation
- https://en.wikipedia.org/wiki/Rendering_equation
- solved approximately by: 'Metropolis light transport', 'Photon mapping', 'Radiosity method'
- subproblem of: 'Rendering equation'

## Global illumination
- subproblem of: 'Rendering'

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

## Finite element method
- also called: 'FEM'
- https://en.wikipedia.org/wiki/Finite_element_method

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

## Radiosity method
- uses: 'Finite element method'
- https://en.wikipedia.org/wiki/Radiosity_(computer_graphics)
- applications: 'Rendering'
- domain: 'Computer graphics'

## Image-based lighting
- https://en.wikipedia.org/wiki/Image-based_lighting

## Ray tracing
- https://en.wikipedia.org/wiki/Ray_tracing_(graphics)
- applications: 'Rendering'
- domain: 'Computer graphics'

# Algorithms

## Metropolis light transport
- also called: 'MLT'
- uses: 'Metropolis–Hastings algorithm'
- solves approximately: 'Rendering equation'
- properties: 'unbiased'
- applications: 'Global illumination'

## Photon mapping
- paper: 'Global Illumination using Photon Maps' (1996)
- https://en.wikipedia.org/wiki/Photon_mapping
- http://graphics.ucsd.edu/~henrik/papers/photon_map/
- solves approximately: 'Rendering equation'
- applications: 'Global illumination'
- properties: 'biased'

## Painter's algorithm
- also called: 'Priority fill'
- https://en.wikipedia.org/wiki/Painter%27s_algorithm
- applications: 'Visibility problem'

## Warnock algorithm
- paper: 'A hidden surface algorithm for computer generated halftone pictures' (1969)
- https://en.wikipedia.org/wiki/Warnock_algorithm
- applications: 'Hidden-surface determination'

## Greedy meshing
- also called: 'Greedy voxel meshing'
- https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/
- https://www.gedge.ca/dev/2014/08/17/greedy-voxel-meshing
- applications: 'Mesh generation', 'Destructive geometry'
- implemented in: 'Java roboleary/GreedyMesh'

## Culling (meshing)
- applications: 'Mesh generation', 'Destructive geometry'
- https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/

## Ray casting
- paper: 'Ray casting for modeling solids' (1982)
- https://en.wikipedia.org/wiki/Ray_casting

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
- paper: 'Attitude determination using vector observations: A fast optimal matrix algorithm' (1993)
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
