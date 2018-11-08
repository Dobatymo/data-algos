-- Metrics, Measures, Distances, Similarities, Losses and Costs --

# papers:
- images: 'Distance functions on digital pictures'

# SimRank
- https://en.wikipedia.org/wiki/SimRank
- paper: 'SimRank: a measure of structural-context similarity (2002)'
- domain: graph theory

# Levenshtein distance
- https://en.wikipedia.org/wiki/Levenshtein_distance
- is a: 'metric', 'edit distance'
- applications: 'Spelling correction', 'Sequence alignment', 'Approximate string matching', 'Linguistic distance'
- properties: 'Discrete'

# Hamming distance
- https://en.wikipedia.org/wiki/Hamming_distance
- applications: 'Coding theory', 'Block code', 'Error detection and correction', 'Telecommunication'
- implemented in: 'Python sklearn.metrics.hamming_loss'
- properties: 'Discrete'

# Damerau–Levenshtein distance
- https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
- is a: 'edit distance'
- is not: 'metric'
- applications: 'Spelling correction'
- properties: 'Discrete'

# Jaro–Winkler distance
- https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance
- is a: 'edit distance'
- is not: 'metric'
- properties: 'Discrete'

# Jaccard index
- https://en.wikipedia.org/wiki/Jaccard_index
- is a: 'metric', 'edit distance'
- properties: 'Discrete'
- implemented in: 'Python sklearn.metrics.jaccard_similarity_score'

# Taxicab metric
- https://en.wikipedia.org/wiki/Taxicab_geometry
- is a: 'metric'
- applications: 'Regression analysis', 'LASSO'

# Cosine similarity
- https://en.wikipedia.org/wiki/Cosine_similarity
- is not: 'metric'
- applications: 'natural language processing', 'data mining'
- metric version: 'angular distance'

# Logistic loss
- also called: 'Log loss', 'Cross entropy loss'
- https://en.wikipedia.org/wiki/Cross_entropy
- applications: 'Deep learning', 'Logistic regression'
- implemented in: 'Python sklearn.metrics.log_loss'
- properties: 'Convex', 'Continuous'

# Mean squared error
- https://en.wikipedia.org/wiki/Mean_squared_error
- applications: 'Statistical model', 'Linear regression'
- implemented in: 'Python sklearn.metrics.mean_squared_error, tf.metrics.mean_squared_error'
- properties: 'Continuous'

# Mean absolute error
- https://en.wikipedia.org/wiki/Mean_absolute_error
- implemented in: 'Python sklearn.metrics.mean_absolute_error, tf.metrics.mean_absolute_error, tf.losses.absolute_difference'

# Hinge loss
- https://en.wikipedia.org/wiki/Hinge_loss
- applications: 'Support vector machine'
- properties: 'Convex', 'Continuous'
- implemented in: 'Python sklearn.metrics.hinge_loss'

# Explained variation
- https://en.wikipedia.org/wiki/Explained_variation
- applications: 'Regression analysis'

# Algebraic distance
- https://en.wikipedia.org/wiki/Distance#Algebraic_distance
- http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FISHER/ALGDIST/alg.htm
- applications: 'Computer vision'
- properties: 'Linear'

# Chamfer distance
- paper: 'Sequential Operations in Digital Picture Processing (1966)'
- applications: 'Computer vision', 'Image similarity'
- approximate: 'Euclidean distance'

# Hausdorff distance
- https://en.wikipedia.org/wiki/Hausdorff_distance
- domain: 'Set theory'
- applications: 'Computer vision'

# Mahalanobis distance
- https://en.wikipedia.org/wiki/Mahalanobis_distance
- applications: 'Cluster analysis', 'Anomaly detection'

# q-gram distance
- for example defined in paper: 'Approximate string-matching with q-grams and maximal matches'