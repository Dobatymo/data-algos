# Problems

## One-class classification
- https://en.wikipedia.org/wiki/One-class_classification
- variants: 'Novelty detection', 'Anomaly detection'

## Novelty detection
- https://en.wikipedia.org/wiki/Novelty_detection
- clean dataset
- type of: 'One-class classification'

## Anomaly detection
- also called: 'Outlier detection'
- https://en.wikipedia.org/wiki/Anomaly_detection
- outliers are included in dataset
- type of: 'One-class classification'

# Algorithms

# Machine learning algorithms

## Isomap
- https://en.wikipedia.org/wiki/Isomap
- paper: 'A Global Geometric Framework for Nonlinear Dimensionality Reduction (2000)'
- solves: 'Nonlinear dimensionality reduction'

## Self-organizing map
- https://en.wikipedia.org/wiki/Self-organizing_map
- type of: 'Artificial neural network'
- unsupervised
- solves: 'Nonlinear dimensionality reduction'
- applications: 'Visualization'
- implemented in: 'Python mvpa2.mappers.som.SimpleSOMMapper, Bio.Cluster.somcluster'

## Autoencoder
- https://en.wikipedia.org/wiki/Autoencoder
- type of: 'Artificial neural network'
- solves: 'Nonlinear dimensionality reduction'
- applications: 'Generative model', 'Feature learning'
- variants: 'Variational autoencoder', 'Contractive autoencoder'
- unsupervised

## OneClass SVM
- solves: 'Novelty detection'
- implemented in: 'libsvm', 'sklearn.svm.OneClassSVM'
- uses: 'Support estimation'

## Isolation Forest
- solves: 'Anomaly detection'
- paper: 'Isolation Forest'
- implemented in: 'sklearn.ensemble.IsolationForest'
- is a: 'Ensemble method'

## Local Outlier Factor
- solves: 'Novelty detection', 'Anomaly detection'
- also called: 'LOF'
- https://en.wikipedia.org/wiki/Local_outlier_factor
- paper: 'LOF: identifying density-based local outliers'
- implemented in: 'sklearn.neighbors.LocalOutlierFactor', 'ELKI'
- is a: 'Nearest neighbor method'

## Elliptic Envelope
- solves: 'Anomaly detection'
- paper: 'A Fast Algorithm for the Minimum Covariance Determinant Estimator (1998)'
- implemented in: 'sklearn.covariance.EllipticEnvelope'
- input: 'Normal distributed data' with n_samples > n_features ** 2
- uses: 'Covariance estimation'
