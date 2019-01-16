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

# Machine learning models and algorithms

## Automatic Differentiation Variational Inference
- also called: 'ADVI'
- paper: 'Automatic Variational Inference in Stan (2015)'
- implemented in: 'Stan'
- does: 'approximate Bayesian inference'

## Hamiltonian Monte Carlo
- also called: 'HMC', 'Hybrid Monte Carlo'
- https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo
- paper: 'Hybrid Monte Carlo (1987)'
- is a: 'Markov chain Monte Carlo algorithm'
- solves: 'Sampling'
- input: 'probability distribution'
- output: 'random samples'
- applications: 'Lattice QCD'
- implemented in: 'Stan'

## No-U-Turn Sampler
- also called: 'NUTS'
- paper: 'The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo (2011)'
- extends: 'Hamiltonian Monte Carlo'
- implemented in: 'Stan'
- solves: 'Sampling'

## Gradient Tree Boosting
- also called: 'Gradient boosting machine' (GBM), 'Gradient boosted regression tree' (GBRT)
- also called: 'Gradient Boosting Decision Tree' (GBDT), 'Multiple Additive Regression Trees' (MART)
- https://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting
- implemented in: 'xgboost', 'LightGBM', 'pGBRT', 'catboost'
- applications: 'Learning to rank'
- easily distributable

## Elastic net regularization
- https://en.wikipedia.org/wiki/Elastic_net_regularization
- implemented in: 'sklearn.linear_model.ElasticNet'
- combines: 'LASSO regularization' and 'Tikhonov regularization'

## Tikhonov regularization
- also called: 'Ridge regression'
- https://en.wikipedia.org/wiki/Tikhonov_regularization

## LASSO regularization
- https://en.wikipedia.org/wiki/Lasso_(statistics)
- also called: 'Least absolute shrinkage and selection operator'
- applications: 'regression analysis'
- implemented in: 'sklearn.linear_model.Lasso' (fitted with 'Coordinate descent')
- properties: 'linear'

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
- paradigm: 'Competitive learning'

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
