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

# ML related, but not really ML

## GPipe
- paper: 'GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism'
- applications: 'Neural Network Scaling and Distribution'
- input: 'Neural network'
- implemented in: 'GPipe library'

# Machine learning models and algorithms

## Sparse FC
- paper: 'Kernelized Synaptic Weight Matrices'
- applications: 'Collaborative Filtering', 'Recommender Systems'
- compare: 'I-AutoRec', 'CF-NADE', 'I-CFN', 'GC-MC'

## BERT
- paper: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding'
- applications: 'Natural Language Processing', 'Question Answering', 'Natural Language Inference', 'Sentiment Analysis'
- is a: 'Language Model'
- implemented in: 'google-research/bert'
- trained on: 'Cloze task', 'Next Sentence Prediction'
- is a: 'Semantic hash', 'Trained hash'
- based on: 'Transformer'

## DrQA
- paper: 'Reading Wikipedia to Answer Open-Domain Questions'
- applications: 'Extractive Question Answering'

## HyperQA
- paper: 'Hyperbolic Representation Learning for Fast and Efficient Neural Question Answering'
- applications: 'Multiple Choice Question Answering / Answer selection'

## RMDL
- paper: 'RMDL: Random Multimodel Deep Learning for Classification' (2018)
- applications: 'Document Classification', 'Image Classification'
- implemented in: 'Python kk7nc/RMDL'

## DCSCN
- paper: 'Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network' (2017)
- applications: 'Single Image Super-Resolution'
- is a: 'CNN'

## SRGAN
- paper: 'Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network' (2016)
- applications: 'Single Image Super-Resolution'
- is a: 'GAN'

## ESRGAN
- paper: 'ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks' (2018)
- is a: 'GAN'
- applications: 'Single Image Super-Resolution'
- first place in: PIRM2018-SR Challenge

## MSRN
- paper: 'Multi-scale Residual Network for ImageSuper-Resolution'
- applications: 'Single Image Super-Resolution'

## SOF-VSR
- paper: 'Learning for Video Super-Resolution through HR Optical Flow Estimation' (2018)
- applications: 'Video super-resolution'
- is a: 'CNN'

## ESPCN
- paper: 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network' (2016)
- applications: 'Single image super-resolution', 'Video super-resolution'
- is a: 'CNN'

## VSR-DUF
- paper: 'Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation' (2018)
- applications: 'Video super-resolution'
- is a: 'CNN'

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
- also called: 'Kohonen map', 'Kohonen network'
- https://en.wikipedia.org/wiki/Self-organizing_map
- type of: 'Artificial neural network'
- unsupervised
- solves: 'Nonlinear dimensionality reduction'
- applications: 'Visualization'
- implemented in: 'Python mvpa2.mappers.som.SimpleSOMMapper, Bio.Cluster.somcluster'
- paradigm: 'Competitive learning'

## Wake-sleep algorithm
- https://en.wikipedia.org/wiki/Wake-sleep_algorithm
- is a: 'Unsupervised learning algorithm'
- properties: 'convergent'
- trains: 'Helmholtz machine'

## Autoencoder
- https://en.wikipedia.org/wiki/Autoencoder
- type of: 'Artificial neural network'
- solves: 'Nonlinear dimensionality reduction'
- applications: 'Generative model', 'Feature learning', 'Image Compression', 'Image Denoising', 'Image Generation'
- variants: 'Variational autoencoder', 'Contractive autoencoder'
- unsupervised
- similar: 'Principal component analysis'

## Recursive autoencoder
- also called: 'RAE'
- paper: 'Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection' (2011)
- applications: 'Paraphrase detection'

## Variational autoencoder
- also called: 'VAE'
- paper: 'Auto-Encoding Variational Bayes' (2013)
- https://en.wikipedia.org/wiki/Autoencoder#Variational_autoencoder_(VAE)
- type of: 'Autoencoder'
- see: 'Variational Bayesian methods'

## Conditional Variational Autoencoder
- also called: 'CVAE'
- paper: 'Learning Structured Output Representation using Deep Conditional Generative Models' (2015)

## OneClass SVM
- solves: 'Novelty detection'
- implemented in: 'libsvm', 'sklearn.svm.OneClassSVM'
- uses: 'Support estimation'

## Support-vector machine
- also called: 'SVM', 'Support-vector network'
- https://en.wikipedia.org/wiki/Support-vector_machine
- implemented in: 'LIBLINEAR', 'sklearn.svm.LinearSVC', 'sklearn.svm.SVC'
- Quadratic Programming problem
- Linear variant optimized by: 'Coordinate descent', 'newGLMNET'

## Support-vector regression
- also called: 'SVR'
- https://en.wikipedia.org/wiki/Support-vector_machine#Regression
- implemented in: 'LIBLINEAR', 'libsvm', 'Python sklearn.svm.SVR, sklearn.svm.LinearSVR'

## Support-vector clustering
- also called: 'SVC'
- paper: 'Support vector clustering' (2001)
- http://www.scholarpedia.org/article/Support_vector_clustering
- implemented in: 'RapidMiner'

## ID3 algorithm
- also called: 'Iterative Dichotomiser 3'
- https://en.wikipedia.org/wiki/ID3_algorithm
- is a: 'Decision tree algorithm', 'Classification algorithm'

## C4.5 algorithm
- book: 'C4.5: Programs for Machine Learning' (1993)
- https://en.wikipedia.org/wiki/C4.5_algorithm
- is a: 'Decision tree algorithm', 'Classification algorithm'
- extension of: 'ID3 algorithm'
- implemented in: 'Weka'

## Classification and regression tree
- also called: 'CART'
- https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29
- is a: 'Decision tree algorithm', 'Classification algorithm', 'Regression algorithm'
- implemented in: 'Python sklearn.tree.DecisionTreeClassifier, sklearn.tree.DecisionTreeRegressor'
- properties: 'non-parametric'

## DBSCAN
- also called: 'Density-based spatial clustering of applications with noise'
- paper: 'A density-based algorithm for discovering clusters in large spatial databases with noise' (1996)
- https://en.wikipedia.org/wiki/DBSCAN
- is a: 'Density-based clustering algorithm'
- implemented in: 'Python sklearn.cluster.dbscan'

## OPTICS algorithm
- also called: 'Ordering points to identify the clustering structure algorithm'
- paper: 'OPTICS: Ordering Points To Identify the Clustering Structure' (1999)
- https://en.wikipedia.org/wiki/OPTICS_algorithm
- is a: 'Density-based clustering algorithm'
- generalization of: 'DBSCAN'

## SUBCLU
- paper: 'Density-Connected Subspace Clustering for High-Dimensional Data' (2004)
- https://en.wikipedia.org/wiki/SUBCLU
- variant of: 'DBSCAN'
- is a: 'Subspace clustering algorithm'

## Isolation Forest
- solves: 'Anomaly detection'
- paper: 'Isolation Forest' (2008)
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

## Conditional random field
- also called: 'CRF'
- https://en.wikipedia.org/wiki/Conditional_random_field
- is a: 'undirected probabilistic graphical model'
- applications: 'POS tagging', 'shallow parsing', 'named entity recognition', 'object recognition'
- usually optimized by: 'L-BFGS', 'Stochastic gradient descent'
- implemented in: 'CRFSuite', 'python-crfsuite'

## Autoregressive–moving-average model
- also called: 'ARMA'
- https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model
- implemented in: 'Python statsmodels.tsa.arima_model.ARMA'
- applications: 'Time series analysis'

## Autoregressive integrated moving average
- also called: 'ARIMA'
- https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
- implemented in: 'Python statsmodels.tsa.arima_model.ARIMA'
- applications: 'Time series analysis'

## Random forest
- https://en.wikipedia.org/wiki/Random_forest
- properties: 'Embarrassingly parallel' (Tree growth step)

## Convolutional neural network
- also called: 'CNN', 'ConvNet'
- https://en.wikipedia.org/wiki/Convolutional_neural_network
- properties: 'Embarrassingly parallel'
- type of: 'Artificial neural network'
- applications: 'Computer vision', 'Natural language processing'

## Recurrent neural network
- also called: 'RNN'
- https://en.wikipedia.org/wiki/Recurrent_neural_network
- type of: 'Artificial neural network'
- applications: 'Natural language processing', 'Speech recognition'

## Long short-term memory
- also called: 'LSTM'
- type of: 'Recurrent neural network'
- https://en.wikipedia.org/wiki/Long_short-term_memory
- applications: 'Sequence learning'

## Transformer
- https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html
- paper: 'Attention Is All You Need' (2017)
- type of: 'Artificial neural network'
- is a: 'Language model'
- applications: 'Natural language processing', 'Machine translation', 'Question answering'
- originally optimized by: 'Adam'

## Multi-Channel Convolutional Neural Network
- also called: 'MCCNN'
- paper: 'Question Answering on Freebase via Relation Extraction and Textual Evidence' (2016)
- applications: 'Relation Extraction'
