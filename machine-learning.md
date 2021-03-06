# Problems

## Classification
- https://en.wikipedia.org/wiki/Statistical_classification
- type: 'Supervised learning'

## Binary classification
- special case of: 'Classification'

## Multiclass classification
- special case of: 'Classification'

## Regression
- also called: 'Regression analysis'
- https://en.wikipedia.org/wiki/Regression_analysis
- type: 'Supervised learning'

## One-class classification
- also called: 'OCC'
- https://en.wikipedia.org/wiki/One-class_classification
- variants: 'Novelty detection', 'Anomaly detection'
- domain: 'Machine learning'

## Positive-unlabeled learning
- also called: 'PU learning'
- https://en.wikipedia.org/wiki/One-class_classification#PU_learning
- paper: 'Building text classifiers using positive and unlabeled examples' (2003) <https://doi.org/10.1109/ICDM.2003.1250918>
- domain: 'Machine learning'
- special case of: 'One-class classification'

## Novelty detection
- https://en.wikipedia.org/wiki/Novelty_detection
- clean dataset
- type of: 'One-class classification'
- domain: 'Machine learning'

## Anomaly detection
- also called: 'Outlier detection'
- https://en.wikipedia.org/wiki/Anomaly_detection
- outliers are included in dataset
- type of: 'One-class classification'
- domain: 'Machine learning'

# ML related, but not really ML

## GPipe
- paper: 'GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism'
- applications: 'Neural Network Scaling and Distribution'
- input: 'Neural network'
- implemented in: 'GPipe library'

# Meta algorithms

## Bootstrap aggregating
- also called: 'bagging'
- paper: 'Bagging predictors' (1996) <https://doi.org/10.1007/BF00058655>
- https://en.wikipedia.org/wiki/Bootstrap_aggregating
- implemented in (libraries): 'sklearn.ensemble.BaggingClassifier'

# Machine learning models and algorithms

## Dense Passage Retriever
- also called: 'DPR'
- solves: 'Information retrieval', 'Ranking'
- uses: 'BERT', 'FAISS'
- applications: 'Open-domain question answering'

## Naive Bayes classifier
- also called: 'Naïve Bayes classifiers'
- solves: 'Classification'
- is a: 'Linear classifier'
- uses model: 'Naïve Bayes probability model'
- decision rule: 'Maximum a posteriori estimation'
- related: 'Logistic regression'
- is a bad estimator for probabilities

## Gaussian Naive Bayes
- usually optimized by: 'Maximum likelihood'
- variant of: 'Naive Bayes classifier'
- implemented in (libraries): 'sklearn.naive_bayes.GaussianNB'

## Multinomial Naive Bayes
- variant of: 'Naive Bayes classifier'
- implemented in (libraries): 'sklearn.naive_bayes.MultinomialNB'

## Bernoulli Naive Bayes
- variant of: 'Naive Bayes classifier'
- implemented in (libraries): 'sklearn.naive_bayes.BernoulliNB'

## Linear discriminant analysis
- also called: 'LDA', 'Normal discriminant analysis', 'NDA', 'Discriminant function analysis'
- https://en.wikipedia.org/wiki/Linear_discriminant_analysis
- related: 'Analysis of variance', 'Principal component analysis'
- implemented in (libraries): 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'
- quadratic variant: 'Quadratic discriminant analysis'
- is a: 'Linear classifier'

## Quadratic discriminant analysis
- also called: 'QDA'
- https://en.wikipedia.org/wiki/Quadratic_classifier#Quadratic_discriminant_analysis
- implemented in (libraries): 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'
- is a: 'Quadratic classifier'

## Bagging SVM
- solves: 'Positive-unlabeled learning'
- paper: 'A bagging SVM to learn from positive and unlabeled examples' (2014) <https://doi.org/10.1016/j.patrec.2013.06.010>

## Structured support vector machine
- also called: 'Structured SVM', 'Structural support vector machine', 'SSVM'
- https://en.wikipedia.org/wiki/Structured_support_vector_machine

## Positive Unlabeled Random Forest
- also called: 'PURF'
- solves: 'Positive-unlabeled learning'
- paper: 'Towards Positive Unlabeled Learning for Parallel Data Mining: A Random Forest Framework' (2014) <https://doi.org/10.1007/978-3-319-14717-8_45>

## REINFORCE
- also called: 'Monto-Carlo policy gradient'
- paper: 'Simple statistical gradient-following algorithms for connectionist reinforcement learning' (1992)
- is a: 'Policy gradient algorithm'
- applications: 'Reinforcement learning'

## Actor-Critic
- paper: 'Neuronlike adaptive elements that can solve difficult learning control problems' (1983)
- type: 'Temporal-Difference Learning'

## Advantage Actor-Critic
- also called: 'Actor Advantage Critic', 'A2C'

## Asynchronous Advantage Actor-Critic
- also called: 'A3C'
- paper: 'Asynchronous Methods for Deep Reinforcement Learning' (2016)
- is a: 'Policy gradient algorithm'
- applications: 'Reinforcement learning'

## Soft Actor-Critic
- also called: 'SAC'
- paper: 'Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor' (2018)
- based on: 'Actor-critic'
- applications: 'Reinforcement learning'
- properties: 'off-policy'

## Q-learning
- https://en.wikipedia.org/wiki/Q-learning
- thesis: 'Learning from delayed rewards' (1989)
- properties: 'model-free', 'value based', 'off-policy'
- applications: 'Reinforcement learning'

## Deep Q-Networks
- also called: 'DQN'
- paper: 'Playing Atari with Deep Reinforcement Learning' (2013)
- based on: 'Q-learning'
- properties: 'value based'

## Self-critical sequence training
- also called: 'SCST'
- paper: 'Self-critical Sequence Training for Image Captioning' (2016)
- form of: REINFORCE algorithm

## Sparse FC
- paper: 'Kernelized Synaptic Weight Matrices'
- applications: 'Collaborative Filtering', 'Recommender Systems'
- compare: 'I-AutoRec', 'CF-NADE', 'I-CFN', 'GC-MC'

## BERT
- also called: 'Bidirectional Encoder Representations from Transformers'
- paper: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding'
- applications: 'Natural Language Processing', 'Question Answering', 'Natural Language Inference', 'Sentiment Analysis'
- is a: 'Masked Language Model', 'Semantic hash', 'Trained hash'
- implemented in: 'google-research/bert'
- trained on: 'Cloze task', 'Next Sentence Prediction'
- based on: 'Transformer'
- domain: 'Unsupervised machine learning'

## OpenAI GPT
- also called: 'Generative Pre-training Transformer'
- paper: 'Improving Language Understandingby Generative Pre-Training'
- based on: 'Transformer'
- is a: 'Language Model'
- domain: 'Unsupervised machine learning'

## GPT-2
- paper: 'Language Models are Unsupervised Multitask Learners'
- https://openai.com/blog/better-language-models/
- successor to: 'OpenAI GPT'
- is a: 'Language Model'
- based on: 'Transformer'
- domain: 'Unsupervised machine learning'

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

## MSRN
- paper: 'Multi-scale Residual Network for ImageSuper-Resolution'
- applications: 'Single Image Super-Resolution'

## Automatic Differentiation Variational Inference
- also called: 'ADVI'
- paper: 'Automatic Variational Inference in Stan (2015)'
- implemented in: 'Stan'
- does: 'approximate Bayesian inference'

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
- solves: 'Nonlinear dimensionality reduction', 'Dimensionality reduction'
- applications: 'Visualization'
- implemented in: 'Python mvpa2.mappers.som.SimpleSOMMapper, Bio.Cluster.somcluster'
- paradigm: 'Competitive learning'

## Recursive autoencoder
- also called: 'RAE'
- paper: 'Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection' (2011)
- applications: 'Paraphrase detection'

## Conditional Variational Autoencoder
- also called: 'CVAE'
- paper: 'Learning Structured Output Representation using Deep Conditional Generative Models' (2015)

## OneClass SVM
- solves: 'Novelty detection'
- implemented in: 'libsvm', 'sklearn.svm.OneClassSVM'
- uses: 'Support estimation'

## Support-vector clustering
- also called: 'SVC'
- paper: 'Support vector clustering' (2001)
- http://www.scholarpedia.org/article/Support_vector_clustering
- implemented in: 'RapidMiner'

## Chinese Whispers
- https://en.wikipedia.org/wiki/Chinese_Whispers_(clustering_method)
- paper: 'Chinese Whispers - an Efficient Graph Clustering Algorithm and its Application to Natural Language Processing Problems' (2006) <https://www.aclweb.org/anthology/W06-3812/>
- applications: 'Clustering', 'Community identification', 'Natural language processing'
- implemented in (libraries): 'dlib'

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
- also called: 'iForest'
- https://en.wikipedia.org/wiki/Isolation_forest
- solves: 'Anomaly detection'
- paper: 'Isolation Forest' (2008) <https://doi.org/10.1109/ICDM.2008.17>
- implemented in: 'sklearn.ensemble.IsolationForest'
- is a: 'Ensemble method'

## Extended Isolation Forest
- also called: 'EIF'
- https://en.wikipedia.org/wiki/Isolation_forest
- paper: 'Extended Isolation Forest' (2019) <https://doi.org/10.1109/TKDE.2019.2947676>
- solves: 'Anomaly detection'
- improvement of: 'Isolation Forest'
- implemented in: 'Python eif'

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

## Random forest
- https://en.wikipedia.org/wiki/Random_forest
- properties: 'Embarrassingly parallel' (Tree growth step)

## Multi-Channel Convolutional Neural Network
- also called: 'MCCNN'
- paper: 'Question Answering on Freebase via Relation Extraction and Textual Evidence' (2016)
- applications: 'Relation Extraction'
