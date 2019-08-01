# Models

-- definition:
	A model is a mathematical description of a function with free parameters which can be found by fitting the model to training data.
	A model is fitted using an 'optimizer'. A fitted model results in an 'algorithm'.
	Fitting is also called training. A model can be deterministic or non-deterministic (stochastic).
	They might also be categorized into 'discriminative' and 'generative'.
	A model should generalize, ie. it should have predictive power on unseen data.

-- topics models:

## Latent semantic analysis ?model or algorithm?
- also called: 'LSA', 'Latent semantic indexing', 'LSI'
- paper: 'Indexing by latent semantic analysis' (1990)
- https://en.wikipedia.org/wiki/Latent_semantic_analysis
- implemented in (libraries): 'Python gensim.models.lsimodel.LsiModel'
- properties: 'deterministic'

## Probabilistic latent semantic analysis
- also called: 'PLSA', 'Probabilistic latent semantic indexing', 'PLSI'
- paper: 'Probabilistic latent semantic analysis' (1999)
- https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis
- trained by: 'Expectation–maximization algorithm'
- applications: 'Statistical natural language processing'
- properties: 'stochastic'

## Latent Dirichlet allocation
- also called: 'LDA'
- paper: 'Latent Dirichlet allocation (2003)'
- https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation
- is a: 'Topic model'
- implemented in (libraries): 'Python gensim.models.ldamodel.LdaModel'
- commonly sampled by: 'variational inference', 'Collapsed Gibbs sampling'
- properties: 'generative', 'stochastic'

## Hierarchical Dirichlet process
- also called: 'HDP'
- paper: 'Hierarchical Dirichlet Processes' (2006)
- https://en.wikipedia.org/wiki/Hierarchical_Dirichlet_process
- implemented in (libraries): 'Python gensim.models.hdpmodel.HdpModel'
- properties: 'stochastic'

## Pachinko allocation model
- also called: 'PAM'
- paper: 'Pachinko allocation: DAG-structured mixture models of topic correlations' (2006)
- https://en.wikipedia.org/wiki/Pachinko_allocation
- is a: 'Topic model'
- properties: 'stochastic'
- implemented in (libraries): 'Python tomotopy.PAModel'

## Hierarchical Pachinko Allocation
- also called: 'HPA'
- paper: 'Mixtures of hierarchical topics with pachinko allocation' (2007)
- implemented in (libraries): 'Python tomotopy.HPAModel'

## Correlated topic model
- also called: 'CTM'
- paper: 'Correlated Topic Models' (2006)
- is a: 'Topic model'

## Dirichlet-multinomial regression
- also called: 'DMR'
- paper: 'Topic models conditioned on arbitrary features with Dirichlet-multinomial regression' (2012)
- implemented in (libraries): 'Python tomotopy.DMRModel'

-- classification or regression models:

## Logistic regression
- also called: 'logistic model', 'logit model'
- https://en.wikipedia.org/wiki/Logistic_regression
- properties: 'deterministic', 'discriminative'

## Conditional random field
- also called: 'CRF'
- https://en.wikipedia.org/wiki/Conditional_random_field
- properties: 'discriminative', 'undirected', 'probabilistic', 'graphical'
- applications: 'Structured prediction', 'POS tagging', 'shallow parsing', 'named entity recognition', 'object recognition'
- usually optimized by: 'L-BFGS', 'Stochastic gradient descent'
- implemented in: 'CRFSuite', 'python-crfsuite'

## k-nearest neighbors algorithm
- also called: 'k-NN', 'KNN'
- https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
- is a: 'Machine learning algorithm', 'Classification algorithm', 'Regression algorithm'
- properties: 'deterministic', 'discriminative', 'non-parametric', 'instance-based learning', 'lazy learning'
- special case of: 'Variable kernel density estimation'
- applications: 'Pattern recognition'
- implemented: 'Python sklearn.neighbors.KNeighborsRegressor, sklearn.neighbors.KNeighborsClassifier'

## Nearest centroid classifier
- https://en.wikipedia.org/wiki/Nearest_centroid_classifier
- is a: 'Classification model'
- input: 'Collection of points with associated labels' (training)
- input: 'Point' (prediction)
- output: 'Label'
- properties: 'reduced data model'
- relies on: 'Nearest neighbor search'

## Support-vector machine
- also called: 'SVM', 'Support-vector network'
- https://en.wikipedia.org/wiki/Support-vector_machine
- implemented in (libraries): 'LIBLINEAR', 'sklearn.svm.LinearSVC', 'sklearn.svm.SVC'
- Quadratic Programming problem
- Linear variant optimized by: 'Coordinate descent', 'newGLMNET'
- properties: 'deterministic', 'discriminative'

## Support-vector regression
- also called: 'SVR'
- https://en.wikipedia.org/wiki/Support-vector_machine#Regression
- implemented in (libraries): 'LIBLINEAR', 'libsvm', 'Python sklearn.svm.SVR, sklearn.svm.LinearSVR'
- properties: 'deterministic'

## Hidden Markov model
- also called: 'HMM'
- https://en.wikipedia.org/wiki/Hidden_Markov_model
- properties: 'generative', 'stochastic'
- type of: 'Markov model'
- applications: 'Speech recognition', 'Natural language processing', 'Bioinformatics'
- domain: 'Machine learning'

## Helmholtz machine
- https://en.wikipedia.org/wiki/Helmholtz_machine
- type of: 'Artificial neural network'
- properties: 'deterministic'

## Hopfield network
- https://en.wikipedia.org/wiki/Hopfield_network

## Boltzmann machine
- also called: 'Stochastic Hopfield network with hidden units'
- paper: 'A learning algorithm for Boltzmann machines' (1985)
- https://en.wikipedia.org/wiki/Boltzmann_machine
- type of: 'Recurrent neural network', 'Markov random field'
- is a: 'Energy based model'
- properties: 'generative', 'stochastic'

## Restricted Boltzmann machine
- also called: 'RBM', 'Harmonium'
- https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine
- properties: 'generative', 'stochastic'
- type of: 'Artificial neural network'
- variant of: 'Boltzmann machine'
- commonly optimized by: 'Contrastive divergence (CD) algorithm'

## Gaussian mixture model
- https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model
- properties: 'generative'

## Bayesian network
- https://en.wikipedia.org/wiki/Bayesian_network
- properties: 'generative'

## Autoencoder
- https://en.wikipedia.org/wiki/Autoencoder
- type of: 'Artificial neural network'
- solves: 'Nonlinear dimensionality reduction'
- applications: 'Generative model', 'Feature learning', 'Image Compression', 'Image Denoising', 'Image Generation'
- variants: 'Variational autoencoder', 'Contractive autoencoder'
- unsupervised
- similar: 'Principal component analysis'
- properties: 'discriminative'

## Variational autoencoder
- also called: 'VAE'
- paper: 'Auto-Encoding Variational Bayes' (2013)
- https://en.wikipedia.org/wiki/Autoencoder#Variational_autoencoder_(VAE)
- type of: 'Autoencoder'
- see: 'Variational Bayesian methods'
- properties: 'generative'

## Generative adversarial network
- also called: 'GAN'
- paper: 'Generative Adversarial Networks' (2014)
- https://en.wikipedia.org/wiki/Generative_adversarial_network

## SRGAN
- paper: 'Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network' (2016)
- applications: 'Single Image Super-Resolution'
- is a: 'GAN'

## ESRGAN
- paper: 'ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks' (2018)
- is a: 'GAN'
- applications: 'Single Image Super-Resolution'
- first place in: PIRM2018-SR Challenge

## Deep Convolutional Generative Adversarial Network
- also called: 'DCGAN'
- paper: 'Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks' (2015)
- is a: 'GAN'
- applications: 'Image generation'
- implemented in (libraries): 'soumith/dcgan.torch'

## Convolutional neural network
- also called: 'CNN', 'ConvNet'
- https://en.wikipedia.org/wiki/Convolutional_neural_network
- properties: 'Embarrassingly parallel', 'discriminative'
- type of: 'Artificial neural network'
- applications: 'Computer vision', 'Natural language processing'

## DCSCN
- paper: 'Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network' (2017)
- applications: 'Single Image Super-Resolution'
- is a: 'CNN'

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
- applications: 'Language modeling', 'Natural language processing', 'Machine translation', 'Question answering'
- originally optimized by: 'Adam'

## Transformer-XL
- paper: 'Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context' (2019)
- is a: 'Language Model'
- based on: 'Transformer'

## Autoregressive–moving-average model
- also called: 'ARMA'
- https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model
- implemented in: 'Python statsmodels.tsa.arima_model.ARMA'
- applications: 'Time series analysis'
- properties: 'autoregressive', 'linear'
- commonly estimated by: 'Box–Jenkins method'

## Autoregressive integrated moving average
- also called: 'ARIMA'
- https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
- implemented in: 'Python statsmodels.tsa.arima_model.ARIMA'
- applications: 'Time series analysis'
- properties: 'autoregressive', 'linear'
- commonly estimated by: 'Box–Jenkins method'

## Autoregressive conditional heteroskedasticity
- also called: 'ARCH'
- https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity
- applications: 'Time series analysis'
- properties: 'autoregressive'

## Probabilistic context-free grammar
- also called: 'PCFG', 'Stochastic context-free grammar', 'SCFG'
- https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar
- applications: 'Computational linguistics', 'RNA structure prediction'

-- Stochastic process

## Ornstein–Uhlenbeck process
- https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

-- physical models

## Ising model
- https://en.wikipedia.org/wiki/Ising_model
- usually solved by: 'Transfer-matrix method'
- usually samples by: 'Metropolis–Hastings algorithm'
- variants: 'ANNNI model', 'Z N model', 'Potts model'
- describes: 'Ferromagnetism'

## n-vector model
- https://en.wikipedia.org/wiki/N-vector_model
- generalization of: 'Ising model', 'Classical XY model', 'Classical Heisenberg model'

## Kuramoto model
- also called: 'Kuramoto–Daido model'
- https://en.wikipedia.org/wiki/Kuramoto_model
- describes: 'Synchronization'

## Neighbour-sensing model
- paper: 'Simulating colonial growth of fungi with the Neighbour-Sensing model of hyphal growth' (2004)
- https://en.wikipedia.org/wiki/Neighbour-sensing_model
- describes: 'Morphogenesis of fungal hyphal networks'
- domain: 'Mycology'

## Self-propelled particles
- also called: 'SPP', 'Vicsek model'
- paper: 'Novel type of phase transition in a system of self-driven particles' (1995)
- applications: 'Swarming behaviour', 'Collective motion'
- properties: 'deterministic'

## Cebeci–Smith model
- https://en.wikipedia.org/wiki/Cebeci%E2%80%93Smith_model
- applications: 'Computational fluid dynamics'

## Baldwin–Lomax model
- https://en.wikipedia.org/wiki/Baldwin%E2%80%93Lomax_model
- applications: 'Computational fluid dynamics'

## van Genuchten–Gupta model
- https://en.wikipedia.org/wiki/Van_Genuchten%E2%80%93Gupta_model
- applications: 'Crop yield', 'Soil salinity'
- domain: 'Agriculture'

## Maas–Hoffman model
- https://en.wikipedia.org/wiki/Maas%E2%80%93Hoffman_model
- applications: 'Crop yield', 'Soil salinity'
- domain: 'Agriculture'

## t-J model
- https://en.wikipedia.org/wiki/T-J_model
- describes: 'High-temperature superconductivity'

## Black–Scholes model
- https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
- describes: 'Financial market'

## Cramér–Lundberg model
- also called: 'Classical compound-Poisson risk model', 'Classical risk process', 'Poisson risk process'
- https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Lundberg_model
- describes: 'Ruin theory'
