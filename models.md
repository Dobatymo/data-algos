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
- implemented in (libraries): 'Python gensim.models.ldamodel.LdaModel', 'tomotopy.LDAModel'
- commonly sampled by: 'variational inference', 'Collapsed Gibbs sampling'
- properties: 'generative', 'stochastic'

## Supervised latent Dirichlet allocation
- also called: 'sLDA'
- paper: 'Supervised Topic Models' (2008)
- implemented in (libraries): 'tomotopy.SLDAModel'
- is a: 'Topic model'
- variant of: 'Latent Dirichlet allocation'

## Labeled LDA
- paper: 'Labeled LDA: a supervised topic model for credit attribution in multi-labeled corpora' (2009)
- variant of: 'Latent Dirichlet allocation'

## Partially Labeled Dirichlet Allocation
- also called: 'PLDA'
- paper: 'Partially Labeled Topic Models for Interpretable Text Mining' (2011) <https://doi.org/10.1145/2020408.2020481>
- variant of: 'Latent Dirichlet allocation'
- implemented in: 'tomotopy::PLDAModel'

## Partially Labeled Dirichlet Process
- also called: 'PLDP'
- paper: 'Partially Labeled Topic Models for Interpretable Text Mining' (2011)
- implemented in: 'soberqian/TopicModel4J'

## SeededLDA
- also called: 'Guided LDA'?
- paper: 'Incorporating Lexical Priors into Topic Models' (2012)
- variant of: 'Latent Dirichlet allocation'
- implemented in: 'vi3k6i5/GuidedLDA'

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
- is a: 'Topic model'

## Hierarchical Latent Tree Analysis
- also called: 'HLTA'
- paper: 'Hierarchical Latent Tree Analysis for Topic Detection' (2014)
- implemented in: 'kmpoon/hlta'
- is a: 'Topic model', 'Probabilistic graphical model'

## Correlated topic model
- also called: 'CTM'
- paper: 'Correlated Topic Models' (2006)
- is a: 'Topic model'

## Dirichlet-multinomial regression
- also called: 'DMR'
- paper: 'Topic models conditioned on arbitrary features with Dirichlet-multinomial regression' (2012)
- implemented in (libraries): 'Python tomotopy.DMRModel'
- is a: 'Topic model'

-- uncategorized models

## Maximum entropy spectral estimation
- https://en.wikipedia.org/wiki/Maximum_entropy_spectral_estimation
- applications: 'Spectral density estimation'
- uses method: 'Principle of maximum entropy'
- domain: 'Information theory'

## Analysis of variance
- also called: 'ANOVA'
- https://en.wikipedia.org/wiki/Analysis_of_variance
- implemented in (libraries): 'statsmodels.stats.anova.anova_lm'

## One-way analysis of variance
- also called: 'One-way ANOVA'
- https://en.wikipedia.org/wiki/One-way_analysis_of_variance
- implemented in (libraries): 'scipy.stats.f_oneway'

## Naïve Bayes probability model
- https://en.wikipedia.org/wiki/Naive_Bayes_classifier
- special case of: 'Bayesian network'
- type: 'conditional probability model'

-- classification or regression models:

## Tensor rank decomposition
- also called: 'Canonical polyadic decomposition', 'CPD', 'PARAFAC model', 'PARallel FACtor analysis', 'CANDECOMP'
- paper: 'The Expression of a Tensor or a Polyadic as a Sum of Products' (1927)
- https://en.wikipedia.org/wiki/Tensor_rank_decomposition

## Linear prediction
- also called: 'Linear predictive coding', 'LPC'
- https://en.wikipedia.org/wiki/Linear_prediction
- applications: 'Audio compression'
- file formats: 'Shorten', 'FLAC', 'MPEG-4 ALS'
- used by standards: 'GSM'
- commonly used loss: 'Root-mean-square deviation'

## Logistic regression
- also called: 'logistic model', 'logit model'
- https://en.wikipedia.org/wiki/Logistic_regression
- properties: 'deterministic', 'discriminative'

## Conditional random field
- also called: 'CRF'
- paper: 'Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data' (2001) <https://dl.acm.org/doi/10.5555/645530.655813>
- https://en.wikipedia.org/wiki/Conditional_random_field
- properties: 'discriminative', 'undirected', 'probabilistic', 'graphical'
- applications: 'Structured prediction', 'POS tagging', 'shallow parsing', 'named entity recognition', 'object recognition'
- usually optimized by: 'L-BFGS', 'Stochastic gradient descent'
- implemented in: 'CRFSuite', 'python-crfsuite'
- generalization of: 'Linear Chain CRF'

## Linear Chain CRF
- also called: 'Linear chain conditional random field'
- implemented in: 'torch_struct.LinearChainCRF'
- specialization of: 'CRF'
- properties: 'discriminative', 'undirected', 'probabilistic'

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

## Linear support-vector machine
- also called: 'SVM', 'Support-vector network'
- paper: 'A training algorithm for optimal margin classifiers' (1992) <https://doi.org/10.1145/130385.130401>
- paper: 'Support-vector networks' (1995) <https://doi.org/10.1007/BF00994018>
- https://en.wikipedia.org/wiki/Support-vector_machine
- Quadratic Programming problem
- usually optimized by: 'Coordinate descent', 'newGLMNET'
- properties: 'deterministic', 'discriminative'
- solves: 'Classification'
- is a: 'linear classifier'
- non-linear variant: 'Support-vector machine (using kernel method)'
- implemented in (libraries): 'LIBLINEAR', 'LIBSVM', 'sklearn.svm.LinearSVC', 'sklearn.svm.SVC'

## Support-vector machine (using kernel method)
- also called: 'SVM'
- https://en.wikipedia.org/wiki/Support-vector_machine
- implemented in (libraries): 'LIBSVM', 'sklearn.svm.SVC'
- usually optimized by: 'SMO-type decomposition method'
- solves: 'Classification'

## Linear support-vector regression
- also called: 'SVR'
- https://en.wikipedia.org/wiki/Support-vector_machine#Regression
- implemented in (libraries): 'LIBLINEAR', 'LIBSVM', 'sklearn.svm.SVR', 'sklearn.svm.LinearSVR'
- non-linear variant: 'Support-vector regression (using kernel method)'
- solves: 'Regression'

## Support-vector regression (using kernel method)
- also called: 'SVR'
- https://en.wikipedia.org/wiki/Support-vector_machine#Regression
- properties: 'deterministic'
- implemented in (libraries): 'LIBSVM', 'sklearn.svm.SVR'
- usually optimized by: 'SMO-type decomposition method'
- solves: 'Regression'

## Support-vector clustering
- also called: 'SVC'

## Hidden Markov model
- also called: 'HMM'
- paper: 'Statistical Inference for Probabilistic Functions of Finite State Markov Chains' (1966) <https://doi.org/10.1214/aoms/1177699147>
- https://en.wikipedia.org/wiki/Hidden_Markov_model
- properties: 'generative', 'stochastic'
- type of: 'Markov model'
- applications: 'Speech recognition', 'Natural language processing', 'Bioinformatics', 'Discrete sequence anomaly detection'
- domain: 'Machine learning'
- implemented in: 'Python hmmlearn', 'Python torch_struct.HMM'
- usually optimized by: 'Forward-backward algorithm', 'Viterbi algorithm'

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

## SRCNN
- also called: 'Super-Resolution Convolutional Neural Network'
- http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
- paper: 'Image Super-Resolution Using Deep Convolutional Networks' (2014)
- uses: 'CNN'

## FSRCNN
- paper: 'Accelerating the Super-Resolution Convolutional Neural Network' (2016)
- applications: 'Single image super-resolution', 'Luma upscaling'
- reference metric: 'mean square error'
- reference optimizer: 'Stochastic gradient descent'
- based on: 'SRCNN'

## FSRCNNX
- implemented in: 'igv/FSRCNN-TensorFlow'
- based on: 'FSRCNN'

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

## GCN
- also called: 'Graph Convolutional Network'
- paper: 'Semi-Supervised Classification with Graph Convolutional Networks' (2017)
- implemented in: 'tkipf/gcn'

## DCGCN
- also called: 'Densely Connected Graph Convolutional Network'
- paper: 'Densely Connected Graph Convolutional Networks for Graph-to-Sequence Learning' (2019)
- implemented in: 'Cartus/DCGCN'
- applications: 'AMR-to-text', 'Neural machine translation', 'graph-to-sequence learning'

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
- properties: 'autoregressive', 'linear', 'univariate'
- commonly estimated by: 'Box–Jenkins method'

## Autoregressive moving average vector model
- also called: 'ARMAV', 'VARMA', 'Vector ARMA'
- applications: 'Time series analysis'
- properties: 'autoregressive', 'multivariate'

## Autoregressive integrated moving average model
- also called: 'ARIMA'
- https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
- implemented in: 'Python statsmodels.tsa.arima.model.ARIMA'
- applications: 'Time series analysis'
- properties: 'autoregressive', 'linear', 'univariate'
- commonly estimated by: 'Box–Jenkins method'

## Autoregressive conditional heteroskedasticity
- also called: 'ARCH'
- https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity
- applications: 'Time series analysis'
- properties: 'autoregressive', 'univariate'

## Exponential smoothing
- also called: 'Holt Winter's exponential smoothing'
- implemented in: 'Python statsmodels.tsa.holtwinters.ExponentialSmoothing'
- applications: 'Time series analysis'
- properties: 'univariate'

## Vector autoregression
- also called: 'VAR'
- https://en.wikipedia.org/wiki/Vector_autoregression
- applications: 'Time series analysis'
- properties: 'autoregressive', 'multivariate'
- implemented in: 'Python statsmodels.tsa.vector_ar', 'Python pyflux.VAR', 'sktime.forecasting.var.VAR'

## Structural vector autoregression
- also called: 'Structural VAR', 'SVAR'
- https://en.wikipedia.org/wiki/Vector_autoregression#Structural_VAR
- applications: 'Time series analysis'
- properties: 'autoregressive', 'multivariate'
- implemented in: 'Python statsmodels.tsa.vector_ar.svar_model.SVAR'

## BATS
- paper: 'Forecasting Time Series With Complex Seasonal Patterns Using Exponential Smoothing' (2012) <https://doi.org/10.1198/jasa.2011.tm09771>
- implemented in: 'tbats'
- applications: 'Time series analysis'

## TBATS
- paper: 'Forecasting Time Series With Complex Seasonal Patterns Using Exponential Smoothing' (2012) <https://doi.org/10.1198/jasa.2011.tm09771>
- implemented in: 'tbats'
- applications: 'Time series analysis'

## Matrix autoregressive model
- also called: 'MAR'
- paper: 'Autoregressive models for matrix-valued time series' (2021) <https://doi.org/10.1016/j.jeconom.2020.07.015>
- applications: 'Time series analysis'
- properties: 'autoregressive', 'multivariate', 'bilinear'
- usually optimized by: 'Alternating Least Squares'
- related problems: 'Nearest Kronecker product problem'

## Probabilistic context-free grammar
- also called: 'PCFG', 'Stochastic context-free grammar', 'SCFG'
- https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar
- applications: 'Computational linguistics', 'RNA structure prediction'

## PixelRNN
- paper: 'Pixel Recurrent Neural Networks' (2016) <https://arxiv.org/abs/1601.06759>

## PixelCNN
- paper: 'Conditional image generation with PixelCNN decoders' (2016) <https://arxiv.org/abs/1606.05328>
- properties: 'generativ', 'autoregressive'
- based on: 'PixelRNN'

## WaveNet
- paper: 'WaveNet: A Generative Model for Raw Audio' (2016) <https://arxiv.org/abs/1609.03499>
- https://en.wikipedia.org/wiki/WaveNet
- type of: 'Artificial neural network'
- applications: 'Audio generation', 'Voice cloning', 'Text-to-speech'
- properties: 'generativ', 'probabilistic', 'autoregressive'
- based on: 'PixelCNN'

-- Stochastic process

## Ornstein–Uhlenbeck process
- https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

-- physical models

## Ising model
- https://en.wikipedia.org/wiki/Ising_model
- usually solved by: 'Transfer-matrix method'
- usually sampled by: 'Metropolis–Hastings algorithm'
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
