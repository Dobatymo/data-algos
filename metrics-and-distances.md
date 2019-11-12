# Metrics, Distances, Measures, Similarities, Losses and Costs

## Endpoint error
- also called: 'EPE', 'End-to-end point error'
- used to evaluate: 'Optical flow'

## Average angular error
- used to evaluate: 'Optical flow'

## SimRank
- https://en.wikipedia.org/wiki/SimRank
- paper: 'SimRank: a measure of structural-context similarity (2002)'
- domain: graph theory

## Levenshtein distance
- https://en.wikipedia.org/wiki/Levenshtein_distance
- is a: 'metric', 'edit distance'
- applications: 'Spelling correction', 'Sequence alignment', 'Approximate string matching', 'Linguistic distance'
- properties: 'Discrete'

## Hamming distance
- https://en.wikipedia.org/wiki/Hamming_distance
- is a: 'metric'
- applications: 'Coding theory', 'Block code', 'Error detection and correction', 'Telecommunication'
- implemented in: 'Python sklearn.metrics.hamming_loss'
- properties: 'Discrete'

## Bit Error Rate
- also called: 'Normalized hamming distance', 'Simple matching coefficient'
- https://en.wikipedia.org/wiki/Bit_error_rate

## Damerau–Levenshtein distance
- https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
- is a: 'edit distance'
- is not: 'metric'
- applications: 'Spelling correction'
- properties: 'Discrete'

## Jaro–Winkler distance
- https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance
- is a: 'edit distance'
- is not: 'metric'
- properties: 'Discrete'

## Jaccard index
- https://en.wikipedia.org/wiki/Jaccard_index
- is a: 'metric', 'edit distance'
- properties: 'Discrete'
- implemented in: 'Python sklearn.metrics.jaccard_similarity_score'

## Taxicab metric
- also called: 'rectilinear distance', 'L1 distance', 'Manhattan distance'
- https://en.wikipedia.org/wiki/Taxicab_geometry
- is a: 'metric'
- applications: 'Regression analysis', 'LASSO'

## City block distance (for images)
- is a: 'metric'
- see: 'Taxicab metric'
- paper: 'Distance functions on digital pictures '(1968)
- input: 'discrete 2d space'

## Square distance
- is a: 'metric'
- paper: 'Distance functions on digital pictures '(1968)
- input: 'discrete 2d space'

## Hexagonal distance
- is a: 'metric'
- paper: 'Distance functions on digital pictures '(1968)
- input: 'discrete 2d space'

## Octagonal distance
- is a: 'metric'
- paper: 'Distance functions on digital pictures '(1968)
- input: 'discrete 2d space'
- approximates: 'Euclidean distance'
- applications: 'Cluster detection', 'Elongated part detection', 'Regularity detection'

## Cosine similarity
- https://en.wikipedia.org/wiki/Cosine_similarity
- is not: 'metric'
- applications: 'natural language processing', 'data mining'
- metric version: 'angular distance'
- implemented in: 'sklearn.metrics.pairwise.cosine_similarity'

## Logistic loss
- also called: 'Log loss', 'Cross entropy loss'
- https://en.wikipedia.org/wiki/Cross_entropy
- applications: 'Deep learning', 'Logistic regression'
- implemented in: 'Python sklearn.metrics.log_loss'
- properties: 'Convex', 'Continuous'

## Root-mean-square deviation
- also called: 'root-mean-square error', 'RMSD', 'RMSE'
- https://en.wikipedia.org/wiki/Root-mean-square_deviation
- implemented in: 'Python skimage.measure.compare_nrmse', 'andrewekhalel/sewar'
- properties: 'Convex', 'Continuous'

## Mean squared error
- also called: 'mean squared deviation', 'MSD', 'MSE'
- https://en.wikipedia.org/wiki/Mean_squared_error
- applications: 'Statistical model', 'Linear regression'
- implemented in: 'Python sklearn.metrics.mean_squared_error, tf.metrics.mean_squared_error, skimage.measure.compare_mse', 'andrewekhalel/sewar'
- properties: 'Continuous'

## Mean absolute error
- https://en.wikipedia.org/wiki/Mean_absolute_error
- implemented in: 'Python sklearn.metrics.mean_absolute_error, tf.metrics.mean_absolute_error, tf.losses.absolute_difference'

## Hinge loss
- https://en.wikipedia.org/wiki/Hinge_loss
- applications: 'Support vector machine'
- properties: 'Convex', 'Continuous'
- implemented in: 'Python sklearn.metrics.hinge_loss'

## Explained variation
- https://en.wikipedia.org/wiki/Explained_variation
- applications: 'Regression analysis'
- implemented in: 'sklearn.metrics.explained_variance_score'

## Algebraic distance
- https://en.wikipedia.org/wiki/Distance#Algebraic_distance
- http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FISHER/ALGDIST/alg.htm
- applications: 'Computer vision'
- properties: 'Linear'

## Chamfer distance
- paper: 'Sequential Operations in Digital Picture Processing (1966)'
- applications: 'Computer vision', 'Image similarity'
- approximates: 'Euclidean distance'

## Hausdorff distance
- https://en.wikipedia.org/wiki/Hausdorff_distance
- domain: 'Set theory'
- applications: 'Computer vision'
- implemented in: 'scipy.spatial.distance.directed_hausdorff'

## Mahalanobis distance
- https://en.wikipedia.org/wiki/Mahalanobis_distance
- applications: 'Cluster analysis', 'Anomaly detection'
- implemented in: 'scipy.spatial.distance.mahalanobis'

## q-gram distance
- for example defined in paper: 'Approximate string-matching with q-grams and maximal matches'
- implemented in: 'R markvanderloo/stringdist'

## Structural similarity
- also called: 'SSIM', 'Structural similarity index'
- paper: 'Image quality assessment: from error visibility to structural similarity (2004)'
- https://en.wikipedia.org/wiki/Structural_similarity
- implemented in: 'Python skimage.measure.compare_ssim', 'tf.image.ssim', 'aizvorski/video-quality', 'sewar.full_ref.ssim'
- properties: 'full reference'
- corresponding distance: 'Structural Dissimilarity' (not a metric)
- applications: 'Video quality evaluation', 'Image Quality Assessment'
- domain: 'Image processing'

## IW-SSIM
- also called: 'Information Content Weighted Structural Similarity Index'
- paper: 'Information Content Weighting for Perceptual Image Quality Assessment' (2010)

## MS-SSIM
- also called: 'Multi-scale Structural Similarity Index'
- paper: 'Multiscale structural similarity for image quality assessment' (2003)
- implemented in: 'sewar.full_ref.msssim'
- properties: 'full reference'

## Visual Information Fidelity
- also called: 'VIF'
- paper: 'Image information and visual quality' (2006)
- https://en.wikipedia.org/wiki/Visual_Information_Fidelity
- properties: 'full reference'
- applications: 'Video quality evaluation', 'Image Quality Assessment'
- domain: 'Image processing'
- implemented in: 'aizvorski/video-quality', 'Python andrewekhalel/sewar'

## ERGAS
- also called: 'Erreur Relative Globale Adimensionnelle de Synthèse' ()
- paper: 'Quality of high resolution synthesised images: Is there a simple criterion ?' (2000)
- implemented in: 'Python andrewekhalel/sewar'
- properties: 'full reference'

## Spatial Correlation Coefficient
- also called: 'SCC'
- paper: 'A wavelet transform method to merge Landsat TM and SPOT panchromatic data' (2010)
- implemented in: 'sewar.full_ref.scc'
- properties: 'full reference'

## Relative Average Spectral Error
- also called: 'RASE'
- paper: 'Fusion of high spatial and spectral resolution images: The ARSIS concept and its implementation' (2000)
- implemented in: 'sewar.full_ref.rase'
- properties: 'full reference'

## Spectral Angle Mapper
- also called: 'SAM'
- same as: 'Cosine similarity'
- paper: 'Discrimination among semi-arid landscape endmembers using the Spectral Angle Mapper (SAM) algorithm'
- implemented in: 'sewar.full_ref.sam'
- properties: 'full reference'

## Video Multimethod Assessment Fusion
- also called: 'VMAF'
- properties: 'full reference'
- applications: 'Video quality evaluation', 'Image Quality Assessment'
- domain: 'Image processing'

## Variation of information
- also called: 'Meila's variation index', 'VI'
- https://en.wikipedia.org/wiki/Variation_of_information
- is a: 'metric', 'Information theoretic index'
- domain: 'Probability theory', 'Information theory'
- applications: 'External evaluation of cluster analysis'
- implemented in: 'josemarialuna/ExternalValidity'

## Van Dongen index
- also called: 'Van Dongen measure'
- paper: 'Performance criteria for graph clustering and Markov cluster experiments' (2000)
- is a: 'Set matching index'

## Normalized Van Dongen index
- also called: 'NVD'
- paper: 'Performance criteria for graph clustering and Markov cluster experiments' (2000)
- is a: 'Point-level index'
- uses: 'Matching' (to match clusters)

## Purity
- https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html#fig:clustfg3
- https://en.wikipedia.org/wiki/Cluster_analysis#External_evaluation
- is a: 'External validity index'
- uses: 'Matching' (to match clusters)
- implemented in: 'josemarialuna/ExternalValidity'

## F-measure
- also called: 'FM'
- uses: 'Matching' (to match clusters)
- implemented in: 'josemarialuna/ExternalValidity'

## Reverse normalized Van Dongen index
- also called: 'rNVD',
- is a: 'Set matching index'
- properties: 'symmetric'
- implemented in: 'FloFlo93/rNVD'

## Centroid index
- also called: 'CI'
- paper: 'Centroid index: Cluster level similarity measure' (2014)
- is a: 'Cluster-level similarity index', 'Cluster-level index'
- uses: 'Matching' (to match clusters)

## Centroid similarity index [20]
- also called: 'CSI'
- is a: 'Point-level index'
- uses: 'Matching' (to match clusters)

## Jensen-Shannon distance
- https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
- is a: 'metric'
- domain: 'Probability theory', 'Information theory'
- applications: 'Bioinformatics', 'Machine learning'

## Peak signal-to-noise ratio
- also called: 'PSNR'
- https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
- implemented in: 'Python skimage.measure.compare_psnr', 'aizvorski/video-quality', 'sewar.full_ref.psnr'

## Polar edge coherence
- also called: 'PEC'
- paper: 'The polar edge coherence: A quasi blind metric for video quality assessment' (2009)
- applications: 'Video quality assessment'

## Relative polar edge coherence
- also called: 'RECO'
- paper: 'The polar edge coherence: A quasi blind metric for video quality assessment' (2009)
- implemented in: 'aizvorski/video-quality'
- applications: 'Video quality assessment'

## Natural Image Quality Evaluator
- also called: 'NIQE'
- paper: 'Making a “Completely Blind” Image Quality Analyzer' (2012)
- properties: 'no-reference'
- implemented in: 'Matlab niqe', 'aizvorski/video-quality'

## Universal Quality Image Index
- also called: 'Quality Index Q', 'UQI'
- paper: 'A universal image quality index' (2002)
- applications: 'Image quality assessment'
- implemented in: 'sewar.full_ref.uqi'

## Spectral Distortion Inde
- paper: 'Multispectral and Panchromatic Data Fusion Assessment Without Reference' (2008)
- implemented in: 'sewar.no_ref.d_lambda'

## Spatial Distortion Index
- paper: 'Multispectral and Panchromatic Data Fusion Assessment Without Reference' (2008)
- implemented in: 'sewar.no_ref.d_s'

## Quality with No Reference
- also called: 'QNR'
- paper: 'Multispectral and Panchromatic Data Fusion Assessment Without Reference' (2008)
- properties: 'no-reference'
- implemented in: 'sewar.no_ref.qnr'
- applications: 'Multispectral imaging'

## Gotoh score
- paper: 'An improved algorithm for matching biological sequences' (1982)
- implemented in: 'abydos.distance.Gotoh', 'life4/textdistance'

## MLIPNS
- also called: 'Modified Language-Independent Product Name Search'
- paper: 'Using product similarity for adding business' (2010)
- implemented in: 'abydos.distance.MLIPNS'
- Phonetic distance

## Indel
- like Levenshtein, but with inserts and deletes only
- implemented in: 'abydos.distance.Indel'

## Sørensen–Dice coefficient
- https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

## Earth mover's distance
- also called: 'Wasserstein metric'
- https://en.wikipedia.org/wiki/Earth_mover%27s_distance
- solved as: 'Transportation problem'
- applications: 'Content-based image retrieval'
- domain: 'Transportation theory'
- implemented in: 'scipy.stats.wasserstein_distance'

## Word Mover's Distance
- also called: 'WMD'
- paper: 'From Word Embeddings To Document Distances (2015)'
- based on: 'Earth mover's distance', 'Word embedding'
- input: two sentences
- output: semantic distance
- time complexity (best average case): O(p^3 log p) where p is the number of unique words in the document
- cheaper variants: 'Word centroid distance' (WCD), 'Relaxed word moving distance' (RWMD)
- implemented in: 'Python gensim.models.Word2Vec.wmdistance'

## Maximum Mean Discrepancy
- also called: 'MMD'
- paper: 'A Kernel Method for the Two-Sample Problem' (2008)

## BLEU
- also called: 'Bilingual evaluation understudy'
- https://en.wikipedia.org/wiki/BLEU
- applications: 'Machine translation'
- implemented in: 'gcunhase/NLPMetrics'

## GLEU
- also called: 'Google-BLEU'
- paper: 'Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation' (2016)
- implemented in: 'gcunhase/NLPMetrics'
- applications: 'Machine translation'

## Generalized BLEU
- also called: 'GLEU', 'Generalized Language Evaluation Understanding'

## Word Error Rate
- also called: 'WER', 'Length normalized edit distance'
- https://en.wikipedia.org/wiki/Word_error_rate
- implemented in: 'gcunhase/NLPMetrics'
- applications: 'Transcription accuracy', 'Machine translation'

## Translation Edit Rate
- also called: 'TER'
- paper: 'A Study of Translation Edit Rate with Targeted Human Annotation' (2006)
- implemented in: 'gcunhase/NLPMetrics'

## NIST
- https://en.wikipedia.org/wiki/NIST_(metric)
- applications: 'Machine translation'
- based on: 'BLEU'

## METEOR
- also called: 'Metric for Evaluation of Translation with Explicit ORdering'
- paper: 'Meteor: An Automatic Metric for MT Evaluation with High Levels of Correlation with Human Judgments' (2007)
- https://en.wikipedia.org/wiki/METEOR
- applications: 'Machine translation'
- implemented in: 'gcunhase/NLPMetrics'

## HyTER
- paper: 'HyTER: Meaning-Equivalent Semantics for Translation Evaluation' (2012)
- applications: 'Machine translation'

## ROUGE
- also called: 'Recall-Oriented Understudy for Gisting Evaluation'
- paper: 'ROUGE: A Package for Automatic Evaluation of Summaries' (2004)
- https://en.wikipedia.org/wiki/ROUGE_(metric)
- applications: 'Machine translation', 'Automatic summarization'
- implemented in: 'gcunhase/NLPMetrics'

## CIDEr
- also called: 'Consensus-based Image Description Evaluation'
- paper: 'CIDEr: Consensus-based Image Description Evaluation' (2015)
- applications: 'Image caption quality'
- implemented in: 'gcunhase/NLPMetrics'

## SARI
- paper: 'Optimizing Statistical Machine Translation for Text Simplification' (2016)
- applications: 'Machine translation', 'Sentence simplification'

## PGD-IL metric (custom name)
- paper: 'Objective Quality Assessment for Image Retargeting Based on Perceptual Geometric Distortion and Information Loss' (2014)
- applications: 'Image retargeting'

## Łukaszyk–Karmowski metric
- https://en.wikipedia.org/wiki/%C5%81ukaszyk%E2%80%93Karmowski_metric
- is not: 'metric'

## Rand index
- https://en.wikipedia.org/wiki/Rand_index
- paper: 'Objective Criteria for the Evaluation of Clustering Methods' (1971)
- is a: 'Pair-counting measure', 'External validity index'
- applications: 'External evaluation of cluster analysis'

## Adjusted Rand index
- also called: 'ARI'
- paper: 'Comparing partitions' (1985)
- https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index
- implemented in (libraries): 'sklearn.metrics.adjusted_rand_score'
- implemented in: 'josemarialuna/ExternalValidity'
- applications: 'External evaluation of cluster analysis'
- is a: 'Pair-counting measure', 'External validity index'
- properties: 'corrected for chance'

## Mutual Information
- also called: 'MI'
- implemented in (libraries): 'sklearn.metrics.mutual_info_score'
- applications: 'External evaluation of cluster analysis'
- is a: 'Information theoretic index'

## Normalized Mutual Information
- also called: 'NMI'
- https://en.wikipedia.org/wiki/Mutual_information#Normalized_variants
- implemented in (libraries): 'sklearn.metrics.normalized_mutual_info_score'
- applications: 'External evaluation of cluster analysis'
- is a: 'Information theoretic index'
- same as: 'V-Measure' (for arithmetic averaging)

## V-Measure
- paper: 'V-Measure: A Conditional Entropy-Based External Cluster Evaluation Measure' (2007)
- implemented in (libraries): 'sklearn.metrics.v_measure_score', 'JuliaStats/Clustering.jl'
- applications: 'External evaluation of cluster analysis'
- is a: 'External validity index'

## Adjusted Mutual Information
- also called: 'AMI'
- https://en.wikipedia.org/wiki/Adjusted_mutual_information
- implemented in (libraries): 'sklearn.metrics.adjusted_mutual_info_score'
- applications: 'External evaluation of cluster analysis'
- is a: 'External validity index'

## Pair Sets Index
- also called: 'PSI'
- paper: 'Set Matching Measures for External Cluster Validity' (2016)
- applications: 'External evaluation of cluster analysis'
- is a: 'External validity index'
- uses: 'Hungarian algorithm', 'Optimal pairing'
- properties: 'corrected for chance'

## Simplified form of PSI
- based on: 'Pair Sets Index'
- paper: 'Set Matching Measures for External Cluster Validity' (2016)
- is a: 'metric'

## Criterion H
- also called: 'CH'
- paper: 'An Experimental Comparison of Model-Based Clustering Methods' (2001)
- is a: 'Point-level index'
- uses: 'Greedy pairing' (to match clusters)
- implemented in: 'josemarialuna/ExternalValidity'

## Centroid Ratio
- also called: 'CR'
- paper: 'Centroid Ratio for a Pairwise Random Swap Clustering Algorithm' (2014)
- uses: 'Greedy pairing' (to match clusters)

## Fowlkes–Mallows index
- https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index
- applications: 'External evaluation of cluster analysis'
- is a: 'Pair-counting measure'
- implemented in (libraries): 'sklearn.metrics.fowlkes_mallows_score'
- is a: 'External validity index'

## Silhouette value
- also called: 'Silhouette coefficient'
- https://en.wikipedia.org/wiki/Silhouette_(clustering)
- implemented in (libraries): 'sklearn.metrics.silhouette_score'
- applications: 'Internal evaluation of cluster analysis'

## Calinski–Harabasz index
- also called: 'Variance ratio criterion'
- paper: 'A dendrite method for cluster analysis' (1972)
- implemented in (libraries): 'sklearn.metrics.calinski_harabasz_score'

## Consensus score
- paper: 'FABIA: factor analysis for bicluster acquisition' (2010)
- implemented in (libraries): 'sklearn.metrics.consensus_score'
- applications: 'Internal evaluation of bicluster analysis'

-- Norms and measures

## Shannon entropy
- paper: 'A mathematical theory of communication' (1948)
- https://en.wikipedia.org/wiki/Entropy_(information_theory)
- implemented in: 'Python skimage.measure.shannon_entropy'
- is a: 'Expected value'

## Euclidean norm
- https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm

## Taxicab norm
- also called: 'Manhattan norm'
- https://en.wikipedia.org/wiki/Norm_(mathematics)#Taxicab_norm_or_Manhattan_norm
