# Metrics, Distances, Measures, Similarities, Losses and Costs

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

## Cosine similarity
- https://en.wikipedia.org/wiki/Cosine_similarity
- is not: 'metric'
- applications: 'natural language processing', 'data mining'
- metric version: 'angular distance'

## Logistic loss
- also called: 'Log loss', 'Cross entropy loss'
- https://en.wikipedia.org/wiki/Cross_entropy
- applications: 'Deep learning', 'Logistic regression'
- implemented in: 'Python sklearn.metrics.log_loss'
- properties: 'Convex', 'Continuous'

## Root-mean-square deviation
- also called: 'root-mean-square error', 'RMSD', 'RMSE'
- https://en.wikipedia.org/wiki/Root-mean-square_deviation
- implemented in: 'Python skimage.measure.compare_nrmse'
- properties: 'Convex', 'Continuous'

## Mean squared error
- also called: 'mean squared deviation', 'MSD', 'MSE'
- https://en.wikipedia.org/wiki/Mean_squared_error
- applications: 'Statistical model', 'Linear regression'
- implemented in: 'Python sklearn.metrics.mean_squared_error, tf.metrics.mean_squared_error, skimage.measure.compare_mse'
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

## Algebraic distance
- https://en.wikipedia.org/wiki/Distance#Algebraic_distance
- http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FISHER/ALGDIST/alg.htm
- applications: 'Computer vision'
- properties: 'Linear'

## Chamfer distance
- paper: 'Sequential Operations in Digital Picture Processing (1966)'
- applications: 'Computer vision', 'Image similarity'
- approximate: 'Euclidean distance'

## Hausdorff distance
- https://en.wikipedia.org/wiki/Hausdorff_distance
- domain: 'Set theory'
- applications: 'Computer vision'

## Mahalanobis distance
- https://en.wikipedia.org/wiki/Mahalanobis_distance
- applications: 'Cluster analysis', 'Anomaly detection'

## q-gram distance
- for example defined in paper: 'Approximate string-matching with q-grams and maximal matches'

## Structural similarity
- also called: 'SSIM', 'Structural similarity index'
- paper: 'Image quality assessment: from error visibility to structural similarity (2004)'
- https://en.wikipedia.org/wiki/Structural_similarity
- implemented in: 'Python skimage.measure.compare_ssim'
- properties: 'full reference'
- corresponding distance: 'Structural Dissimilarity' (not a metric)
- applications: 'Video quality evaluation', 'Image Quality Assessment'
- domain: 'Image processing'

## IW-SSIM
- also called: 'Information Content Weighted Structural Similarity Index'
- paper: 'Information Content Weighting for Perceptual Image Quality Assessment' (2010)

## Visual Information Fidelity
- also called: 'VIF'
- https://en.wikipedia.org/wiki/Visual_Information_Fidelity
- properties: 'full reference'
- applications: 'Video quality evaluation', 'Image Quality Assessment'
- domain: 'Image processing'

## Video Multimethod Assessment Fusion
- also called: 'VMAF'
- properties: 'full reference'
- applications: 'Video quality evaluation', 'Image Quality Assessment'
- domain: 'Image processing'

## Variation of information
- https://en.wikipedia.org/wiki/Variation_of_information
- is a: 'metric'
- domain: 'Probability theory', 'Information theory'

## Jensen-Shannon distance
- https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
- is a: 'metric'
- domain: 'Probability theory', 'Information theory'
- applications: 'Bioinformatics', 'Machine learning'

## Peak signal-to-noise ratio
- also called: 'PSNR'
- https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
- implemented in: 'Python skimage.measure.compare_psnr'

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

## HyTER
- paper: 'HyTER: Meaning-Equivalent Semantics for Translation Evaluation' (2012)
- applications: 'Machine translation'

## ROUGE
- also called: 'Recall-Oriented Understudy for Gisting Evaluation'
- https://en.wikipedia.org/wiki/ROUGE_(metric)
- applications: 'Machine translation', 'Automatic summarization'

## PGD-IL metric (custom name)
- paper: 'Objective Quality Assessment for Image Retargeting Based on Perceptual Geometric Distortion and Information Loss' (2014)
- applications: 'Image retargeting'

## Łukaszyk–Karmowski metric
- https://en.wikipedia.org/wiki/%C5%81ukaszyk%E2%80%93Karmowski_metric
- is not: 'metric'

## Adjusted Rand index
- also called: 'ARI'
- https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index
- implemented in (libraries): 'sklearn.metrics.adjusted_rand_score'
- applications: 'External evaluation of cluster analysis'

## Mutual Information
- also called: 'MI'
- implemented in (libraries): 'sklearn.metrics.mutual_info_score'
- applications: 'External evaluation of cluster analysis'

## Normalized Mutual Information
- also called: 'NMI'
- https://en.wikipedia.org/wiki/Mutual_information#Normalized_variants
- implemented in (libraries): 'sklearn.metrics.normalized_mutual_info_score'
- applications: 'External evaluation of cluster analysis'

## Adjusted Mutual Information
- also called: 'AMI'
- https://en.wikipedia.org/wiki/Adjusted_mutual_information
- implemented in (libraries): 'sklearn.metrics.adjusted_mutual_info_score'
- applications: 'External evaluation of cluster analysis'

## Fowlkes–Mallows index
- https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index
- applications: 'External evaluation of cluster analysis'

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
