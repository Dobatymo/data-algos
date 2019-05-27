# Hashes

## Fowler–Noll–Vo hash function
- also called: 'FNV-1a hash'
- https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
- applications: 'hash table', 'checksum'
- properties: 'non-cryptographic'

## MurmurHash
- https://en.wikipedia.org/wiki/MurmurHash
- applications: 'hash table'
- properties: 'non-cryptographic'

## SpookyHash
- https://en.wikipedia.org/wiki/Jenkins_hash_function#SpookyHash
- http://www.burtleburtle.net/bob/hash/spooky.html
- bits: 128
- properties: 'non-cryptographic'

## CityHash
- https://github.com/google/cityhash
- properties: 'non-cryptographic'
- applications: 'hash table'

## FarmHash
- https://github.com/google/farmhash
- properties: 'non-cryptographic'
- successor of: 'CityHash'

## Meow hash
- https://github.com/cmuratori/meow_hash
- fast for large inputs
- properties: 'non-cryptographic'

## Metro hash
- https://github.com/jandrewrogers/MetroHash
- http://www.jandrewrogers.com/2015/05/27/metrohash/
- fast for small inputs
- properties: 'non-cryptographic'

## xxHash
- https://github.com/Cyan4973/xxHash
- properties: 'non-cryptographic'

# Cryptographic hashes

## MD5
- https://en.wikipedia.org/wiki/MD5
- bits: '128'
- properties: 'cryptographically broken'
- superseeded by: 'SHA-1'

## SHA-1
- also called: 'Secure Hash Algorithm 1'
- https://en.wikipedia.org/wiki/SHA-1
- bits: '160'
- properties: 'cryptographically insecure'
- superseeded by: 'SHA-2'

## SHA-2
- https://en.wikipedia.org/wiki/SHA-2
- bits: '224', '256', '384', '512'
- superseeded by: 'SHA-3'

## SHA-3
- also called: 'Keccak algorithm'
- https://en.wikipedia.org/wiki/SHA-3

# Semantic hashes / Perceptual hashes / Approximate Hash Based Matching (AHBM) / Fuzzy Hashing / Fingerprinting

## Smooth inverse frequency
- also called: 'SIF', 'Smoothed inverse frequency'
- paper: 'A Simple but Tough-to-Beat Baseline for Sentence Embeddings' (2016)
- input: 'text'
- requires: 'Machine learning'

## Unsupervised smoothed inverse frequency
- also called: 'uSIF'
- paper: 'Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline' (2018)
- poster: http://www.cs.toronto.edu/~kawin/acl2018_usif_poster.pdf
- requires: 'Machine learning'

## Universal sentence encoder
- paper: 'Universal Sentence Encoder' (2018)
- applications: 'semantic hashing'
- requires: 'Machine learning'
- input: 'text'
- metric: 'Cosine distance'

##
- paper: 'Comprehensive feature-based robust video fingerprinting using tensor model' (2016)
- applications: 'Content-based near-duplicate video detection'
- input: 'video'

## Centroid of Gradient Orientations
- also called: 'CGO'
- paper: 'Video Fingerprinting Based on Centroids of Gradient Orientations' (2006)
- paper: 'Robust video fingerprinting for content-based video identification' (2007)
- applications: 'Content-based near-duplicate video detection'
- metric: 'Squared Euclidean distance'
- input: 'video'

## Shazam algorithm
- paper: 'An Industrial-Strength Audio Search Algorithm' (2003)
- input: 'Waveform'

## Philips audio fingerprinting algorithm
- paper: 'A Highly Robust Audio Fingerprinting System' (2002)
- input: 'Waveform'
- metric: 'Hamming distance'?

## Now Playing (Neural Network Fingerprinter)
- paper: 'Now Playing: Continuous low-power music recognition' (2017)
- uses: 'Vector quantization'
- requires: 'Machine learning'

## Chromaprint
- input: 'Waveform'
- https://github.com/acoustid/chromaprint
- https://oxygene.sk/2011/01/how-does-chromaprint-work/
- based on: 'Computer Vision for Music Identification' (2005), 'Pairwise Boosted Audio Fingerprint' (2009)
- applications: 'Acoustic fingerprint'

## Average hash
- also called: 'aHash'
- http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
- implemented in: 'Python ImageHash', 'cv::img_hash::AverageHash'

## Difference hash
- also called: 'dHash'
- http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
- implemented in: 'Python dhash', 'Python ImageHash'
- metric: 'Hamming distance'
- variant: 'Median Hash'

## Perception hash
- also called: 'pHash', 'Perceptual hash'
- http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
- implemented in: 'Python ImageHash', 'cv::img_hash::PHash'
- based on: 'Discrete Cosine Transformation'

## Wavelet hash
- also called: 'wHash'
- https://fullstackml.com/2016/07/02/wavelet-image-hash-in-python/
- implemented in: 'Python ImageHash'
- based on: 'Discrete Wavelet Transformation'

## PhotoDNA
- paper: 'Robust Image Hashing' (2000)
- https://www.microsoft.com/en-us/photodna
- input: 'image'

## Invariant moments perceptual hash
- paper: 'Perceptual Hashing for Color Images Using Invariant Moments' (2011)
- implemented in: 'cv::img_hash::ColorMomentHash', 'pHash Pro'
- input: 'image'
- metric: 'L2 norm'

## Block mean hash
- also called: 'Blockhash'
- paper: 'Block Mean Value Based Image Perceptual Hashing' (2006)
- implemented in: 'cv::img_hash::BlockMeanHash', 'commonsmachinery/blockhash-python', 'pHash Pro'
- input: 'image'
- metric: 'Bit Error Rate'

## Marr-Hildreth Operator based hash
- paper: 'Implementation and benchmarking of perceptual image hash functions' (2010)
- implemented in: 'cv::img_hash::MarrHildrethHash', 'pHash::ph_mh_imagehash'
- metric: 'Bit Error Rate'
- input: 'image'

## Radial hASH
- also called: 'RASH', 'Radial variance based hash'
- paper: 'Robust image hashing based on radial variance of pixels' (2005)
- implemented in: 'cv::img_hash::RadialVarianceHash', 'pHash::ph_image_digest'
- based on: 'Radon transform'
- input: 'image'
- metric: 'Peak of Cross Correlation'

## DCT based hash
- is this the same as: 'PHash'?
- paper: 'Robust video hash extraction' (2004)
- implemented in: 'pHash::ph_dct_imagehash'
- metric: 'Hamming distance'
- input: 'image'

## Compact Fourier Mellin Transform (CFMT)-based hash
- paper: 'Duplicate Image Detection in Large Scale Databases' (2007)
- metric: 'L1 distance', 'L2 distance' (preferred)
- input: 'image'

## MinHash
- paper: 'On the resemblance and containment of documents' (1997)
- https://en.wikipedia.org/wiki/MinHash
- properties: 'probabilistic'
- applications: 'Locality-sensitive hashing', 'Set similarity', 'data mining', 'bioinformatics', 'clustering'
- implemented in: 'ekzhu/datasketch'
- approximates: 'Jaccard similarity'

## SimHash
- paper: 'Similarity Estimation Techniques from Rounding Algorithms'
- https://en.wikipedia.org/wiki/SimHash
- metric: 'Hamming distance'
- applications: 'Locality-sensitive hashing'
