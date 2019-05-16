# Hashes

## Fowler–Noll–Vo hash function
- also called: 'FNV-1a hash'
- https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function

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

## Universal sentence encoder
- paper: 'Universal Sentence Encoder' (2018)
- applications: 'semantic hashing'
- machine learning
- input: 'text'

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
- uses: 'Machine learning', 'Vector quantization'

## Chromaprint
- input: 'Waveform'
- https://github.com/acoustid/chromaprint
- https://oxygene.sk/2011/01/how-does-chromaprint-work/
- based on: 'Computer Vision for Music Identification' (2005), 'Pairwise Boosted Audio Fingerprint' (2009)
- applications: 'Acoustic fingerprint'

## PhotoDNA
- paper: 'Robust Image Hashing' (2000)
- https://www.microsoft.com/en-us/photodna
- input: 'image'

## Invariant moments perceptual hash
- paper: 'Perceptual Hashing for Color Images Using Invariant Moments' (2011)
- input: 'image'
- metric: 'L2 norm'

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
