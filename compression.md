# Compression

-- algorithms

## Dynamic Markov compression
- also called: 'DMC'
- paper: 'Data Compression using Dynamic Markov Modelling' (1987)
- https://en.wikipedia.org/wiki/Dynamic_Markov_compression
- applications: 'Lossless compression'

## Prediction by partial matching
- also called: 'PPM'
- https://en.wikipedia.org/wiki/Prediction_by_partial_matching
- applications: 'Lossless compression', 'Language modelling'

## PPMd
- also called: 'Partial Matching by Dmitry'
- variant of: 'Prediction by partial matching'
- implemented in (applications): '7-Zip', 'PKZIP'
- applications: 'Lossless text compression'

## Asymmetric numeral systems
- also called: 'ANS'
- paper: 'The use of asymmetric numeral systems as an accurate replacement for Huffman coding' (2015)
- https://en.wikipedia.org/wiki/Asymmetric_numeral_systems#tANS
- form of: 'Entropy encoding'
- applications: 'Lossless compression'

## tANS
- also called: 'Tabled Asymmetric numeral systems'
- https://en.wikipedia.org/wiki/Asymmetric_numeral_systems#Tabled_variant_(tANS)
- variant of: 'Asymmetric numeral systems'

## Tunstall coding
- thesis: 'Synthesis of noiseless compression codes' (1967)
- https://en.wikipedia.org/wiki/Tunstall_coding
- precursor to: 'Lempel-Ziv'

## Golomb coding
- paper: 'Run-length encodings' (1966)
- https://en.wikipedia.org/wiki/Golomb_coding
- applications: 'Lossless compression'

## Rice coding
- https://en.wikipedia.org/wiki/Golomb_coding#Rice_coding
- https://unix4lyfe.org/rice-coding/
- applications: 'Lossless compression'
- specialized form of: 'Golomb coding'

## LZ77
- also called: 'LZ1'
- applications: 'Lossless compression'
- form of: 'Dictionary coder'

## LZ78
- also called: 'LZ2'
- applications: 'Lossless compression'
- form of: 'Dictionary coder'

## DEFLATE
- https://en.wikipedia.org/wiki/DEFLATE
- based on: 'LZ77', 'Huffman coding'
- implemented in (libraries): 'zlib'
- implemented in (applications): 'PKZIP', 'gzip'
- applications: 'Lossless compression'
- file formats: 'Zip', 'gzip', 'PNG'
- RFC: 1951

## Lempel–Ziv–Welch
- also called: 'LZW'
- paper: 'A Technique for High-Performance Data Compression' (1984)
- https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch
- based on: 'LZ78'
- implemented in (applications): 'compress'
- used by: 'GIF'

## Golomb-Rice coding
- https://en.wikipedia.org/wiki/Golomb_coding
- applications: 'Lossless compression'

## Block Gilbert Moore coding
- also called: 'BGMC'

## Elias delta coding
- paper: 'Universal codeword sets and representations of the integers' (1975)
- https://en.wikipedia.org/wiki/Elias_delta_coding
- is a: 'Universal code'

## Microsoft Point-to-Point Compression
- also called: 'MPPC'
- https://en.wikipedia.org/wiki/Microsoft_Point-to-Point_Compression
- applications: 'Lossless compression'
- based on: 'LZ77'
- RFC: 2118

-- lossless

## Zstandard
- https://facebook.github.io/zstd/
- https://en.wikipedia.org/wiki/Zstandard
- implemented in: 'facebook/zstd'
- RFC: 8478
- reference implementation license: 'BSD license'
- applications: 'Lossless compression'

## LZFSE
- also called: 'Lempel–Ziv Finite State Entropy'
- https://en.wikipedia.org/wiki/LZFSE
- uses: 'LZ77', 'tANS'
- implemented in: 'lzfse/lzfse'
- reference implementation license: 'BSD license'
- applications: 'Lossless compression'

## bzip2
- https://en.wikipedia.org/wiki/Bzip2
- uses: 'Burrows–Wheeler transform', 'Move-to-front transform', 'Huffman coding', 'Run-length encoding', 'Elias delta coding'
- file formats: 'bzip2'
- applications: 'Lossless compression'

## Brotli
- https://en.wikipedia.org/wiki/Brotli
- uses: 'LZ77', 'Huffman coding'
- implemented in: 'google/brotli'
- applications: 'HTTP compression'

## Huffyuv
- also called: 'HuffYUV'
- https://en.wikipedia.org/wiki/Huffyuv
- similar: 'Lossless jpeg'
- input: 'video'
- implemented in: 'Huffyuv', 'FFmpeg'
- supported colorspaces: 'RGB24', 'RGB32', 'RGBA', 'YUY2', 'YV12'
- reference implementation license: 'GNU General Public License'
- applications: 'Lossless video compression'

## Lagarith
- https://lags.leetcode.net/codec.html
- https://en.wikipedia.org/wiki/Lagarith
- based on: 'Huffyuv'
- uses: 'Run Length Encoding', 'Arithmetic compression'
- implemented in: 'Lagarith'
- reference implementation license: 'GNU General Public License'
- applications: 'Lossless video compression'

## MSU Lossless Video Codec
- https://en.wikipedia.org/wiki/MSU_Lossless_Video_Codec
- applications: 'Lossless video compression'

## YULS
- also called: 'YUVsoft's Lossless Video Codec'
- applications: 'Lossless video compression'

## Lossless H.264
- https://en.wikipedia.org/wiki/H.264/MPEG-4_AVC
- uses: 'Context-adaptive binary arithmetic coding'
- implemented in (applications): 'x264'
- applications: 'Lossless video compression'

## UCL
- http://www.oberhumer.com/opensource/ucl/
- implemented by (applications): 'UPX'
- applications: 'Lossless compression'

-- lossy

## SILK
- also called: 'SILK Speech Codec'
- https://en.wikipedia.org/wiki/SILK
- uses: 'Linear prediction'
- basis for: 'Opus'

## CELT
- https://en.wikipedia.org/wiki/CELT
- applications: 'Lossy audio compression'
- goals: 'low-latency'
- uses: 'Modified discrete cosine transform', 'Pyramid Vector Quantization'
- implemented in: 'libcelt'
- reference implementation license: '2-clause BSD'

## Opus
- https://en.wikipedia.org/wiki/Opus_(audio_format)
- based on: 'SILK', 'CELT'
- RFC: 6716
- implemented in (libraries): 'libopus'
- reference implementation license: 'New BSD License'

## Audio Lossless Coding
- also called: 'MPEG-4 Audio Lossless Coding', 'MPEG-4 ALS'
- https://en.wikipedia.org/wiki/Audio_Lossless_Coding
- uses: 'Linear prediction', 'Golomb-Rice coding', 'Block Gilbert Moore coding'
- applications: 'Lossless audio compression'

## FLAC
- https://en.wikipedia.org/wiki/FLAC
- uses: 'Linear prediction', 'Golomb-Rice coding', 'Run-length encoding'
- implemented in (libraries): 'libFLAC'
- reference implementation license: 'BSD License'
- goals: 'DRM free'
- applications: 'Lossless audio compression'

## JPEG
- https://en.wikipedia.org/wiki/JPEG
- uses: 'Discrete cosine transform', 'Run-length encoding', 'Huffman coding'
- implemented in (libraries): 'libjpeg', 'Guetzli'
- file formats: 'JIF', 'JFIF', 'Exif'
- properties: 'lossless operations on blocks'
- applications: 'Lossy image compression'

## H.264
- also called: 'MPEG-4 AVC'
- https://en.wikipedia.org/wiki/H.264/MPEG-4_AVC#Derived_formats
- uses: 'Context-adaptive binary arithmetic coding', 'Hadamard transform'
- properties: 'patented'
- implemented in (applications): 'x264'
- implemented in (libraries): 'OpenH264'
- applications: 'Lossy video compression'
