# NLP

## Unicode Text Segmentation algorithm
- paper: 'Unicode Standard Annex #29'
- implemented in: 'org.apache.lucene.analysis.standard.StandardTokenizer'
- applications: 'Segmentation'

## Kstem
- paper: 'Viewing morphology as an inference process' (1993) <https://doi.org/10.1145/160688.160718>
- http://lexicalresearch.com/kstem-doc.txt
- applications: 'Stemming'
- processed language: 'English'
- implemented in: 'org.apache.lucene.analysis.en.KStemmer'

## S-Stemmer
- paper: 'How effective is suffixing?' (1991) <https://doi.org/10.1002/(SICI)1097-4571(199101)42:1%3C7::AID-ASI2%3E3.0.CO;2-P>
- applications: 'Stemming'
- processed language: 'English'
- implemented in: 'org.apache.lucene.analysis.en.EnglishMinimalStemmer'

## Lovins stemming algorithm
- paper: 'Development of a Stemming Algorithm' (1968)
- applications: 'Stemming'
- processed language: 'English'
- implemented in: 'Snowball'

## Porter-Stemmer
- paper: 'An algorithm for suffix stripping' (1980) <https://doi.org/10.1108/eb046814>
- https://tartarus.org/martin/PorterStemmer/
- applications: 'Stemming'
- processed language: 'English'
- implemented in (libraries): 'Lucene', 'nltk.stem.porter.PorterStemmer'

## Lancaster-Stemmer
- paper: 'Another stemmer' (1990) <https://doi.org/10.1145/101306.101310>
- applications: 'Stemming'
- implemented in (libraries): 'nltk.stem.lancaster.LancasterStemmer'

## Snowball-Stemmer
- applications: 'Stemming'
- processed language: various
- implemented in (libraries): 'Lucene', 'nltk.stem.snowball.SnowballStemmer'
