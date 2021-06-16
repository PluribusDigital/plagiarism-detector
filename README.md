# Plagiarism Detector

This repo is provides all the code from this _TBD_ blog post

When successfully run, you will get output that shows how similar some documents are to the original text and which ones are susupected of plagiarism.

```
                Group Person Task Category Native English  Knowledge  Difficulty  difference  plagiarized
g4pE_taskb.txt      4      E    b    light     non-native          3           2    0.000069         True
g0pA_taskb.txt      0      A    b      cut         native          4           3    0.000078         True
g2pE_taskb.txt      2      E    b    light     non-native          3           2    0.000174         True
g3pA_taskb.txt      3      A    b    heavy         native          3           2    0.000237         True
g0pD_taskb.txt      0      D    b    light     non-native          2           2    0.000242         True
g0pE_taskb.txt      0      E    b    heavy     non-native          2           2    0.000269         True
g2pA_taskb.txt      2      A    b    heavy         native          3           2    0.000271         True
g1pD_taskb.txt      1      D    b      cut         native          4           1    0.000279         True
g1pA_taskb.txt      1      A    b    heavy         native          4           3    0.000332         True
g0pB_taskb.txt      0      B    b      non         native          3           3    0.000340        False
g2pC_taskb.txt      2      C    b      non         native          5           3    0.000343        False
g1pB_taskb.txt      1      B    b      non         native          5           4    0.000359        False
g3pC_taskb.txt      3      C    b      non         native          4           3    0.000360        False
g2pB_taskb.txt      2      B    b      non         native          4           3    0.000362        False
g4pC_taskb.txt      4      C    b      non     non-native          2           2    0.000365        False
g3pB_taskb.txt      3      B    b      non         native          2           3    0.000372        False
g0pC_taskb.txt      0      C    b      non         native          3           3    0.000376        False
g4pD_taskb.txt      4      D    b      cut     non-native          5           1    0.000379        False
g4pB_taskb.txt      4      B    b      non     non-native          5           1    0.000390        False
```

## Installation Instructions

#### Prerequisites

1. (_optional_) [Docker](https://www.docker.com/get-started)

#### Download and install

```
git clone https://github.com/PluribusDigital/plagiarism-detector.git
cd plagiarism-detector
```

## Run in your local environment

```
<set your favorite virtual environment up>
pip install -r requirements.txt
python src/app.py {taska,taskb,taskc,taskd,taske}
```

## Run using Docker

```
docker-compose up --build
docker-compose down
```

## References

1. [A Corpus of Plagiarised Short Answers](https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html)
1. [Triangle/Sector similarity](https://github.com/taki0112/Vector_Similarity)
1. [How to Use Tfidftransformer & Tfidfvectorizer?](https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.YMo6JJNKhqs)
