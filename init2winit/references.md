# References

## Datasets

### CIFAR-10 and CIFAR-100
```
@techreport{cifarKrizhevsky09learningmultiple,
    author = {Alex Krizhevsky},
    title = {Learning multiple layers of features from tiny images},
    institution = {University of Toronto},
    year = {2009}
}
```

### ImageNet
```
@article{imagenet,
  Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
  Title = {{ImageNet Large Scale Visual Recognition Challenge}},
  Year = {2015},
  journal = {International Journal of Computer Vision (IJCV)},
  volume = {115},
  number = {3},
  pages = {211-252}
}
```

### MNIST
```
@article{mnist,
  author = {LeCun, Yann and Cortes, Corinna},
  title = {{MNIST} handwritten digit database},
  url = {http://yann.lecun.com/exdb/mnist/},
  year = 2010
}
```

### Fashion MNIST

```
@article{fashionMNIST,
  author       = {Han Xiao and Kashif Rasul and Roland Vollgraf},
  title        = {Fashion-{MNIST}: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
  date         = {2017-08-28},
  year         = {2017},
  journal   = {CoRR},
  volume    = {cs.LG/1708.07747},
  url       = {http://arxiv.org/abs/1708.07747},
  primaryClass  = {cs.LG},
}
```

### SVHN

```
@inproceedings{svhn,
  author = {Netzer, Yuval and Wang, Tao and Coates, Adam and Bissacco, Alessandro and Wu, Bo and Ng, Andrew Y.},
  title = {Reading Digits in Natural Images with Unsupervised Feature Learning},
  booktitle = {NIPS Workshop on Deep Learning and Unsupervised Feature Learning},
  url = {http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf},
  year = {2011},
}
```

### LM1B

```
@article{lm1b,
  author    = {Ciprian Chelba and
               Tomas Mikolov and
               Mike Schuster and
               Qi Ge and
               Thorsten Brants and
               Phillipp Koehn},
  title     = {One Billion Word Benchmark for Measuring Progress in Statistical Language
               Modeling},
  journal   = {CoRR},
  volume    = {abs/1312.3005},
  year      = {2013},
  url       = {http://arxiv.org/abs/1312.3005},
  archivePrefix = {arXiv},
  eprint    = {1312.3005},
}
```

### WMT

See the various WMT translation datasets in TFDS, e.g.
[WMT15](https://www.tensorflow.org/datasets/catalog/wmt15_translate).
As of early 2021, <http://www.statmt.org/wmt20/> should also have a lot of
relevant information and links to the workshops for different years.

### Uniref50

The first paper describes the original data and the second paper describes the
machine learning dataset derived from it.

```
@article{uniref50bio,
    author = {Suzek, Baris E. and Wang, Yuqi and Huang, Hongzhan and McGarvey, Peter B. and Wu, Cathy H. and the UniProt Consortium},
    title = {{UniRef} clusters: a comprehensive and scalable alternative for improving sequence similarity searches},
    journal = {Bioinformatics},
    volume = {31},
    number = {6},
    pages = {926-932},
    year = {2014},
    month = {11},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btu739},
    url = {https://doi.org/10.1093/bioinformatics/btu739},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/31/6/926/569379/btu739.pdf},
}

@article {uniref50ml,
	author = {Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Myle and Zitnick, C. Lawrence and Ma, Jerry and Fergus, Rob},
	title = {Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences},
	elocation-id = {622803},
	year = {2020},
	doi = {10.1101/622803},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/12/15/622803},
	eprint = {https://www.biorxiv.org/content/early/2020/12/15/622803.full.pdf},
	journal = {bioRxiv}
}
```

## Models

### Residual Networks
```
@misc{resnet,
  title={Deep residual learning for image recognition. CoRR abs/1512.03385 (2015)},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  year={2015}
}
```

### Wide ResNet-*-*
```
@article{zagoruyko2016wide,
  title={Wide residual networks},
  author={Zagoruyko, Sergey and Komodakis, Nikos},
  journal={arXiv preprint arXiv:1605.07146},
  year={2016}
}
```
