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

### ogbg_molpcba
@inproceedings{ogbg_molpcba,
  author    = {Weihua Hu and
               Matthias Fey and
               Marinka Zitnik and
               Yuxiao Dong and
               Hongyu Ren and
               Bowen Liu and
               Michele Catasta and
               Jure Leskovec},
  editor    = {Hugo Larochelle and
               Marc Aurelio Ranzato and
               Raia Hadsell and
               Maria{-}Florina Balcan and
               Hsuan{-}Tien Lin},
  title     = {Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  booktitle = {Advances in Neural Information Processing Systems 33: Annual Conference
               on Neural Information Processing Systems 2020, NeurIPS 2020, December
               6-12, 2020, virtual},
  year      = {2020},
  url       = {https://proceedings.neurips.cc/paper/2020/hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html},
  timestamp = {Tue, 19 Jan 2021 15:57:06 +0100},
  biburl    = {https://dblp.org/rec/conf/nips/HuFZDRLCL20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


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


## Models

### Residual Networks
```

@INPROCEEDINGS{heResnet2016,
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  title={Deep Residual Learning for Image Recognition}, 
  year={2016},
  pages={770-778},
  doi={10.1109/CVPR.2016.90}
}

@misc{resnetArxiv,
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
