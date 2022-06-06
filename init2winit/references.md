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
```
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

### fastMRI

Information and access information are available at
[fastMRI](https://fastmri.org/)'s webpage. Of note, the data input pipeline
we replicate consists of the knee single-coil challenge only.

```
@article{doi:10.1148/ryai.2020190007,
  author = {Knoll, Florian and Zbontar, Jure and Sriram, Anuroop and Muckley, Matthew J. and Bruno, Mary and Defazio, Aaron and Parente, Marc and Geras, Krzysztof J. and Katsnelson, Joe and Chandarana, Hersh and Zhang, Zizhao and Drozdzalv, Michal and Romero, Adriana and Rabbat, Michael and Vincent, Pascal and Pinkerton, James and Wang, Duo and Yakubova, Nafissa and Owens, Erich and Zitnick, C. Lawrence and Recht, Michael P. and Sodickson, Daniel K. and Lui, Yvonne W.},
  title = {fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning},
  journal = {Radiology: Artificial Intelligence},
  volume = {2},
  number = {1},
  pages = {e190007},
  year = {2020},
  doi = {10.1148/ryai.2020190007},
  note = {PMID: 32076662},
  URL = { https://doi.org/10.1148/ryai.2020190007 },
  eprint = { https://doi.org/10.1148/ryai.2020190007 }
}

@inproceedings{zbontar2018fastMRI,
  title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
  author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Tullie Murrell and Zhengnan Huang and Matthew J. Muckley and Aaron Defazio and Ruben Stern and Patricia Johnson and Mary Bruno and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and Nafissa Yakubova and James Pinkerton and Duo Wang and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1811.08839},
  year={2018}
}
```


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

### Vision Transformers (ViT)
```
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={ICLR},
  year={2021}
}

@article{steiner2021train,
  title={How to train your ViT? data, augmentation, and regularization in vision transformers},
  author={Steiner, Andreas and Kolesnikov, Alexander and Zhai, Xiaohua and Wightman, Ross and Uszkoreit, Jakob and Beyer, Lucas},
  journal={arXiv preprint arXiv:2106.10270},
  year={2021}
}
```

### U-Net
We implement a U-Net replicating the code provided [here](https://github.com/facebookresearch/fastMRI/blob/main/fastmri_examples/unet/train_unet_demo.py)
as part of the fastMRI workload cited below.
```
@inproceedings{zbontar2018fastMRI,
    title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
    author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Tullie Murrell and Zhengnan Huang and Matthew J. Muckley and Aaron Defazio and Ruben Stern and Patricia Johnson and Mary Bruno and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and Nafissa Yakubova and James Pinkerton and Duo Wang and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1811.08839},
    year={2018}
}
```

Below is the citation for the original U-Net paper.
```
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical image computing and computer-assisted intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```

### Graph network
```
@article{graphnetwork,
title = {Relational inductive biases, deep learning, and graph networks},
author  = {Peter Battaglia and Jessica Blake Chandler Hamrick and Victor Bapst and Alvaro Sanchez and Vinicius Zambaldi and Mateusz Malinowski and Andrea Tacchetti and David Raposo and Adam Santoro and Ryan Faulkner and Caglar Gulcehre and Francis Song and Andy Ballard and Justin Gilmer and George E. Dahl and Ashish Vaswani and Kelsey Allen and Charles Nash and Victoria Jayne Langston and Chris Dyer and Nicolas Heess and Daan Wierstra and Pushmeet Kohli and Matt Botvinick and Oriol Vinyals and Yujia Li and Razvan Pascanu},
year  = {2018},
URL = {https://arxiv.org/pdf/1806.01261.pdf},
journal = {arXiv}
}
```

```
@inproceedings{
xu2018how,
title={How Powerful are Graph Neural Networks?},
author={Keyulu Xu and Weihua Hu and Jure Leskovec and Stefanie Jegelka},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=ryGs6iA5Km},
}
```