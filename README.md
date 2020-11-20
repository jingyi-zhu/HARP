This repo handles the black-box image attack problem following the metric and network structure in [Carlini & Wagner](http://dx.doi.org/10.1109/SP.2017.49). We propose to feed second-order approximation into the covariance matrix for the random perturbation (dubbed as HARP), and the corresponding presentation slides/video can be found at [conf1](https://www.abstractsonline.com/pp8/#!/9022/presentation/8993) and [conf2](https://opt-ml.org/papers.html). 


Environment setup:
```
conda create -n python35 python=3.5
conda activate python35
pip install -r requirement.txt
```

Run universal-image-attack:
```
python main.py
```
