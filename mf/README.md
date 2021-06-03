# Extended pre-processing pipeline for text classification: On the role of meta-feature representations, sparsification and selective sampling

This repository contains an **Unofficial** Python 3 implementation of Metafeatures method proposed on **A Thorough Evaluation of Distance-Based Meta-Features for Automated Text Classification** and extended as meta-feature representation phase of Extended pre-processing pipeline for text classification.

## Acknowledgment

I would like to acknowledge to Sergio Canuto for sharing the original code of [error-based](errbased.py) Metafeatures. Thank you.

## Original Citation

```
@ARTICLE{8326556,
  author={Canuto, Sérgio and Sousa, Daniel Xavier and Gonçalves, Marcos André and Rosa, Thierson Couto},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={A Thorough Evaluation of Distance-Based Meta-Features for Automated Text Classification}, 
  year={2018},
  volume={30},
  number={12},
  pages={2242-2256},
  doi={10.1109/TKDE.2018.2820051}
}
```
## Extended pre-processing pipeline for text classification Citation.

```
@article{cunha20,
title = {Extended pre-processing pipeline for text classification: On the role of meta-feature representations, sparsification and selective sampling},
journal = {Information Processing & Management},
volume = {57},
number = {4},
pages = {102263},
year = {2020},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2020.102263},
url = {https://www.sciencedirect.com/science/article/pii/S030645731931461X},
author = {Washington Cunha and Sérgio Canuto and Felipe Viegas and Thiago Salles and Christian Gomes and Vitor Mangaravite and Elaine Resende and Thierson Rosa and Marcos André Gonçalves and Leonardo Rocha}
}
``` 

## Installing

Clone this repository in your machine and execute the installation with pip.

### Requirements

This project is based on ```python==3.6```. The dependencies are as follow:
```
Cython==0.29.23
joblib==1.0.1
mlpack3==3.0.2.post1
numpy==1.17.4
pandas==1.1.5
python-dateutil==2.8.1
pytz==2021.1
scikit-learn==0.21.3
scipy==1.5.4
six==1.16.0
tqdm==4.60.0
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Washington Cunha** - *Initial work* - [waashk](https://gitlab.com/waashk)

See also the list of [contributors](https://gitlab.com/waashk/pipeline/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
