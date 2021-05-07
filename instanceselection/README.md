
# Extended pre-processing pipeline for text classification: On the role of meta-feature representations, sparsification and selective sampling

This repository contains an **Unofficial** Python 3 implementation of Cover method proposed on **Compression-Based Selective Sampling for Learning to Rank** and extended as selective sampling phase of Extended pre-processing pipeline for text classification.

## Original Citation

```
@inproceedings{10.1145/2983323.2983813,
author = {Silva, Rodrigo M. and Gomes, Guilherme C.M. and Alvim, M\'{a}rio S. and Gon\c{c}alves, Marcos A.},
title = {Compression-Based Selective Sampling for Learning to Rank},
year = {2016},
isbn = {9781450340731},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/2983323.2983813},
doi = {10.1145/2983323.2983813},
booktitle = {Proceedings of the 25th ACM International on Conference on Information and Knowledge Management},
pages = {247–256},
numpages = {10},
keywords = {selective sampling, active learning, compression, learning to rank},
location = {Indianapolis, Indiana, USA},
series = {CIKM '16}
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
