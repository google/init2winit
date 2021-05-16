# init2winit

A Jax/Flax codebase for running deterministic, scalable, and well-documented deep learning experiments, with a particular emphasis on neural network initialization, optimization, and tuning experiments.

There is not yet a stable version (nor an official release of this library).
All APIs are subject to change.

This is a research project, not an official Google product.


## Installation
To install the latest development version inside a virtual environment, run

```
python3 -m venv env-i2w
source env-i2w/bin/activate
pip install --upgrade pip
pip install "git+https://github.com/google/init2winit.git#egg=init2winit"
pip install --upgrade jax jaxlib==0.1.66+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

where `cuda111` corresponds to the installed version of CUDA. For more Jax install information see the [Jax README](https://github.com/google/jax#installation).

## Usage

An example MNIST experiment can be run with the following command:

```sh
python3 main.py \
    --experiment_dir=/tmp/test_mnist \
    --model=fully_connected \
    --dataset=mnist \
    --num_train_steps=10
```

For local debugging we recommend using the `fake` dataset:

```sh
python3 main.py \
    --experiment_dir=/tmp/test_fake \
    --num_train_steps=10 \
    --dataset=fake \
    --hparam_overrides='{"input_shape": [28, 28, 1], "output_shape": [10]}'
```

The `hparam_overrides` accepts a serialized JSON object with hyperparameter names/values to use. See the flags in `main.py` for more information on possible configurations.

See the [`dataset_lib`](https://github.com/google/init2winit/tree/master/init2winit/dataset_lib) and [`model_lib`](https://github.com/google/init2winit/tree/master/init2winit/model_lib) directories for currently implemented datasets and models.


## Citing
To cite this repository:

```
@software{init2winit2021github,
  author = {Justin M. Gilmer and George E. Dahl and Zachary Nado},
  title = {{init2winit}: a JAX codebase for initialization, optimization, and tuning research},
  url = {http://github.com/google/init2winit},
  version = {0.0.1},
  year = {2021},
}
```


## Contributors
Contributors (past and present):

- Ankush Garg
- Behrooz Ghorbani
- Cheolmin Kim
- David Cardoze
- George E. Dahl
- Justin M. Gilmer
- Michal Badura
- Sneha Kudugunta
- Varun Godbole
- Zachary Nado

