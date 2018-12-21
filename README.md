# NASBench: A Neural Architecture Search Dataset and Benchmark

This repository contains the code used for generating and interacting with the
NASBench dataset. The dataset contains **423,624 unique neural networks**
exhaustively generated and evaluated from a fixed graph-based search space. More
information about the search space will be presented in our upcoming paper.

Each network is trained and evaluated multiple times on CIFAR-10 and we present
the metrics in a queriable API. The current release contains over **1.27
million** trained and evaluated models.

## Setup

1.  Clone this repo.

```
git clone https://github.com/google-research/nasbench
cd nasbench
```

2. (optional) Create a virtualenv for this library.

```
virtualenv venv
source venv/bin/activate
```

3. Install the project along with dependencies.

```
pip install -e .
```

**Note:** the only required dependency is TensorFlow. The above instructions
will install the CPU version of TensorFlow to the virtualenv. For other install
options, see https://www.tensorflow.org/install/.

## Download the dataset

**This project is currently still work-in-progress and the dataset is not yet
available for download.**

## Using the dataset

Example usage (see `example.py` for a full runnable example):

```python
# Load the data from file (this will take some time)
dataset = api.NASBench('/path/to/nasbench.tfrecord')

# Create an Inception-like module (5x5 convolution replaced with two 3x3
# convolutions).
model_spec = api.ModelSpec(
    # Adjacency matrix of the module
    matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
            [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
            [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
            [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
            [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
            [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
            [0, 0, 0, 0, 0, 0, 0]],   # output layer
    # Operations at the vertices of the module, matches order of matrix
    ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

# Query this model from dataset, returns a dictionary containing the metrics
# associated with this model.
data = dataset.query(model_spec)
```

See `nasbench/api.py` for more information, including the constraints on valid
module matrices and operations.

## How the dataset was generated

The dataset generation code is provided for reference, but the dataset has
already been fully generated.

The list of unique computation graphs evaluated in this dataset was generated
via `nasbench/scripts/generate_graphs.py`. Each of these graphs was evaluated
multiple times via `nasbench/scripts/run_evaluation.py`.

## How to run the unit tests

Unit tests are included for some of the algorithmically complex parts of the
code. The tests can be run directly via Python. Example:

```
python nasbench/tests/model_builder_test.py
```

## Disclaimer

This is not an official Google product.
