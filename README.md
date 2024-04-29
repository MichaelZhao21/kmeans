# KMeans

A K-means classifier from scratch!

## Dataset

Using the news health tweets dataset: https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter

## Installation and Setup

The main runner code is in `main.py` while the model code is placed in `model/kmeans.py`.

To install libraries, create a virtual environment and install the required libraries with:

```bash
# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Install libraries
python3 -m pip install -r requirements.txt
# or
pip install -r requirements.txt
```

The `main.py` file has some configuration variables at the top, namely the dataset link and the K values used.

Run the model and tests with:

```bash
python3 main.py
```

You will be presented with the iterations for each K, their centroids, and the errors. The CSV of the output will be saved to output.csv and the table will be rendered as a Markdown table, stored in output.md.

