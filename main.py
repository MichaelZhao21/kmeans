import os
import urllib.request
import zipfile
from model.kmeans import KMeans
import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/static/public/438/health+news+in+twitter.zip"
PATH = "data/Health-Tweets/"
K_VALS = [5, 10, 15, 20, 25, 50, 100, 250, 1000]

def download_data():
    # Check if dataset exists
    if not os.path.exists("data"):
        # Download dataset and unzip it
        with urllib.request.urlopen(DATASET_URL) as response:
            with open("data.zip", "wb") as f:
                f.write(response.read())

        # Unzip the dataset
        with zipfile.ZipFile("data.zip", "r") as zip_ref:
            zip_ref.extractall("data")


def main():
    # Download data and create model
    download_data()
    model = KMeans(PATH + "cnnhealth.txt")

    # Run tests
    cluster_counts = []
    errs = []
    for k in K_VALS:
        cc, err = model.cluster(k)
        cluster_counts.append('\n'.join([f'{i+1}: {v} tweets' for i, v in enumerate(cc)]))
        errs.append(err)

    # Tabulate
    data = {'K': K_VALS, 'SSE': errs, 'Size of each cluster': cluster_counts}
    df = pd.DataFrame(data)

    df.to_csv("output.csv", index=False)

    # Print info
    print("===== Saved Output to output.csv =====")
    print(df)
    print("=====================================\n")

    # Create a table from the df
    down = df.to_markdown()
    with open("output.md", "w") as f:
        f.write(down)


if __name__ == "__main__":
    main()
