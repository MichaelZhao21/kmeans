import re
import numpy as np
from itertools import chain
import random

# Iterative part should not run more than 10 times
MAX_ITER = 10

class KMeans:
    def __init__(self, filename, limit=-1):
        self.filename = filename
        self.limit = limit

        self._no_sp_char = re.compile("[^a-zA-Z\s@]")

        # Load data from file
        lines = self._read_file(filename)

        # If there is a limit set, remove that many lines
        if limit != -1:
            lines = lines[:limit]

        # Clean input
        self._clean_input(lines)
        print("Loaded", filename, "\n")

        # Calculate distance matrix
        print("Calculating Jaccard distance between every pair of tweets (this may take a while)...")
        self._calc_closest()
        print("\tFinished!\n")

    def _read_file(self, filename: str) -> list[str]:
        with open(filename, "r") as file:
            return file.readlines()

    def _clean_entry(self, raw: str) -> str:
        """Cleans the given entry, removing unneeded characters and changing it into a list of strings"""
        split_raw = self._no_sp_char.sub("", raw).strip().lower().split()
        return [
            s
            for s in split_raw
            if "@" not in s and not s.startswith("http") and s != "rt"
        ]

    def _clean_input(self, lines: list[str]):
        """Cleans the input and creates a numerical index for the text"""
        # Clean up the data (turn into a 2d array, where each entry is a row)
        self.raw_text = np.array([line.split("|")[2] for line in lines])
        self.raw_text = [self._clean_entry(t) for t in self.raw_text]

        # Create an index with all unique characters
        self.items = sorted(list(set(chain(*self.raw_text))))
        self.index = {it: i for i, it in enumerate(self.items)}

        # Convert text to indexed sets
        t2i = lambda line: set([self.index[s] for s in line])
        self.text = [t2i(t) for t in self.raw_text]

        # n = # of tweets
        self.n = len(self.text)
    
    def _jaccard_distance(self, a: set, b: set) -> float:
        """Calculates the Jaccard distance between 2 input sets"""
        u = len(a.union(b))
        if u == 0:
            return 1
        return 1 - len(a.intersection(b)) / u

    def _calc_closest(self):
        self.closest = []
        self.distances = np.full((self.n, self.n), 0.)
        self.no_same = []
        for i, t in enumerate(self.text):
            dists = [self._jaccard_distance(t, s) for s in self.text]

            # Sort by distance, performing a stable sort to have index secondary ordering
            self.closest.append(np.argsort(np.array(dists), kind='stable'))
            self.distances[i] = dists

            # If tweet is completely different from everything else, add it to the no_same arr
            # This is bc its jaccard distance is 1 for everything except itself
            if sum(dists) == self.n-1:
                self.no_same.append(i)

    def _find_closest(self, item: int, centers: list[int]) -> int:
        for c in self.closest[item]:
            if c in centers:
                return centers.index(c)
        return -1


    def _find_new_center(self, cluster: set[int]) -> int:
        min = self.n
        min_num = 0

        # Find min distance
        for c in cluster:
            mask = np.full(self.n, False)
            mask[list(cluster)] = True
            d = np.sum(self.distances[c], where=mask)
            if d < min:
                min = d
                min_num = c
        return min_num
    
    def _calc_sse(self, clusters: list[set[int]], centers: list[int]) -> float:
        sse = 0
        for i, c in enumerate(clusters):
            cluster_dist = self.distances[centers[i]][list(c)]
            sse += np.sum([d**2 for d in cluster_dist])
        return sse

    def cluster(self, K: int) -> tuple[list[int], float]:
        """Runs the K-means clustering algorithm, returning the cluster counts and SSE error"""
        print(f'========== Running the K-means clustering for k={K} ==========')
        # Generate initial centers
        centers = []
        for _ in range(K):
            # Loop until we can get a rand number TT
            while True:
                c = random.randint(0, len(self.text)-1)

                # Make sure it's not in the list
                if c in centers:
                    continue

                # Make sure its not the exact same as anything else
                uhoh = False
                for ce in centers:
                    if self.text[ce] == self.text[c]:
                        uhoh = True
                        break
                if uhoh:
                    continue

                # Otherwise, add it to centers list
                centers.append(c)
                break

        print("Initial centers:", centers, "\n")
        
        # List of sets, where each set represents the tweets in that index's cluster
        clusters = np.array([set() for _ in range(K)])

        # Loop until the centers don't change (iterative part!!)
        for round in range(MAX_ITER):
            # Assign every tweet to its center
            for i in range(self.n):
                idx = self._find_closest(i, centers)
                if idx == -1:
                    idx = random.randint(0, K)
                clusters[idx].add(i)

            # Print clusters
            print(f'Clusters for iter {round}:', [{centers[i]: len(c)} for i, c in enumerate(clusters)])

            # Find new centers for each cluster
            new_centers = []
            for c in clusters:
                new_centers.append(self._find_new_center(c))
            print("\tNew centers:", new_centers)

            # Compare old and new centers
            if centers == new_centers:
                print(f'Converged after {round} iterations! Exiting...')
                break

            # Assign new centers
            centers = new_centers

            # Reach max rounds = BAD
            if round == MAX_ITER - 1:
                print("ERROR: Reached max iterations without converging!!")
                break

            # Reset clusters
            for c in clusters:
                c.clear()
    
        # Calculate error
        sse = self._calc_sse(clusters, centers)
        print("\nSSE:", sse, "\n============================================================\n")

        # Extract the clusters into a list
        cluster_counts = [len(s) for s in clusters]

        return cluster_counts, sse
