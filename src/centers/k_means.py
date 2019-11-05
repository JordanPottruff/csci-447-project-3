# k_means.py
# Implementation of the K-Means algorithm for finding clusters.
import random as rand
import src.util as util

# The smallest percentage change in distortion that will NOT stop the centroid calculation.
DISTORTION_CHANGE_CUTOFF = 0.001


# The K-Means algorithm implementation. This will find the k clusters in the training data and then store the class
# distribution of the clusters to use for a nearest neighbor search done by 'run'.
class KMeans:

    # Creates an instance of the algorithm and instantly begins clustering. This way, the run method can be used as
    # soon as the object is created.
    def __init__(self, training_data, k):
        self.training_data = training_data.copy()
        self.k = k
        # The centroids and cluster_classes list are used to represent the clusters themselves. centroids[i] is the
        # centroid for the ith cluster, and cluster_classes[i] is a map of each class to the probability it occurs in
        # the ith cluster. Note that any classes with 0% probabilities (no training data in the cluster had that class)
        # are not included in the dictionary. Lastly, we also store the distortion of the final clusters.
        self.centroids, self.clusters, self.cluster_classes, self.distortion = self.calculate_clusters()

    # Returns the centroids representing each cluster and the  distribution of classes within each cluster. Both values
    # are lists, where index i in either list represents the ith cluster. The class distributions are in the form of a
    # dictionary that maps each class to the frequency of the class in the cluster.
    def calculate_clusters(self):
        # First, generate centroids randomly
        centroids = self.generate_random_centroids()
        clusters = []

        distortion = None
        distortion_change = None
        # The cluster calculation will continue until we have "converged", which is measured by having a smaller
        # percentage change in distortion between two cycles than DISTORTION_CHANGE_CUTOFF.
        while distortion_change is None or abs(distortion_change) >= DISTORTION_CHANGE_CUTOFF:
            # Reset the clusters to have 0 observations.
            clusters = [[] for i in range(self.k)]

            # Assign each observation to the cluster corresponding to the closest centroid.
            for obs in self.training_data.get_data():
                closest_centroid_i = self.find_closest_centroid(obs, centroids)
                clusters[closest_centroid_i].append(obs)

            # With all observations assigned to a cluster, we now recalculate the centroids as the cluster means.
            for i in range(self.k):
                if len(clusters[i]) != 0:
                    centroids[i] = self.calculate_cluster_mean(clusters[i])

            # Update distortion variables to determine whether to exit the loop now.
            new_distortion = self.calculate_distortion(clusters, centroids)
            if distortion is not None:
                # Percentage change in distortion from the previous set of clusters to these new clusters:
                distortion_change = (new_distortion - distortion) / distortion
            distortion = new_distortion

        # Finally, instead of returning each cluster itself, we are only interested in knowing the class distributions
        # in each cluster.
        cluster_classes = [{} for i in range(len(clusters))]
        for i in range(len(clusters)):
            cluster_classes[i] = util.calculate_class_distribution(clusters[i], self.training_data.class_col)

        return centroids, clusters, cluster_classes, distortion

    # Creates a list of centroids that are "random." Returns the centroids as a 2D list.
    def generate_random_centroids(self):
        centroids = [[] for i in range(self.k)]
        # Initialize the means using randomly selected attribute values.
        for i in range(self.k):
            centroids[i] = self.generate_random_centroid()
        return centroids

    # Creates a single random centroid. A centroid is really just a vector of attribute values, so we accomplish this
    # by randomly selecting an attribute value from our training data.
    def generate_random_centroid(self):
        num_cols = len(self.training_data.get_data()[0])
        num_training_data = len(self.training_data.get_data())

        # Initialize the centroid with None values for all columns
        centroid = [None for i in range(num_cols)]
        # For the attribute columns, assign a value picked randomly from the training set.
        for attr_col in self.training_data.attr_cols:
            # Random pick from the training set:
            selected_obs = self.training_data.get_data()[rand.randrange(0, num_training_data)]
            # Assigning that pick's value for the current attribute:
            centroid[attr_col] = selected_obs[attr_col]
        return centroid

    # Out of the list of all centroids, this will return the corresponding index for the centroid that is closest to the
    # given observation.
    def find_closest_centroid(self, obs, centroids):
        # Index of the closest centroid seen so far.
        closest_centroid_i = None
        # Distance to the closest centroid seen so far.
        min_dist = None
        for i in range(len(centroids)):
            dist = self.training_data.distance(centroids[i], obs)
            if closest_centroid_i is None or dist < min_dist:
                closest_centroid_i = i
                min_dist = dist
        return closest_centroid_i

    # Calculates the mean of the cluster. This is tricky because not all attributes are numeric -- some are strings.
    # When we encounter a string attribute, we simply calculate the mode (using a frequency dictionary) and use that
    # as the "mean".
    def calculate_cluster_mean(self, cluster):
        num_cols = len(self.training_data.get_data()[0])
        attr_cols = self.training_data.attr_cols
        str_attr_cols = self.training_data.get_str_attr_cols()

        # If the cluster has to items in it, we return the mean as a random coordinate.
        if len(cluster) == 0:
            return self.generate_random_centroid()

        # The sums and freqs lists are used for calculating the average and mode of numeric and string attributes,
        # respectively. Both have the same number of columns as our observations, but only the columns corresponding
        # to string (freqs[i]) or numeric (sums[i]) will be filled in.
        sums = [0 for i in range(num_cols)]
        freqs = [{} for i in range(num_cols)]

        # This section does one of two things:
        # (1) calculates the sum of numeric attributes (and stores it in the corresponding columns of 'sums'),
        # (2) calculates the frequency of values in string attributes (and stores it in a dictionary in the freqs list).
        for obs in cluster:
            # We iterate over the attribute columns only (to ignore class and unused columns)
            for attr_col in attr_cols:
                # If our attribute column is string-valued...
                if attr_col in str_attr_cols:
                    # ...then update the frequency table by...
                    str_val = obs[attr_col]
                    if str_val in freqs[attr_col]:
                        # ...incrementing an existing count, if present, or
                        freqs[attr_col][str_val] += 1
                    else:
                        # ...setting the count equal to 1 if value not previously seen.
                        freqs[attr_col][str_val] = 1
                else:
                    # Otherwise, if our column is numeric, we just continue building a sum for that column.
                    sums[attr_col] += obs[attr_col]

        # We now calculate the "means", which is an average for numeric columns and the mode for string columns.
        mean = [0 for i in range(num_cols)]
        # We again want to just iterate over the attribute columns instead of all the columns
        for attr_col in attr_cols:
            # If our attribute column is string-valued...
            if attr_col in str_attr_cols:
                # ...we find the most frequent value...
                freq = freqs[attr_col]
                max_attr_val = None
                for attr_val in freq:
                    if max_attr_val is None or freq[attr_val] > freq[max_attr_val]:
                        max_attr_val = attr_val
                # ...and set the mean for this column to be that value.
                mean[attr_col] = max_attr_val
            else:
                # Otherwise, if our column is numeric, we divide the sum by the size of the cluster to get an average.
                mean[attr_col] = sums[attr_col] / len(cluster)

        # Lastly, we return this mean.
        return mean

    # Calculates the distortion of the given clusters using the training data. Essentially, distortion is a measure of
    # how far each point is from the cluster it belongs to. We use this to optimize our centroid placement.
    def calculate_distortion(self, clusters, centroids):
        distortion = 0
        for cluster_i in range(len(clusters)):
            for obs in clusters[cluster_i]:
                centroid = centroids[cluster_i]
                distortion += self.training_data.distance(obs, centroid)
        return distortion

    # Classifies an example by finding the nearest cluster and then returning the class distribution for that cluster.
    def run(self, example):
        closest_centroid_i = self.find_closest_centroid(example, self.centroids)
        return self.cluster_classes[closest_centroid_i]
