# pam_nn.py
# Implementation of the PAM algorithm, with additional nearest neighbor functionality.
import random
import math
import src.util as util


# The PAM implementation. This will attempt to find k clusters in the training data, and stores the class distribution
# of points belonging to these clusters in order to perform a nearest neighbor classification later on.
class PamNN:

    # Creates an instance of the PAM algorithm. Clusters and their medoids are calculated upon object creation so that
    # the run method can be used right away.
    def __init__(self, training_data, k):
        self.training_data = training_data.copy()
        self.k = k
        self.medoids, self.clusters, self.cluster_classes, self.distortion = self.fast_calculate_medoids(k)

    def faster_calculate_medoids(self, k):
        tuple_data = []
        for data in self.training_data.data:
            tuple_data.append(tuple(data))
        medoids = random.sample(tuple_data, k)
        clusters, distortion = self.create_clusters(medoids, tuple_data)
        td = 0
        while True:
            nearest = {}
            dn = {}
            ds = {}
            x_zeros = []
            for example in tuple_data:
                if example not in medoids:
                    x_zeros.append(example)
            print(len(tuple_data))
            print(len(medoids))
            print(len(x_zeros))
            for x in x_zeros:
                (nearest[x], dn[x], ds[x]) = self.find_closest_medoids(x, medoids)
            delta_td_star = [0 for i in range(k)]
            x_star = [None for i in range(k)]
            for xj_ind in range(len(x_zeros)):
                xj = x_zeros[xj_ind]
                dj = dn[xj]
                delta_td = [-dj for i in range(k)]

                for x0 in tuple_data:
                    if x0 is xj:
                        continue
                    doj = self.training_data.distance(x0, xj)
                    if x0 not in dn:
                        (nearest[x0], dn[x0], ds[x0]) = self.find_closest_medoids(x0, medoids)
                    nx0 = nearest[x0]
                    dnx0 = dn[x0]
                    dsx0 = ds[x0]
                    delta_td[nx0] = delta_td[nx0] + min(doj, dsx0) - dnx0
                    if doj < dnx0:
                        for i in range(k):
                            if i == nx0:
                                continue
                            delta_td[i] = delta_td[i] + doj - dnx0
                for i in range(k):
                    if delta_td[i] < delta_td_star[i]:
                        delta_td_star[i] = delta_td[i]
                        x_star[i] = xj
            if min(delta_td_star) >= 0:
                break

            min_ind = self.get_min_index(delta_td_star)
            count = 0
            while delta_td_star[min_ind] < 0:
                count += 1
                temp = medoids[min_ind]
                medoids[min_ind] = x_star[min_ind]
                x_star[min_ind] = temp

                td = td + delta_td_star[min_ind]
                delta_td_star[min_ind] = 0
                for j in range(k):
                    if delta_td_star[j] >= 0:
                        continue
                    new_delta_td = 0
                    for x0 in x_zeros + [medoids[j]]:
                        doj = self.training_data.distance(x0, x_star[j])
                        if x0 not in dn:
                            (nearest[x0], dn[x0], ds[x0]) = self.find_closest_medoids(x0, medoids)
                        dnx0 = dn[x0]
                        dsx0 = ds[x0]
                        if nearest[x0] == j:
                            delta = min(doj, dsx0) - dnx0
                        else:
                            delta = min(doj - dnx0, 0)
                        new_delta_td = new_delta_td + delta
                    if new_delta_td <= delta_td_star[j]:
                        delta_td_star[j] = new_delta_td
                    else:
                        delta_td_star[j] = 0
                min_ind = self.get_min_index(delta_td_star)
            print(str(td) + " " + str(count))

        clusters, distortion = self.create_clusters(medoids, tuple_data)
        cluster_classes = []
        for cluster in clusters:
            cluster_classes.append(util.calculate_class_distribution(cluster, self.training_data.class_col))
        return medoids, clusters, cluster_classes, td

    def get_min_index(self, a_list):
        min_i = 0
        for i in range(1, len(a_list)):
            if a_list[min_i] > a_list[i]:
                min_i = i
        return min_i

    def fast_build(self, k):
        tuple_data = []
        for data in self.training_data.data:
            tuple_data.append(tuple(data))
        td = float("inf")
        m1 = None
        S = random.sample(tuple_data, 10 + math.ceil(math.sqrt(len(tuple_data))))
        for xj_ind in range(len(S)):
            xj = S[xj_ind]
            td_j = 0
            for x0_ind in range(len(S)):
                if x0_ind == xj_ind:
                    continue
                x0 = S[x0_ind]
                td_j = td_j + self.training_data.distance(x0, xj)
                if td_j < td:
                    td = td_j
                    m1 = xj
        medoids = [m1]
        for i in range(1, k):
            delta_td_star = float("inf")
            x_star = None

            S = random.sample([item for item in tuple_data if item not in medoids], math.ceil(10+math.sqrt(len(tuple_data))))
            dn = {}
            for x0 in S:
                dn[x0] = float("inf")
            for xj_ind in range(len(S)):
                xj = S[xj_ind]
                delta_td = 0
                for x0_ind in range(len(S)):
                    if x0_ind == xj_ind:
                        continue
                    x0 = S[x0_ind]
                    new_dist = self.training_data.distance(x0, medoids[i-1])
                    if new_dist < dn[x0]:
                        dn[x0] = new_dist
                    closest_dist = dn[x0]
                    dist = self.training_data.distance(x0, xj) - closest_dist
                    if dist < 0:
                        delta_td = delta_td + dist
                if delta_td < delta_td_star:
                    delta_td_star = delta_td
                    x_star = xj
            td = td + delta_td_star
            medoids.append(x_star)
        return medoids

    def fast_calculate_medoids(self, k):
        tuple_data = []
        for data in self.training_data.data:
            tuple_data.append(tuple(data))
        medoids = self.fast_build(k)
        td = 0
        while True:
            nearest = {}
            dn = {}
            ds = {}
            x_zeros = []
            for example in tuple_data:
                if example not in medoids:
                    x_zeros.append(example)
            for x in x_zeros:
                (nearest[x], dn[x], ds[x]) = self.find_closest_medoids(x, medoids)
            delta_td_star = 0
            m_star = None
            x_star = None
            for xj_ind in range(len(x_zeros)):
                xj = x_zeros[xj_ind]
                dj = dn[xj]
                delta_td = [-dj for i in range(k)]
                for x0 in tuple_data:
                    if x0 is xj:
                        continue
                    doj = self.training_data.distance(x0, xj)
                    if x0 not in dn:
                        (nearest[x0], dn[x0], ds[x0]) = self.find_closest_medoids(x0, medoids)
                    dnx0 = dn[x0]
                    dsx0 = ds[x0]
                    nx0 = nearest[x0]
                    delta_td[nx0] = delta_td[nx0] + min(doj, dsx0) - dnx0
                    if doj < dnx0:
                        for i in range(k):
                            if i == nx0:
                                continue
                            delta_td[i] = delta_td[i] + doj - dnx0
                min_td_i = 0
                for i in range(1, k):
                    if delta_td[i] < delta_td[min_td_i]:
                        min_td_i = i
                if delta_td[min_td_i] < delta_td_star:
                    delta_td_star = delta_td[min_td_i]
                    m_star = min_td_i
                    x_star = xj_ind
            if delta_td_star >= 0:
                break
            temp = medoids[m_star]
            medoids[m_star] = x_zeros[x_star]
            x_zeros[x_star] = temp

            td = td + delta_td_star
            # print(str(td) + "   " + str(delta_td_star))

        clusters, distortion = self.create_clusters(medoids, tuple_data)
        cluster_classes = []
        for cluster in clusters:
            cluster_classes.append(util.calculate_class_distribution(cluster, self.training_data.class_col))
        return medoids, clusters, cluster_classes, td

    def closest_medoid(self, example, medoids):
        min_dist = float('inf')
        for medoid in medoids:
            min_dist = min(min_dist, self.training_data.distance(medoid, example))
        return min_dist

    def find_closest_medoids(self, example, medoids):
        closest_i = -1
        closest_dist = float('inf')
        second_closest_dist = float('inf')
        for medoid_i in range(len(medoids)):
            medoid = medoids[medoid_i]
            dist = self.training_data.distance(medoid, example)
            if dist < closest_dist:
                second_closest_dist = closest_dist
                closest_dist = dist
                closest_i = medoid_i
            elif dist < second_closest_dist:
                second_closest_dist = dist
        return closest_i, closest_dist, second_closest_dist

    # The calculation of medoids, clusters, cluster class distributions, and final distortion using the PAM algorithm.
    def calculate_medoids(self, k):
        # We randomly choose our medoids to begin.
        medoids = random.sample(self.training_data.data, self.k)
        clusters, distortion = self.create_clusters(medoids, self.training_data.data)

        count = 0
        while True:
            count += 1
            if count % 1 == 0:
                print("Swap #" + str(count))
            best_distortion = distortion
            best_medoid_i = None
            best_obs = (-1, -1)
            for medoid_i in range(len(medoids)):
                for cluster_i in range(len(clusters)):
                    for obs_i in range(len(clusters[cluster_i])):
                        original_medoid = medoids[medoid_i]
                        medoids[medoid_i] = clusters[cluster_i][obs_i]
                        new_clusters, new_distortion = self.create_clusters(medoids, self.training_data.data)
                        if new_distortion < best_distortion:
                            best_distortion = new_distortion
                            best_medoid_i = medoid_i
                            best_obs = (cluster_i, obs_i)
                        medoids[medoid_i] = original_medoid
            if best_medoid_i is None:
                break
            else:
                medoids[best_medoid_i] = clusters[best_obs[0]][best_obs[1]]
                clusters, distortion = self.create_clusters(medoids, self.training_data.data)

        cluster_classes = []
        for cluster in clusters:
            cluster_classes.append(util.calculate_class_distribution(cluster, self.training_data.class_col))
        return medoids, clusters, cluster_classes, distortion

    def create_clusters(self, medoids, data):
        clusters = [[] for i in range(len(medoids))]
        distortion = 0
        for example in data:
            if example not in medoids:
                min_dist = self.training_data.distance(example, medoids[0])
                min_medoid_i = 0
                for medoid_i in range(1, len(medoids)):
                    cur_dist = self.training_data.distance(example, medoids[medoid_i])
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        min_medoid_i = medoid_i
                clusters[min_medoid_i].append(example)
                distortion += min_dist**2
        return clusters, distortion

    # Classifies the given example using the clusters found by PAM. We simply find the nearest cluster to our example
    # point and then classify it according to the probability distribution of that cluster.
    def run(self, example):
        closest_i, closest_dist, second_closest_dist = self.find_closest_medoids(example, self.medoids)
        return self.cluster_classes[closest_i]