# general import
import time
import itertools
import numpy as np
from sklearn.cluster import KMeans

# import from cobras_ts
from cobras_ts.cluster import Cluster
from cobras_ts.clustering import Clustering
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.querier.commandlinequerier import CommandLineQuerier

# XCobrasExplainer
from model_explainer import ClusteringExplainer


class XCOBRAS_kmeans(COBRAS_kmeans):
    def __init__(self, 
                 X=None, 
                 budget=10, 
                 model_explainer=ClusteringExplainer() , 
                 querier=None, 
                 one_versus_all = True):
        """ Constructor of the "XCOBRAS_kmeans", extends  class "COBRAS_kmeans"

        Args:
            budget (int, optional): _description_. Defaults to 10.
        """
        self.budget = budget
        self.fitted = False
        self.model_explainer = model_explainer
        self.number_query_GT = 0
        self.arret_gt = False
        self.one_versus_all = one_versus_all
        

    def fit(self, X, feature_names=None, y=CommandLineQuerier(), store_intermediate_results=True):
        """Function that mimics the sklearn "fit" function.
        TODO... compléter après l'avoir terminée

        Args:
            X (np.array): dataset of size (nb_samples, nb_features)
            y (Querier object (from cobras_ts), optional): object that answers the must-link or cannot_link questions. Defaults to CommandLineQuerier.

        Returns:
            - a :class:`~clustering.Clustering` object representing the resulting clustering
            - a list of intermediate clustering labellings for each query (each item is a list of clustering labels)
            - a list of timestamps for each query
            - the list of must-link constraints that was queried
            - the list of cannot-link constraints that was queried
        """
        # calling the super class' constructor
        super().__init__(
            data = X, 
            querier = y, 
            max_questions  = self.budget,
            store_intermediate_results = store_intermediate_results
        )
        self.feature_names = feature_names
        # performs clustering
        self.fitted = True
        return self.cluster()

    # Override this method to include explanations
    def determine_split_level(self, superinstance, clustering_to_store):
        """ Determine the splitting level for the given super-instance using a small amount of queries

        For each query that is posed during the execution of this method the given clustering_to_store is stored as an intermediate result.
        The provided clustering_to_store should be the last valid clustering that is available

        :return: the splitting level k
        :rtype: int
        """
        
        
        # need to make a 'deep copy' here, we will split this one a few times just to determine an appropriate splitting
        # level
        si = self.create_superinstance(superinstance.indices)

        must_link_found = False
        # the maximum splitting level is the number of instances in the superinstance
        max_split = len(si.indices)
        split_level = 0
        while not must_link_found and len(self.ml) + len(self.cl) < self.max_questions:
            if len(si.indices) == 2:
                # if the superinstance that is being splitted just contains 2 elements split it in 2 superinstances with just 1 instance
                new_si = [self.create_superinstance([si.indices[0]]), self.create_superinstance([si.indices[1]])]
            else:
                # otherwise use k-means to split it
                new_si = self.split_superinstance(si, 2)

            if len(new_si) == 1:
                # we cannot split any further along this branch, we reached the splitting level
                split_level = max([split_level, 1])
                split_n = 2 ** int(split_level)
                return min(max_split, split_n)

            s1 = new_si[0]
            s2 = new_si[1]
            
            pt1 = min([s1.representative_idx, s2.representative_idx])
            pt2 = max([s1.representative_idx, s2.representative_idx])

            ############################################################################################################
            ##### YACINE #####
            # Avant de demander à l'utilisateur ce qu'il veut répondre:
            # 1. EXPLAIN-IT -> générer des features importances des classes + des deux instances
            # 2. Les montrer (où et comment ?)
            # 3. Y répondre
            # 
            # ****Mettre ça ici pour le moment, voir si on utiliserait pas un autre  "querier" plutot****
            ##################

            # Construct the labeling ?
            # cas où on prend 1 vs all
            y_hat = np.array(clustering_to_store) 
            explanations = [None, None]

            if self.one_versus_all:
                # convert points of the same cluster to 1, others to 0
                current_label = clustering_to_store[pt1]
                mask = y_hat == current_label

                y_hat[~mask] = 0 # label 0 others 
                y_hat[mask]  = 1 # label 1
            
            if len(y_hat[y_hat == 1]) != len(clustering_to_store):
                # meaning number of class > 1
                self.arret_gt= True
                shap_values = self.model_explainer.fit_explain(self.data, y_hat, [pt1, pt2], feature_names=self.feature_names)
                explanations[0] = shap_values[0]
                explanations[1] = shap_values[1]
            else: # gt
                if self.arret_gt :
                    print("It's not supposed to print this message")
                else:
                    self.number_query_GT +=1
            ############################################################################################################

            
            
            if self.querier.query_points(pt1, pt2, explanations[0], explanations[1]):
                self.ml.append((pt1, pt2))
                must_link_found = True
                if self.store_intermediate_results:
                    self.intermediate_results.append(
                        (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))
                continue
            else:
                self.cl.append((pt1, pt2))
                split_level += 1
                if self.store_intermediate_results:
                    self.intermediate_results.append(
                        (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))

            si_to_choose = []
            if len(s1.train_indices) >= 2:
                si_to_choose.append(s1)
            if len(s2.train_indices) >= 2:
                si_to_choose.append(s2)

            if len(si_to_choose) == 0:
                split_level = max([split_level, 1])
                split_n = 2 ** int(split_level)
                return min(max_split, split_n)

            si = min(si_to_choose, key=lambda x: len(x.indices))

        split_level = max([split_level, 1])
        split_n = 2 ** int(split_level)
        return min(max_split, split_n)        
        

    def get_all_SICM(self):
        """getter that gets all the :
        SI (super instances) - C(centroids) - M(mapping function: SI -> Cluster)
        TODO peut-être essayer d'optimiser ça (passer moins de temps à tout reconstruire après chaque étape)
        Returns:
            np.array: array of all the super instances of size (nb_si)
            np.array: array of all their corresponding "centroid" of size (nb_si, nb_features)
            list:     mapping function: list : argument(centroid) -> associated cluster
        """
        if self.fit == None:
            ...
        
        all_clusters = self.clustering.clusters

        # lists to store all the current* SI and 
        #  TODO peut-être essayer d'optimiser ça (passer moins de temps à tout reconstruire après chaque étape)
        # 
        all_super_instances = []
        map_si_to_cluster = []

        for ci, cluster in enumerate(all_clusters):
            temp_si = cluster.super_instances
            all_super_instances+=temp_si
            for i in range(len(temp_si)):
                map_si_to_cluster.append(ci)
 
        all_super_instances = np.array(all_super_instances)
        all_centroids = np.array([si.centroid for si in all_super_instances])
        return all_super_instances, all_centroids, map_si_to_cluster
        
    def predict(self, X):
        """
        Function that mimics the "predict" function of any other sklearn model.
        Returns the "label" (here, cluster) of each data.

        Args:
            X (np.array): dataset of size (nb_samples, nb_features)

        Returns:
            np.array: array of the associated labels of size (nb_samples,)
        """
        
        # TODO changer ça si on prend encompte la "sauvegarde" ou "construction ittérative"
        _, all_centroids, map_si_to_cluster  = self.get_all_SICM()
        
        # ---- GET THE CLOSEST SUPER INSTANCE
        # Use sklearn.KMeans to get the closest super instance (faster)
        k = KMeans(n_clusters=all_centroids.shape[0], max_iter=1,n_init=1)
        k.cluster_centers_ = all_centroids
        # cannot call the predict function of kmeans without at least one fitting iteration
        k.fit(all_centroids)
        # to make sure the indices were not swapped
        k.cluster_centers_ = all_centroids

        # ---- PREDICT THE LABELS
        #    -  kmeans has "nb_super_instances" centroids
        #    -  in reality, several super instances maps to the same cluster
        KMeans_labels = k.predict(X)
        COBRAS_labels = np.array([map_si_to_cluster[i] for i in  KMeans_labels])

        # Returns the clustering labels 
        return COBRAS_labels
    
    def get_cluster_and_all_super_instances(self, super_instance):
        """Function that looks for the super instances leading to the same cluster.
        Objective: Look for all the partitions that are refering to the same cluster. 

        Args:
            super_instance (cobras_ts.superinstance_kmeans.SuperInstance_kmeans): a super instance

        Returns:
            dict: The key is of type:   cobras_ts.cluster.Cluster
                The value is of type: list(cobras_ts.superinstance_kmeans.SuperInstance_kmeans) representing the same cluster
        """
        my_dict = self.clustering.get_cluster_to_generalized_super_instance_map()
        return [{k:list(itertools.chain.from_iterable(v))} for k, v in my_dict.items() if [super_instance] in v][0]

    def score(self, X, y):
        # TODO :)
        pass