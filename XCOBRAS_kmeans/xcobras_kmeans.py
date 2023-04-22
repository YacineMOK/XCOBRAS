# general import
import numpy as np
from sklearn.cluster import KMeans
# import from cobras_ts
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.querier.commandlinequerier import CommandLineQuerier


class XCOBRAS_kmeans(COBRAS_kmeans):
    def __init__(self, budget=10, X=None, querier=None):
        """ Constructor of the "XCOBRAS_kmeans", extends  class "COBRAS_kmeans"

        Args:
            budget (int, optional): _description_. Defaults to 10.
        """
        self.budget = budget
        self.fitted = False
        self.clustering = None
        

    def fit(self, X, y=CommandLineQuerier(), store_intermediate_results=True):
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

        # performs clustering
        self.fitted = True
        return super().cluster()


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

        Args:
            X (np.array): dataset of size (nb_samples, nb_features)

        Returns:
            np.array: array of the associated labels of size (nb_samples,)
        """
        
        # TODO changer ça si on prend encompte le 
        _, all_centroids, map_si_to_cluster  = self.get_all_SICM()
        
        # does nothing. does one TODO mieux le dire
        k = KMeans(n_clusters=all_centroids.shape[0], max_iter=1,n_init=1)
        k.cluster_centers_ = all_centroids
        # cannot call the predict function of kmeans otherwise
        k.fit(all_centroids)
        # to make sure the indices were not swapped
        k.cluster_centers_ = all_centroids

        # predict the labels 
        #    -  kmeans has "nb_si" centroids
        #    -  in reality, several si mapps to the same cluster
        KMeans_labels = k.predict(X)
        COBRAS_labels = np.array([map_si_to_cluster[i] for i in  KMeans_labels])

        # Returns the "COBRAS" 
        return COBRAS_labels
    
    def score(self, X, y):
        # TODO :)
        pass