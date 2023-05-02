from cobras_ts.querier import Querier
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class XQuerier(Querier):

    def __init__(self, labels, strat="commun_fraction", top_n=3, threshold = 0.5, test=False):
        super(XQuerier, self).__init__()
        self.labels = labels # ground truth
        self.top_n = top_n
        self.strat = strat
        self.threshold = threshold
        self.test = test

    def query_points(self, idx1, idx2, exp1=None, exp2=None):
        
        if exp1 == None or exp2 == None or self.strat=="ground_truth":
            # use ground truth
            answer = self.labels[idx1] == self.labels[idx2]

            return answer
        
        # ---

        if self.strat=="exp_sim":
            ...

        # ---
        # print(f"exp1: {exp1.values} | exp2: {exp2.values} ")
        exp1_values = exp1.values
        exp2_values = exp2.values
        # sort the values
        # print(f"set d'exp= {len(set(exp1.feature_names))}")
        ind_exp1 =  np.abs(exp1_values).argsort()[-self.top_n:][::-1]
        ind_exp2 =  np.abs(exp2_values).argsort()[-self.top_n:][::-1]
        # print(f"ind_exp1: {ind_exp1} | ind_exp2: {ind_exp1} ")
        # sort feature_names
        feature_names_exp1 = np.array(list(exp1.feature_names))[ind_exp1]
        feature_names_exp2 = np.array(list(exp2.feature_names))[ind_exp2]
        # commun/shared feature_names
        fi_intersection = set(feature_names_exp1).intersection(set(feature_names_exp2))

        # CASE 1: (disjoint feature_names)
        if fi_intersection == set():
            # top-n feature importance are different
            # --cannot compare--
            # better to SPLIT
            return False
        
        # CASE 2: (shared feature_names)
        commun_fraction = len(fi_intersection)

        if self.strat == "commun_fraction":
            answer = commun_fraction*1./self.top_n > self.threshold
            return answer
        
        if self.strat == "cosine_similarity":
            # TODO respect top_n
            return cosine_similarity(np.array([exp1.values]), np.array([exp2.values])) >= self.threshold 

        # if self.strat == "top_n":
        #     values_to_compare = np.zeros(commun_fraction)
        #     for feature in fi_intersection:
        #         ...
            





                



