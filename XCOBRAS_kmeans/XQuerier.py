from cobras_ts.querier import Querier
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class XQuerier(Querier):

    def __init__(self, labels, xai_method="shap", strat="commun_fraction", top_n=3, threshold = 0.5, test=False):
        super(XQuerier, self).__init__()
        self.labels = labels # ground truth
        self.top_n = top_n
        self.xai_method = xai_method
        self.strat = strat
        self.threshold = threshold
        self.test = test

    def query_points(self, idx1, idx2, exp1=None, exp2=None):
        
        if exp1 == None or exp2 == None or self.strat=="ground_truth":
            # use ground truth
            answer = self.labels[idx1] == self.labels[idx2]

            return answer

        if self.xai_method == "shap":
            return self.query_points_shap(idx1, idx2, exp1, exp2)
        else:
            # lime
            return self.query_points_lime(idx1, idx2, exp1, exp2)


    def query_points_shap(self, idx1, idx2, exp1=None, exp2=None):

        if self.strat=="exp_sim":
            # TODO
            ...

        # ---
        exp1_values = exp1.values
        exp2_values = exp2.values
        
        # sort the values
        ind_exp1 =  np.abs(exp1_values).argsort()[-self.top_n:][::-1]
        ind_exp2 =  np.abs(exp2_values).argsort()[-self.top_n:][::-1]

        # sort feature_names
        feature_names_exp1 = np.array(list(exp1.feature_names))[ind_exp1]
        feature_names_exp2 = np.array(list(exp2.feature_names))[ind_exp2]
        
        # commun/shared feature_names
        fi_intersection = set(feature_names_exp1).intersection(set(feature_names_exp2))
        commun_fraction = len(fi_intersection)

        if self.strat == "commun_fraction":
            answer = commun_fraction*1./self.top_n > self.threshold
            return answer
        
        if self.strat == "cosine_similarity":
            answer = cosine_similarity(np.array([exp1.values]), np.array([exp2.values])) >= self.threshold 
            return answer

    def query_points_lime(self, idx1, idx2, exp1=None, exp2=None):
        ...
        # Pourquoi ensemble ? - label 1
        exp1_label1 = np.asanyarray([[a, b] for (a,b) in exp1.as_list(label=1)])
        exp2_label1 = np.asanyarray([[a, b] for (a,b) in exp2.as_list(label=1)])

        if self.strat == "commun_fraction":
            feature_names_exp1 = set(exp1_label1[:self.top_n, 0])
            feature_names_exp2 = set(exp2_label1[:self.top_n, 0])
            fi_intersection = set(feature_names_exp1).intersection(set(feature_names_exp2))
            commun_fraction = len(fi_intersection)
            answer = commun_fraction*1./self.top_n > self.threshold
            return answer

        # ordonner et
        exp1_sorted, exp2_sorted, _ = self.sort_lime_explanation(exp1_label1, exp2_label1)
        if self.strat == "cosine_similarity":
            answer = cosine_similarity(np.array([exp1_sorted]), np.array([exp2_sorted])) >= self.threshold 
            return answer

    def sort_lime_explanation(self, exp1, exp2):
        """Order the two lime explanations according to the first explanation

        Args:
            exp1 (np.array): first  lime explanation np.asanyarray()
            exp2 (np.array): second lime explanation np.asanyarray()

        Returns:
            np.array: sorted_exp1_values 1D
            np.array: sorted_exp2_values 1D
            list:     corresponding feature order
        """
        res_exp1, res_exp2 = [], []
        exp1, exp2 = dict(exp1), dict(exp2)
        for idx in exp1.keys():

            res_exp1.append(exp1[idx])
            try: 
                res_exp2.append(exp2[idx])
            except:
                print(idx)
                print(list(exp1.keys()))
                print(list(exp2.keys()))
        return np.array(res_exp1), np.array(res_exp2), list(exp1.keys())
        

        



