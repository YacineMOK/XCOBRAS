import numpy as np
import pandas as pd
from sklearn import metrics, datasets

import warnings
warnings.filterwarnings("ignore")

from utils.utils import read_arff_dataset
from RandomQuerier import RandomQuerier
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.querier.labelquerier import LabelQuerier

def save_results(dataname, budget, intermediate_clusterings, y, sub_path="ground_truth", additional=None):
    print("\t...Saving the results...")
    path = "./results/" + sub_path + "/"
    data = {
        'budget': [],
        'strat' : [],
        'number clusters' : [],
        'ARI': []
    }
    for i in range(10, budget+1, 10):
        tmp_clustering = intermediate_clusterings[i-1]
        tmp_ari = metrics.adjusted_rand_score(tmp_clustering,y)
        
        data['budget'].append(i)                                 # VAR
        data['strat'].append(sub_path)                       # cst
        data['number clusters'].append(len(set(tmp_clustering))) # VAR
        data['ARI'].append(tmp_ari)                              # VAR

    df_results = pd.DataFrame(data=data)

    if additional:
        df_results.to_csv(path+dataname+"_budget"+str(budget)+"_Trial_"+str(additional),
                                index=False)
    else:
        df_results.to_csv(path+dataname+"_budget"+str(budget)+"_GT",
                                index=False)
    print("\tsaved!")

def main():
    path="../../../datasets/deric_benchmark/real-world/"
    datasets = ['wine', 'wisc', 'glass']
    budget = 180
    rnd_trials = 20

    for dataname in datasets:
        print(f"dataset: {dataname}")

        # load files
        data = read_arff_dataset(path+dataname+".arff")
        X, y = data.drop(["class"], axis=1).values, data["class"].values
        # feature_names = X.columns

        print(f"-- COBRAS: Ground Truth")
        # run cobras - GROUND TRUTH
        clusterer = COBRAS_kmeans(X, LabelQuerier(y), budget)
        clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster()

        # save results
        save_results(dataname, budget, intermediate_clusterings, y, sub_path="ground_truth")

        print("-- COBRAS: Random")
        for i in range(rnd_trials):
            print(f"---- trial={i}")
            # run cobras - RANDOM
            clusterer = COBRAS_kmeans(X, RandomQuerier(), budget)
            clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster()

            # save results
            save_results(dataname, budget, intermediate_clusterings, y, sub_path="random", additional=i)

if __name__ == '__main__':
    main()