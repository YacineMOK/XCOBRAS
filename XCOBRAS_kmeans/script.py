import argparse
from utils.utils import *

# ------- IMPORTS
# -- model(s)
from XQuerier import XQuerier
from xcobras_kmeans import XCOBRAS_kmeans

# -- metrics
from sklearn.metrics import adjusted_rand_score

# -- plot(s)
from utils.plots import plot_2D, plot_boundary
import matplotlib.pyplot as plt

# -- dataset(s) 
from sklearn import datasets
from scipy.io import arff
import pandas as pd
import numpy as np

# -- others
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    """Function that parses the arguments passed before the execution.

    Returns:
        args: arguments
    """
    parser  = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help="Path to the dataset")            # positional argument
    parser.add_argument('--dataname', type=str, help="Name of the dataset (useful when testing)")
    parser.add_argument('-b' , '--budget', type=int, help="Number of queries")      # positional argument
    parser.add_argument('--test', default=False, action='store_true', help="If set on True, results will be saved")
    parser.add_argument('-xm', '--explain-model', choices=['rbf_svm', 'm2'], default='rbf_svm', help="The supervised model that fits on the current partitioning and that we will explain") #
    parser.add_argument('-s', '--strat', choices=['commun_fraction', 'ground_truth', 'exp_sim', 'cosine_similarity'], default="commun_fraction", help="The strategy with which the querier interprets the explanations")
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help="threshold on which our agent will base its answers to queries")
    parser.add_argument('-n' , '--top-n', type=int, default=3, help="Top n XAI features to concider when taking a decision")      #
    parser.add_argument('-ts', '--test-size', default=0.4, type=float, help="percentage of") #
    parser.add_argument('-v' , '--verbose', default=False, action='store_true')     # on/off flag
    args = parser.parse_args()
    return args

def main():
    """Main function
    """
    # parse the args
    inputs = parse_args()

    # read file TODO gérer les CSV ou les arff
    data = read_arff_dataset(inputs.dataset)
    X, y = data.drop(["class"], axis=1), data["class"]
    feature_names = X.columns

    # build querier
    xai_querier = XQuerier(
        y.values,
        strat     = inputs.strat,
        top_n     = inputs.top_n,
        threshold = inputs.threshold
    )

    # instanciate and train the model
    xcobras_kmeans = XCOBRAS_kmeans(inputs.budget, model=inputs.explain_model, test_size=inputs.test_size, verbose=inputs.verbose)
    clustering, intermediate_clusterings, runtimes, ml, cl = xcobras_kmeans.fit(X.values, feature_names=feature_names, y=xai_querier)
    

    # Final ARI score
    if inputs.verbose:
        print(f"Length of intermediate_clusterin: {len(intermediate_clusterings)} | Budget: {inputs.budget}")
        y_hat = xcobras_kmeans.predict(X.values)
        print(f"Model's final ARI score: {adjusted_rand_score(y_hat, y):.3f}")

    if inputs.test:
        
        print("------")
        path = "./results/"+inputs.strat+"/"
        print(f"Writing the results in: {path}")
        f_all = open(path+inputs.dataname+"_all.csv", "a")
        f = None

        if inputs.strat == "commun_fraction":
            f = open(path+inputs.dataname+"_top_"+str(inputs.top_n)+".csv", "w")
            f.write("top_n;budget;ARI\n")

        if inputs.strat == "cosine_similarity":
            f = open(path+inputs.dataname+"_cosine_"+str(int(inputs.threshold*100))+".csv", "w")
            f.write("threshold;budget;ARI\n")

        for i in range(10, inputs.budget, 10):
            tmp_clustering = intermediate_clusterings[i]
            tmp_ari = adjusted_rand_score(tmp_clustering,y)

            f_all.write(str(inputs.top_n) + ";" + str(int(inputs.threshold*100)) + ";" + str(i) + ";" +  str(tmp_ari)+"\n")
           
            if inputs.strat == "commun_fraction":
                f.write(str(inputs.top_n) + ";" + str(i) + ";" +  str(tmp_ari) +"\n")

            if inputs.strat == "cosine_similarity":
                f.write(str(int(inputs.threshold*100)) + ";" + str(i) + ";" +  str(tmp_ari) +"\n")


        print("------")

    

if __name__ == '__main__':
    main()

# exemple d'execution
# python script.py "../../../datasets/deric benchmark/artificial/target.arff"  --budget 10