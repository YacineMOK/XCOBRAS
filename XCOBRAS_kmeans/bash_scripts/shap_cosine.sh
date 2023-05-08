#! /bin/sh
pyth=/mnt/c/Users/user/anaconda3/envs/XClustering_env39/python.exe
path="../../../../datasets/deric_benchmark/real-world/"
strat="cosine_similarity"
budget="180"
xai="shap"

echo "xai method = ${xai} | strat = ${strat}"

for threshold in 0.50 0.60 0.70 0.80 0.90
do
    for dataset in wine wisc glass
    do
        echo " -- threshold=${threshold}: ${dataset} dataset..."
        $pyth ../script.py ${path}${dataset}.arff --dataname ${dataset} --budget ${budget} --threshold ${threshold} --xai-model ${xai} --strat ${strat} --test
    done
done