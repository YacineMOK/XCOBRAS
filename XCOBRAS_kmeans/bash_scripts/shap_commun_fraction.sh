#! /bin/sh
pyth=/mnt/c/Users/user/anaconda3/envs/XClustering_env39/python.exe
path="../../../../datasets/deric_benchmark/real-world/"
strat="commun_fraction"
budget="180"
threshold="0.50"
xai="shap"

echo "xai method = ${xai} | strat = ${strat}"
 &
for n in 3 5 7 9 10
do  
    for  dataset in wine wisc glass
    do
        echo " -- top-n=${n} | threshold=${threshold} | ${dataset} dataset..."
        $pyth ../script.py ${path}${dataset}.arff --dataname ${dataset} --budget ${budget} --top-n ${n} --threshold ${threshold} --xai-model ${xai} --strat ${strat} --test
    done
done