#! /bin/sh
pyth=/mnt/c/Users/user/anaconda3/envs/XClustering_env39/python.exe
path="../../../datasets/deric_benchmark/real-world/"
strat="commun_fracture"
budget="180"

echo "strat=${strat}"

for n in 3 5 7 10
do
    for threshold in 0.50
    do 
        for dataset in wine wisc glass 
        do
            echo " -- ${dataset} top-n=${n} threshold=${threshold}"
            $pyth script.py ${path}${dataset}.arff --dataname ${dataset} --budget ${budget} --top-n ${n} --threshold ${threshold} --strat ${strat} --test
        done
    done
done