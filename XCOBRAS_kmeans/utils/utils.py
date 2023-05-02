from scipy.io import arff
import pandas as pd
import numpy as np

def read_arff_dataset(dataset_path):
    """function that reads arff files.

    Args:
        dataset_path (str): path of the file/dataset

    Returns:
        pandas.DataFrame: dataset
    """
    temp_data = arff.loadarff(open(dataset_path, 'r'))
    dataset = pd.DataFrame(temp_data[0])
    try:
        dataset["class"] = dataset["class"].str.decode('utf-8') 
    except KeyError:
        dataset["Class"] = dataset["Class"].str.decode('utf-8') 
        dataset["class"] = dataset["Class"]
        dataset = dataset.drop(["Class"], axis=1)
    return dataset


