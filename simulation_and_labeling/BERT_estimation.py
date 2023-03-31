import torch
import pandas as pd
import Cause_BERT_helper as causebert_helper
import datasets
from transformers import DistilBertTokenizer


if __name__ == "__main__":
    ### Change to the directory and file name where you store your parametric experiment results
    input_file = "~/downloads/Project/SP23_causal_text/data/music_rep_preprocessed_boosted.parquet"

    ### Note

    ### clear up the dataset
    test_data = pd.read_parquet(input_file)

    transformed_data = causebert_helper.experiment_data_builder(test_data["text"], \
                                            test_data["proxy_star_PU"], ## can be replaced with any versions of T\
                                            test_data["confound"],\
                                            test_data["Y"])
    
    model = causebert_helper.training_cause_text()

    model.train(transformed_data["train"], transformed_data["val"], epoches = 3)
    Q0_estimate, Q1_estimate = model.test(transformed_data["test"])
    ATE_text = model.ATE_text_adjust(Q0_estimate, Q1_estimate)

    print("The Text Cause adjustment effect is", ATE_text * 100)


    
    
    
    
    

    

    