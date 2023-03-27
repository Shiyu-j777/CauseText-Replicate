### The code that execute the experiment of the data
### This is the version 

import pandas as pd
import experiment_utils as exp_helper
import numpy as np
import os






if __name__ == "__main__":
    ## The file to be processed
    input_file = "~/downloads/Project/SP23_causal_text/data/music_rep_preprocessed.parquet"

    ##input_file = "~/downloads/Project/SP23_causal_text/data/music_preprocessed.tsv"

    ### input desirable effect here:
    treatment_level = "low"
    noise_level = "low"
    confound_level = "low"
    offset = 0.9

    parameter_dictionary = {"treatment_effect":{"low": 0.4,
                                         "high": 0.8},
                            "confound_effect":{"low" : -0.4,
                                               "high": 4.0},
                            "noise":{"low": 0,
                                     "high": 1}}




    ################################
    #### Simulate the Effect #######
    ################################

    labeled_data = pd.read_parquet(input_file)

    ##labeled_data = pd.read_csv(input_file, sep = "\t")

    ## calculate a propensity score for P(T=1|C=0) and P(T=1|C=1) and stored in tuples

    confound_propensity = exp_helper.calculate_propensity(labeled_data)

    print(confound_propensity)

    labeled_data["Y"] = labeled_data.apply(exp_helper.simulate_effect, axis = 1, 
                                           args = (parameter_dictionary["treatment_effect"][treatment_level],
                                                   confound_propensity, 
                                                   parameter_dictionary["confound_effect"][confound_level],
                                                   parameter_dictionary["noise"][noise_level],
                                                   offset))
    
    ################################
    #### Calculate the Effect #######
    ################################
    
    oracle = exp_helper.adjusted_ATE(labeled_data["true_label"], \
                                     labeled_data["confound"], \
                                        labeled_data["Y"])

    print("The parameter setting is printed below:")

    print("Treatment Effect:", treatment_level)
    print("Confound Effect:", confound_level)
    print("Noise:", noise_level)
    
    print("Oracle is", oracle*100)

    unadjusted = exp_helper.naive_ATE(labeled_data["proxy_lex"], labeled_data["Y"])

    print("Phi-unadjusted is", unadjusted*100)

    proxy_lex = exp_helper.adjusted_ATE(labeled_data["proxy_lex"], \
                                     labeled_data["confound"], \
                                        labeled_data["Y"])
    
    print("Phi-proxy-lex is", proxy_lex * 100)

    



    

    

    





    

