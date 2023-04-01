### The code that execute the experiment of one of the processed data
### Different from the source code, this version intend to provide more annotation and more detailed steps of what has been done


import pandas as pd
import experiment_utils as exp_helper
import numpy as np
import os
import t_boost as t_boost_helper






if __name__ == "__main__":
    ## The file to be processed
    input_file = "~/downloads/Project/SP23_causal_text/data/music_preprocessed.tsv"

    output_file = "~/downloads/Project/SP23_causal_text/data/music_preprocessed_boosted.parquet"

    ##input_file = "~/downloads/Project/SP23_causal_text/data/music_preprocessed.tsv"

    ### input desirable effect here:
    treatment_level = "low"
    noise_level = "low"
    confound_level = "low"
    offset = 0.9
    penalty_l2 = 0.

    ### Below is the effect that is tested in the paper

    parameter_dictionary = {"treatment_effect":{"low": 0.4,
                                         "high": 0.8},
                            "confound_effect":{"low" : -0.4,
                                               "high": 4.0},
                            "noise":{"low": 0,
                                     "high": 1}}




    ################################
    #### Simulate the Effect #######
    ################################


    labeled_data = pd.read_csv(input_file, sep = "\t", error_bad_lines=False)

    ## calculate a propensity score for P(T=1|C=0) and P(T=1|C=1) and stored in tuples

    
    ################################
    #### Calculate the Effect ######
    ################################

    

    unadjusted = exp_helper.naive_ATE(labeled_data["T_proxy"], labeled_data["Y"])

    proxy_lex = exp_helper.adjusted_ATE(labeled_data["T_proxy"], \
                                     labeled_data["C"], \
                                        labeled_data["Y"])

    ######################
    #### T-boosting ######
    ######################

    ## create tokenized data
    X_matrix, vocab, vectorizer = t_boost_helper.tokenize_dataset(labeled_data)

    
    labeled_data["proxy_star_logit"] = t_boost_helper.t_boost_label_logit(X_matrix, labeled_data["T_proxy"], \
                                       threshold = 0.22, penalty_weight = 0.00077, flip_zeros = True)
    
    labeled_data["proxy_star_PU"] = t_boost_helper.t_boost_label_PU(X_matrix,  labeled_data["T_proxy"], \
                                       threshold = 0.22, inner_penalty= 0.00359, outer_penalty= 0.00077, flip_zeros = True)
    

    ## Effect after boosting
    T_boost_logit = exp_helper.adjusted_ATE(labeled_data["proxy_star_logit"], \
                                     labeled_data["C"], \
                                        labeled_data["Y"])
    
    ## Effect after boosting
    T_boost_PU = exp_helper.adjusted_ATE(labeled_data["proxy_star_PU"], \
                                     labeled_data["C"], \
                                        labeled_data["Y"])
    


    print("The parameter setting is printed below:")

    print("Treatment Effect:", treatment_level)
    print("Confound Effect:", confound_level)
    print("Noise:", noise_level)
   

    print("Phi-unadjusted is", unadjusted*100)
    print("Phi-proxy-lex is", proxy_lex * 100)


    print("T_boost effect with logistic regression is", T_boost_logit * 100)
    print("T_boost effect with PU is", T_boost_PU * 100)

    labeled_data.reset_index(inplace=True)

    labeled_data.to_parquet(output_file)

