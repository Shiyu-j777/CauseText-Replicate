### The code that execute preprocessing of the raw data

import pandas as pd
import pre_process_util as pre_helper
import numpy as np
import os






if __name__ == "__main__":
    ## The file to be processed
    input_file = "~/downloads/Project/SP23_causal_text/data/music.tsv"
    save_file = "~/downloads/Project/SP23_causal_text/data/music_rep_preprocessed.parquet"


    ###########################
    #### Label the data #######
    ###########################

    ### NOTICE: Potential Inconsistency with the original paper
    ### 1) the current data has some abnormal entries with tab in it
    ### 2) After removing abnormal entries, there are more than 17000 samples (indicated in pp.6 of Pryzant et. al) in the final data (I have 55980 entries in this data)

    raw_data = pd.read_csv(input_file, sep = "\t", on_bad_lines = 'skip')

    ## Label the true label
    raw_data['true_label'] = raw_data.apply(pre_helper.label_true_treament, axis = 1)

    ## label the noise label
    raw_data["proxy_noise"] = raw_data.apply(pre_helper.label_proxy_treatment, axis = 1, args =("random", 0.93, None))

    ## label the proxy-lex label
    ### Get the positive vocab
    positive_vocab = pre_helper.read_positive_lexicon(os.path.join(os.path.dirname(__file__), "positive-words.txt"))
    raw_data["proxy_lex"] = raw_data.apply(pre_helper.label_proxy_treatment, axis = 1, args =("lexicon", None, positive_vocab))

    ## Label the confounders here
    raw_data["confound"] = raw_data.apply(pre_helper.label_confounder, axis = 1)

    output_data = raw_data[["text", "confound", "true_label", "proxy_noise", "proxy_lex"]]


    ### The original data has adjusted for the propensity in that it makes P(T=1|C=0) = 0.9 and P(T=1|C=1) = 0.7, 
    ##  WHICH IS NOT STATED in the paper but performed in the actual code
    output_data = pre_helper.adjust_for_propensity(output_data, 0.9, 0.7)

    ## Perform adjustment on the proxy_lex data for propensity
    ## WHICH IS NOT STATED in the paper but performed in the actual code
    output_data = pre_helper.precision_recall_adjustment(output_data, column = "proxy_lex", 
                                                         precision = 0.94, recall = 0.98)
    
    output_data.reset_index(inplace = True)
    

    output_data.to_parquet(save_file)


    





    

