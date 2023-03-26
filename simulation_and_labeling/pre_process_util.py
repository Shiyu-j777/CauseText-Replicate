### helper functions for cleaning the data
import numpy as np
import re
import os

def tokenize_sentence(word):
    ## Simple function to Tokenize the sentence
    return(re.findall('\w+', word.lower()))

def read_positive_lexicon(file_path):
     lexicon = {
        line.strip().lower()  for line in
        open(file_path) if not line.startswith(";") and line.strip()
     }
     return(lexicon)

def label_true_treament(single_row_df):
    ## take a dataframe, and randomly/text-based assign true labels
    ## INPUT:
    ## single_row_df: @pd.dataframe: a single row of df that the operation is on 
    ## OUTPUT:
    ## In-place revision of true label
    if single_row_df["rating"] <= 2.0:
        return(0)
    else:
        return(1) 
    
def label_confounder(single_row_df):
    ## take a dataframe, and randomly/text-based assign true labels
    ## INPUT:
    ## single_row_df: @pd.dataframe: a single row of df that the operation is on 
    ## OUTPUT:
    ## In-place revision of true label
    if single_row_df["product"] == "audio cd":
        return(1)
    else:
        return(0) 


def label_proxy_treatment(single_row_df, method = "random", true_prob = 0.93, vocab = None):
    ## take a dataframe, and randomly/text-based assign proxy labels
    ## INPUT:
    ## single_row_df: @pd.dataframe: a single row of df that the operation is on 
    ## method: type(char) "random" - accuracy dictated by the probs parameter or 
    ########### "lexicon" - lexicon overlap based method
    ## probs: the probability of the correct labels, used by "random" method only, default to 93% according to paper
    if method == "random": ## 0.93 percentage of correct treatment
        correct = np.random.choice([False, True], p = [1 - true_prob, true_prob])
        if not correct:
            return(1 - single_row_df["true_label"])
        else:
            return(single_row_df["true_label"])
    elif method == "lexicon":
            document_words = set(tokenize_sentence(single_row_df["text"]))
            overlap = [word in vocab for word in document_words]
            if any(overlap):
                return(1)
            else:
                return(0)

def simulate_effect(single_row_df, treatment_effect , confound_effect , noise, offset ):
    ## take a dataframe, and randomly simulate treatment effect based on the following formula
    #### Formula for both effects on
    #### Y ~ Bernoulli(sigmoid(confound_effect*(propensity - offset) + treatment_effect + normal_noise))
    ## INPUT:
    ## single_row_df: @pd.dataframe: a single row of df that the operation is on 
    ## treatment_effect: float, parameter for treatment effect
    ## confound_effect: float, parameter for confound effects
    ## noise: the noise variance of the normal distribution term
    ## offset: the offset term applied to the confound effect
    ########### "lexicon" - lexicon overlap based method
    ## probs: the probability of the correct labels, used by "random" method only, default to 93% according to paper
    pass