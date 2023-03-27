### helper functions for cleaning the data
import numpy as np
import re


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

def adjust_for_propensity(df, p0, p1):
    ## adjust for the propensity T=1|C by dropping some data that T=1
    ## The paper uses 0.9, 0.7
    ## INPUT:
    ## df: the dataset to change(pd.DataFrame)
    ## p0: 0.9 the desired propensity T=1|C=0
    ## p1: 0.7 the desired propensity T=1|C=1
    ## OUTPUT:
    ## df: the revised dataset that has adjusted to the reasonable propensity
    target_prop = [p0, p1]
    for i in range(2):
        current_subset = df.loc[df["confound"] == i]
        current_subset_T = current_subset.loc[current_subset["true_label"] == 1]
        treat_size = current_subset_T.shape[0]
        current_confound_size = current_subset.shape[0]
        target_size = (current_confound_size- treat_size) *target_prop[i]/(1 - target_prop[i])
        drop_prob = (treat_size-target_size) / treat_size

        df = df.drop(current_subset_T.sample(frac=drop_prob).index)

    return(df)

def precision_recall_adjustment(df, column = "proxy_lex",
                                precision = 0.98, recall = 0.94):
    
    true_1 = df.loc[df["true_label"] == 1]
    true_0 = df.loc[df["true_label"] == 0]

    true_positive = true_1.loc[true_1[column] == 1]
    false_positive = true_0.loc[true_0[column] == 1]
    false_negative = true_1.loc[true_1[column] == 0]

    current_precision = len(true_positive)/(len(true_positive) + len(false_positive))
    current_recall = len(true_positive)/(len(true_positive) + len(false_negative))


    if ((current_precision > precision) and (current_recall > recall)): ## if adjusting TP first
        desirable_size_for_precision = len(false_positive) *  precision / (1-precision)
        desirable_size_for_recall = len(false_negative) *  recall / (1-recall)
        if desirable_size_for_precision < desirable_size_for_recall: ## prioritize smaller target size first
            ## adjust for precision first
            drop_prob = (len(true_positive) - desirable_size_for_precision)/len(true_positive)
            df = df.drop(true_positive.sample(frac = drop_prob).index)

            ## adjust for recall then: doing it on false_negative
            desirable_size_for_recall = len(true_positive) * (1-drop_prob) * (1-recall) / recall
            drop_prob = (len(false_negative) - desirable_size_for_recall)/len(false_negative)
            df = df.drop(false_negative.sample(frac = drop_prob).index)
        else: ## prioritize recall first
            drop_prob = (len(true_positive) - desirable_size_for_recall)/len(true_positive)
            df = df.drop(true_positive.sample(frac = drop_prob).index)

            ## adjust for precision then: doing it on p
            desirable_size_for_precision = len(true_positive) * (1-drop_prob) * (1-precision) / precision
            drop_prob = (len(false_positive) - desirable_size_for_precision)/len(false_positive)
            df = df.drop(false_positive.sample(frac = drop_prob).index)
    elif (current_precision > precision) and current_recall < recall: ## adjust precision first
        desirable_size_for_precision = len(false_positive) *  precision / (1-precision)
        drop_prob = (len(true_positive) - desirable_size_for_precision)/len(true_positive)
        df = df.drop(true_positive.sample(frac = drop_prob).index)

        ## adjust recall

        desirable_size_for_recall = len(true_positive) * (1-drop_prob) * (1-recall) / recall
        drop_prob = (len(false_negative) - desirable_size_for_recall)/len(false_negative)
        df = df.drop(false_negative.sample(frac = drop_prob).index)
    elif (current_precision < precision) and current_recall > recall:
        desirable_size_for_recall = len(false_negative) *  recall / (1-recall)
        drop_prob = (len(true_positive) - desirable_size_for_recall)/len(true_positive)
        df = df.drop(true_positive.sample(frac = drop_prob).index)

        ## adjust for precision then: doing it on p
        desirable_size_for_precision = len(true_positive) * (1-drop_prob) * (1-precision) / precision
        drop_prob = (len(false_positive) - desirable_size_for_precision)/len(false_positive)
        df = df.drop(false_positive.sample(frac = drop_prob).index)
    else: ## adjust FP TP independently
        
        desirable_size_for_precision = len(true_positive) * (1-precision) / precision
        drop_prob = (len(false_positive) - desirable_size_for_precision)/len(false_positive)

        df = df.drop(false_positive.sample(frac = drop_prob).index)

        desirable_size_for_recall = len(true_positive)  * (1-recall) / recall
        drop_prob = (len(false_negative) - desirable_size_for_recall)/len(false_negative)
        
        df = df.drop(false_negative.sample(frac = drop_prob).index)

    return(df)



    

