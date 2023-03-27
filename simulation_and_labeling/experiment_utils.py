import numpy as np

def sigmoid(x):
    ### This is a scalar version of the sigmoid function
    return(1/(1+np.exp(-x)))


def calculate_propensity(df):
    ## Calculate the propensity of the confound effects
    ## INPUT:
    ## df: the dataframe that consists of all the treatment/confound information
    ## RETURN:
    ## a tuple (P(T=1|C=0), P(T=1|C=0))

    propensity_0 = np.sum([label ==1 and confound == 0 for 
                           label, confound in
                             zip(df["true_label"], df["confound"])])/\
                                np.sum(df["confound"] == 0)
    propensity_1 = np.sum([label ==1 and confound == 1 for 
                           label, confound in
                             zip(df["true_label"], df["confound"])])/\
                                np.sum(df["confound"] == 1)
    return(propensity_0, propensity_1)

def simulate_effect(single_row_df, treatment_effect, propensity, confound_effect , noise, offset ):
    ## take a dataframe, and randomly simulate treatment effect based on the following formula
    #### Formula for both effects on
    #### Y ~ Bernoulli(sigmoid(confound_effect*(propensity - offset) + treatment_effect + normal_noise))
    ## INPUT:
    ## single_row_df: @pd.dataframe: a single row of df that the operation is on 
    ## treatment_effect: float, parameter for treatment effect
    ## propensity: float, the tuple that consists (P(T|C=0), P(T|C=1))
    ## confound_effect: float, parameter for confound effects
    ## noise: the noise variance of the normal distribution term
    ## offset: the offset term applied to the confound effect
    ## RETURN:
    ## randomly generated bernoulli outcome based on the formula

    if noise > 0:
        probability = sigmoid(confound_effect * (propensity[single_row_df["confound"]] - offset) +
                           treatment_effect * single_row_df["true_label"]  +
                           np.random.normal(loc = 0, scale = noise))
    else:
        probability = sigmoid(confound_effect * (propensity[single_row_df["confound"]] - offset) +
                           treatment_effect * single_row_df["true_label"])


    return(int(np.random.choice([0,1], p = [1-probability, probability])))


def adjusted_ATE(treatment, confound, Y):
    ### Output the effect of ATE adjusted by the confounder
    ### Take the average of treatment effect across each group, unweighted
    AE_by_group = []
    
    for confound_type in set(confound):
        t_index_in_group = [t == 1 and c == confound_type for t, c in zip(treatment, confound)]
        no_t_index_in_group = [t == 0 and c == confound_type for t, c in zip(treatment, confound)]
        TE_of_group = np.mean(Y.iloc[t_index_in_group]) - np.mean(Y.iloc[no_t_index_in_group])
        AE_by_group.append(TE_of_group)
    
    return(np.mean(AE_by_group))

def naive_ATE(treatment, Y):
    ### Output the effect of ATE un-adjusted by the confounder
    ### Take the average of treatment effect
    t_index = [ti == 1 for ti in treatment]
    no_t_index = [ti == 0 for ti in treatment]
    ATE = np.mean(Y.iloc[t_index]) - np.mean(Y.iloc[no_t_index])
    
    return(ATE)


    