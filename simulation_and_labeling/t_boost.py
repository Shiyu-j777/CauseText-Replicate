import re
from collections import Counter
from sklearn import feature_extraction
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from scipy import stats
import numpy as np



def tokenize_sentence(sentence):
    ### a simple tokenizer
    return(re.findall('\w+', sentence))

def tokenize_dataset(df,
                     stopwords = None,
                     vocabulary_size = 2000,
                     binary_variable = True):
    ### produce an X dataset for the classifier for the T-boosting
    ### This implementation has referred to the original source code, with a different naming to explain what happened
    ### INPUT:
    ###### df: the dataframe that contains the text to be treated
    ###### stopwords: the list/set of stopwords to be excluded
    ###### vocabulary_size = the size of vocabulary that will be considered
    ###### binary_variable = whether the model will use binary indicator or counts about whether a word exists
    ### OUTPUT:
    ###### transformed_corpus: a matrix of output
    ###### vocabulary: the vocab used for the study
    ###### vectorizer of the scikit object


    def not_stop_word(word):
        return(not (word in stopwords))
    
    if stopwords is None:
        vocabulary_count = Counter([word for sentence in df["text"] for word in tokenize_sentence(sentence.lower())])
    else:
        vocabulary_count = Counter([word for sentence in df["text"] for word in tokenize_sentence(sentence.lower()) 
                                    if not_stop_word(word)])
        
    vocabulary = [word for word, _ in vocabulary_count.most_common(vocabulary_size)]

    # vectorize inputs
    vectorizer_of_data = feature_extraction.text.CountVectorizer(
        lowercase=True,
        tokenizer=tokenize_sentence,
        vocabulary=vocabulary,
        binary= binary_variable,
        ngram_range=(1, 1)) ### This part follows how the original source code does it
    raw_corpus = list(df['text'])
    vectorizer_of_data.fit(raw_corpus)
    transformed_corpus = vectorizer_of_data.transform(raw_corpus).todense()
    return transformed_corpus, vocabulary, vectorizer_of_data


def t_boost_label_logit(X, labels, threshold, penalty_weight, flip_zeros = True):
    ### Return boosted labels using logistic regression given raw regression X matrices and Y matrix
    ### INPUT:
    ###### X: the numerical matrix for regression
    ###### labels: the outcome
    ###### threshold: the probability threshold to flip
    ###### penalty: penalty term for l2 penalty
    ### OUTPUT:
    ###### Return New labels

    logit_reg = SGDClassifier(loss = "log_loss", penalty='l2', alpha = penalty_weight)
    logit_reg.fit(X, labels)
    star_labels = create_expand_labels(logit_reg, X, labels, threshold, flip_zeros)
    
    return(star_labels)




class PUClassifier(object):
    ### NOTE: This comes directly from the original implementation while I reduced unnecessary codes not used for replication
    ### I also add Annotation to help with the understanding of the paper
    ### PU estimator consists of two steps
    ### 1. An inner estimator that classifies T=1 based on X
    ### 2. An outer estimator that comes from reweighted 

    ### FOR THE ORIGINAL Implementation of this method, goes to: https://github.com/rpryzant/causal-text/blob/main/src/main.py
    """
    Learning classifiers from only positive and unlabeled data (Noto et al., 2008)

        self.inner_estimator: g(x), a traditional classifier for p(s=1 | x) and c computation
        self.outer_estimator: f(x), classifier learned from c-based data reweighting

    example:
        s = y.copy()
        s[:n_unlabeled] = 0  # overwrite some 1's
        pu = PUClassifier(n_folds=5)
        pu.fit(X, s)
        ypred = pu.predict(X[s==0])

    """
    def __init__(self, inner_alpha, outer_alpha, n_folds=5):
        self.inner_alpha = inner_alpha
        self.inner_estimator = SGDClassifier(loss="log", penalty="l2", alpha=inner_alpha)

        self.outer_alpha = outer_alpha
        self.n_folds = n_folds
        self.fitted = False

    # for sklearn compatibility
    def get_params(self, deep=True):
        return {"n_folds": self.n_folds, 'inner_alpha': self.inner_alpha, 'outer_alpha': self.outer_alpha}    

    # for sklearn compatibility
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def compute_c(self, X, T_proxy):
        c = np.zeros(self.n_folds)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)
        ## use cross validation to find the average probability from classifier for observations with proxy-label 1
        ## \bar{P}_{inner}(T=1) for T=1
        for i, (itr, ite) in enumerate(skf.split(T_proxy, T_proxy)):
            self.inner_estimator.fit(X[itr], T_proxy[itr])
            ## derive a mean probability estimate for cases that have label = 1 in the test set
            ## This is the 
            c[i] = self.inner_estimator.predict_proba(X[ite][T_proxy[ite]==1])[:,1].mean()
        self.c = c.mean()


    def sample(self, X, T_proxy):
        if not hasattr(self, "c"):
            self.compute_c(X, T_proxy)
        
        X_positive = X[T_proxy==1]
        X_unlabeled = X[T_proxy==0]
        n_positive = X_positive.shape[0]
        n_unlabeled = X_unlabeled.shape[0]

        ### Create an augmented dataset where unlabelled cases are duplicated by a 1 and 0 out comes.

        X_train = np.r_[X_positive, X_unlabeled, X_unlabeled]
        y_train = np.concatenate([
            np.repeat(1, n_positive), 
            np.repeat(1, n_unlabeled), 
            np.repeat(0, n_unlabeled)])
        
        ### Run the logistic regression with the augmented data

        self.inner_estimator.fit(X, T_proxy)
        ### Get the predicted probability for the positive labels
        p_unlabeled = self.inner_estimator.predict_proba(X_unlabeled)[:,1]
        ### Weight of each sample is the odd of 1-c times the odd of the predicted probability of each unlabeled token
        w_positive = ((1 - self.c) / self.c) * (p_unlabeled / (1 - p_unlabeled))
        w_negative = 1 - w_positive
        sample_weight = np.concatenate(
            [np.repeat(1.0, n_positive), 
            w_positive, 
            w_negative])
        return X_train, y_train, sample_weight


    def fit(self, X, s):
        if not hasattr(self, "c"):
            self.compute_c(X, s)

        ### Created expanded dataset with weight

        X_train, y_train, sample_weight = self.sample(X, s)

        alpha = self.outer_alpha

        self.outer_estimator = SGDClassifier(
            loss="log", 
            penalty="l2", 
            alpha=alpha, 
            class_weight={1:1}).fit(
            X_train, y_train, sample_weight=sample_weight)

        self.fitted = True


    def predict_proba(self, X):
        if not self.fitted:
            raise Exception('not fitted yet!')

        return self.outer_estimator.predict_proba(X)

    def predict(self, X, threshold=0.8):
        ### NOT USED HERE
        if not self.fitted:
            raise Exception('not fitted yet!')
        raise NotImplementedError


def t_boost_label_PU(X, labels, threshold, inner_penalty, outer_penalty, flip_zeros = True):
    ### Return boosted labels using PU classifier given raw regression X matrices and Y matrix
    ### INPUT:
    ###### X: the numerical matrix for regression
    ###### labels: the outcome
    ###### threshold: the probability threshold to flip
    ###### penalty: penalty term for l2 penalty
    ### OUTPUT:
    ###### Return New labels

    PU_classifier = PUClassifier(inner_penalty,  outer_penalty)
    PU_classifier.fit(X, labels)
    star_labels = create_expand_labels(PU_classifier, X, labels, threshold, flip_zeros)
    
    return(star_labels)

def create_expand_labels(reg_model, X, labels, threshold, flip_zeros):

    if flip_zeros:
        X_candidates = X[labels == 0, :]
    else:
        X_candidates = X

    ### perform prediction
    probs = reg_model.predict_proba(X_candidates)[:, 1]
    ### Get z-score and cut ** Note Not stated in the paper but shown in the code
    probs_zscore = stats.zscore(probs)
    

    ### Get index of z-score where the event is larger than threshold
    changed_labels = np.array([1 if zscore > threshold else 0 for zscore in probs_zscore])
    

    ### Output labels

    labels_output = labels.copy()
    if flip_zeros:
        labels_output[labels == 0] = changed_labels
    else:
        labels_output = changed_labels

    return(labels_output)




    
    






