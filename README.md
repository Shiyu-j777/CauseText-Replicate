# CauseText-Replicate

This is a replicate project that aims to replicate the simulation results (Section 6) in the Paper _Causal Effects of Linguistic Properties_ (Pryzant et. al, 2021). The link to the latest paper is https://doi.org/10.48550/arXiv.2010.12919. 

## Data Cleaning

Data cleaning is performed according to section 6.1.1 of the original paper, which are:

1) Define True Labels ($\tilde{T}$): label all 5-star comments as 1 and all 1/2 star comments as 0

2) Label Confounder Variable ($C$): label entries about CD as 1 and other as 0

3) Define Proxy-noise labels ($\hat{T}_{noise}$): randomly assigned wrong labels based on True Treatment label.

4) Define Proxy-lex labels ($\hat{T}_{lexicon}$): an indicator of whether there is an overlap with the positive lexicon

__Important Replication Note on Sampling Strategy__ 

According to Pryzant et. al (2021), the dataset `music.tsv` on the original project's github repository (https://github.com/rpryzant/causal-text) will have around 17000 samples after the cleaning. However, taking the same procedure as suggested by the paper, I obtained ~56K samples. 

Upon further examination, in their source code, the authors balanced the dataset by:

1) Dropping some data so the propensity to $P(T=1|C=0) = 0.9$ and $P(T=1|C=1) = 0.7$, and then 

2) For the proxy label, further drop some data such that they have a 0.94 precision and 0.98 recall (They resemble cases when the true_labels can be discovered all the time). 

Using the same setup, I am able to obtain a similarly-sized dataset of ~16K observations.

_Side Note 1:_ I would argue that low recall, high precision should be the feature of the designed proxy labels: the T-boost will improve the estimation by flipping over 0 labels, which presumably helps only when increasing recall is more important. However, consider that this simulated data has much more 1 labels than 0, the adjustment of flipping makes more sense.


## T-boosting

For the rest of the steps, I develop my results on the proxy lexicon labels ($\hat{T}_{lexicon}$).

The paper mentioned simple logistic regression estimator, while the paper also tested PU estimator.

The steps include:

1) Using a classifier to predict treatment label

> In this case, the classifier uses binary indicator of whether the text includes the top 2000 words.

2) Flip 0 labels to 1 if $P_{\theta}(T = 1|W_i) > 0.5$. The resulting label is $\hat{T}_{lexicon}^*$ in section 5.1.

**Implementation Notes:**

For this step, the paper doesn't seem to do the exact same thing as described in the dataset. 

1) Section 5.1 suggests a cut off of 0.5 based on the probability. However, the authors are using a cut off of **0.22 on the z-score of the predicted probabilities for observations that have** $\hat{T} = 0$.





