# CauseText-Replicate

This is a replicate project that aims to replicate the simulation results (Section 6) in the Paper _Causal Effects of Linguistic Properties_ (Pryzant et. al, 2021), and provide a little bit more detailed explanation to the methods from the implementation perspectives. The link to the latest paper is https://doi.org/10.48550/arXiv.2010.12919. 

The purpose of this replication project is to fully understand the different steps in the paper, so I separate them into different python files to provide clarity to the workflow.

## Prepare Simualtion Data

### Data Cleaning and Treatment Label Derivation

_This step is documented in_ `preprocess.py`

Data cleaning is performed according to section 6.1.1 of the original paper, which are:

a. Define True Labels ($\tilde{T}$): label all 5-star comments as 1 and all 1/2 star comments as 0

b. Label Confounder Variable ($C$): label entries about CD as 1 and other as 0

c. Define Proxy-noise labels ($\hat{T}_{noise}$): randomly assigned wrong labels based on True Treatment label.

d. Define Proxy-lex labels ($\hat{T}_{lexicon}$): an indicator of whether there is an overlap with the positive lexicon

__Important Replication Note on Sampling Strategy__ 

According to Pryzant et. al (2021), the dataset `music.tsv` on the original project's github repository (https://github.com/rpryzant/causal-text) will have around 17000 samples after the cleaning. However, taking the same procedure as suggested by the paper, I obtained ~56K samples. 

Upon further examination, in their source code, the authors balanced the dataset by:

a. Dropping some data so the propensity to $P(T=1|C=0) = 0.9$ and $P(T=1|C=1) = 0.7$, and then 

b. For the proxy label, further drop some data such that they have a 0.94 precision and 0.98 recall (They resemble cases when the true_labels can be discovered all the time). 

Using the same setup, I am able to obtain a similarly-sized dataset of ~16K observations.

_Side Note 1:_ I would argue that low recall, high precision should be the feature of the designed proxy labels: the T-boost will improve the estimation by flipping over 0 labels, which presumably helps only when increasing recall is more important. However, consider that this simulated data has much more 1 labels than 0, the adjustment of flipping makes more sense.

### Simulating Y

_This step could be found in_ `parametric_experiment.py`

The outcome variable Y follows the following distribution: 


## TEXTCAUSE: Walking Through the Paper

### T-boosting: Improve the Recall of the Classifier

_This step could be found in_ `parametric_experiment.py`

_Helper functions for this step could be found in_ `t_boost.py`

For the rest of the steps, I develop my results on the proxy lexicon labels ($\hat{T}_{lexicon}$).


The steps include:

a. Using a classifier to predict treatment label, namely predicting $P_{\theta}(\hat{T}=1|W_i)$

1) In this case, the classifier uses binary indicator of whether the text includes the top 2000 words.

2) The paper mentioned simple logistic regression estimator, while the paper also tested PU (Positive and Unlabelled) estimator.

3) I use the PU classifier from the original source code (https://github.com/rpryzant/causal-text/blob/main/src/main.py) I annotate the code, and deleted unnecessary steps (finding optimal in alpha and out alpha for the actual application) for the experiment replication.

b. Flip 0 labels to 1 if $P_{\theta}(T = 1|W_i) > 0.5$. The resulting label is $\hat{T}_{lexicon}^*$ in section 5.1.

**Implementation Notes:**

For this step, the paper doesn't seem to do the exact same thing as described in the dataset. 

1) Section 5.1 suggests a cut off of 0.5 based on the probability. However, the authors are using a cut off of **0.22 on the z-score of the predicted probabilities for observations that have** $\hat{T} = 0$.


### W-adjust: Adjusting for other linguistic properties in the language

The Training steps include: 

1. Padding/Trimming reviews into 128 tokens, and randomly masking 1 of the tokens other than the [CLS] token.

2. Passing the batched samples to BERT and evaluate Masked LM loss: a cross entropy loss on the masked token only (`mlm_loss` in the code, $R(W)$ in the paper)

  - The output embedding of the masked token passes a linear layer of 200, a GeLU layer, a layernorm and final projection to the vocabulary

3. Obtain the embedding of [CLS] token ( $\mathbf{b}(w)$ ) and concatenate with one-hot representation of the confound variable ($C$) (in 0,1 in this replication code, but could be generalized). Namely, $[\mathbf{b}(w)\ \ C]$.

4. At training time, if the example has a positive treatment, pass the example through a linear layer $[M_{1}^b\ \ M_1^c]$ of (embedding_size+2)*200, a Relu layer and a final linear projection layer $M_1^p$ to outcome Y (200 * 2). The result is called $Q(Y|W,\mathbf{b}(w),C, T=1)$. However, if the example has a negative treatment, we pass the example through different layers $[M_{0}^b\ \ M_0^c]$ and $M_0^p$, which is $Q(Y|W,\mathbf{b}(w),C, T=0)$ in the paper. For example $i$, A cross entropy loss was calculated between $Q(Y|W,\mathbf{b}(w),C, T_i)$ and $Y$. This loss is called `Y_loss` in the code.

  - Note: The treatment here can be any label that we have derived from the proxy_labels: proxy_lex (W-adjust in the table), proxy_noise, T_boost (TEXTCAUSE) in the table can all work.

5. Backprop the loss: $\frac{1}{B}\sum_{b}(\beta{L}(Y_b, Q(Y_b|\mathbf{b}(w),C, T_b))+\alpha\cdot{R(w)})$. Empirically, $\alpha =1, \beta = 0.1$. 

  - Note: Again, this $T$ can be any of the following: $\hat{T}^{\*}\_{lex}$, $\hat{T}\_{proxy-lex}$ , $\hat{T}_{proxy-noise}$
  - Additional Note: The original paper doesn't include $\beta$ term. They include it to be compatible with its predecessor.

5. At inference time, we calculate both Q1 and Q0, and, according to the paper, the text adjusted ATE is: $\frac{1}{N}\sum_{i}(Q(Y_b|\mathbf{b}(w),C, T=1) - Q(Y_b|\mathbf{b}(w),C, T=0))$. The authors also provide a platt-scale to scale the results, while I don't implement here, and they also didn't use it here.



## Simulation Results

I replicate the first column with all results but the $\psi_{semi-oracle}$ and the W-adjust results. I ran 10 experiments and report the mean.

The experiment was run on apple M1-pro Chip with pyTorch mps support. The training takes around 24-26 minutes per replicate.

|Replication Number|$\psi_{oracle}$|$\psi_{unadjusted}$|$\psi_{proxy-lex+C}$|$\psi_{proxy-noise+C}$|T-boost-Logit|T-boost-PU|TEXTCAUSE|
|---|---|---|---|---|---|---|---|
|1|9.839|6.155|5.841|6.014|5.966|6.360|6.98|
|2|9.006|6.319|7.222|4.710|7.101|6.898|6.472|
|3|10.253|8.427|7.829|5.820|9.496|8.199|8.494|
