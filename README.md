# CauseText-Replicate

This is a replicate project that aims to replicate the simulation results in the Paper _Causal Effects of Linguistic Properties_ (Pryzant et. al, 2021). The link to the latest paper is https://doi.org/10.48550/arXiv.2010.12919. 

## Data Cleaning

Data cleaning is performed according to section 6.1.1 of the original paper, which are:

1) Define True Labels ($\tilde{T}$): label all 5-star comments as 1 and all 1/2 star comments as 0

2) Label Confounder Variable ($C$): label entries about CD as 1 and other as 0

3) Define Proxy-noise labels ($\hat{T}_{noise}$): randomly assigned wrong labels based on True Treatment label.

_Note:_ According to Pryzant et. al (2021), the dataset `music.tsv` on the original project's github repository (https://github.com/rpryzant/causal-text) will have around 17000 samples after the cleaning. However, taking the same procedure, I obtained ~56K samples. 
