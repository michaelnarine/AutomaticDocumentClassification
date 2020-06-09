# Automatic Document Classification

This program implements an extension of the k-Nearest Neighbors Automatic Document Classification (kNN ADC) technique developed in the paper "Efficient and Scalable MetaFeature-based Document Classification using Massively Parallel Computing" [[1]](#1).

This program extended the kNN ADC technique by modifying the term and document norms and weights when computing the similarity values for each term document pair in a query document to documents in a training set containing these terms.

This program consists of a python script, which generates the term-document pairs, term-frequencies, and term-norms for each document in the training set, and a CUDA program that classifies a query document using parallel computing techniques.

## References
<a id="1">[1]</a> 
Canuto, S., Gonçalves, M., Santos, W., Rosa, T., & Martins, W. (2015, August). An efficient and scalable metafeature-based document classification approach based on massively parallel computing. In Proceedings 9 of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 333-342).
