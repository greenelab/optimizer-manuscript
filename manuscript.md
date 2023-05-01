---
title: Optimizers manuscript
keywords:
- markdown
- publishing
- manubot
lang: en-US
date-meta: '2023-05-01'
author-meta:
- Jake Crawford
- Casey S. Greene
header-includes: |
  <!--
  Manubot generated metadata rendered from header-includes-template.html.
  Suggest improvements at https://github.com/manubot/manubot/blob/main/manubot/process/header-includes-template.html
  -->
  <meta name="dc.format" content="text/html" />
  <meta property="og:type" content="article" />
  <meta name="dc.title" content="Optimizers manuscript" />
  <meta name="citation_title" content="Optimizers manuscript" />
  <meta property="og:title" content="Optimizers manuscript" />
  <meta property="twitter:title" content="Optimizers manuscript" />
  <meta name="dc.date" content="2023-05-01" />
  <meta name="citation_publication_date" content="2023-05-01" />
  <meta property="article:published_time" content="2023-05-01" />
  <meta name="dc.modified" content="2023-05-01T14:04:45+00:00" />
  <meta property="article:modified_time" content="2023-05-01T14:04:45+00:00" />
  <meta name="dc.language" content="en-US" />
  <meta name="citation_language" content="en-US" />
  <meta name="dc.relation.ispartof" content="Manubot" />
  <meta name="dc.publisher" content="Manubot" />
  <meta name="citation_journal_title" content="Manubot" />
  <meta name="citation_technical_report_institution" content="Manubot" />
  <meta name="citation_author" content="Jake Crawford" />
  <meta name="citation_author_institution" content="Genomics and Computational Biology Graduate Group, Perelman School of Medicine, University of Pennsylvania, Philadelphia, PA, USA" />
  <meta name="citation_author_orcid" content="0000-0001-6207-0782" />
  <meta name="twitter:creator" content="@jjc2718" />
  <meta name="citation_author" content="Casey S. Greene" />
  <meta name="citation_author_institution" content="Department of Biochemistry and Molecular Genetics, University of Colorado School of Medicine, Aurora, CO, USA" />
  <meta name="citation_author_institution" content="Center for Health AI, University of Colorado School of Medicine, Aurora, CO, USA" />
  <meta name="citation_author_orcid" content="0000-0001-8713-9213" />
  <meta name="twitter:creator" content="@GreeneScientist" />
  <link rel="canonical" href="https://greenelab.github.io/optimizer-manuscript/" />
  <meta property="og:url" content="https://greenelab.github.io/optimizer-manuscript/" />
  <meta property="twitter:url" content="https://greenelab.github.io/optimizer-manuscript/" />
  <meta name="citation_fulltext_html_url" content="https://greenelab.github.io/optimizer-manuscript/" />
  <meta name="citation_pdf_url" content="https://greenelab.github.io/optimizer-manuscript/manuscript.pdf" />
  <link rel="alternate" type="application/pdf" href="https://greenelab.github.io/optimizer-manuscript/manuscript.pdf" />
  <link rel="alternate" type="text/html" href="https://greenelab.github.io/optimizer-manuscript/v/c5471d9871088ab81646ad28a6f0b6cbfedb322c/" />
  <meta name="manubot_html_url_versioned" content="https://greenelab.github.io/optimizer-manuscript/v/c5471d9871088ab81646ad28a6f0b6cbfedb322c/" />
  <meta name="manubot_pdf_url_versioned" content="https://greenelab.github.io/optimizer-manuscript/v/c5471d9871088ab81646ad28a6f0b6cbfedb322c/manuscript.pdf" />
  <meta property="og:type" content="article" />
  <meta property="twitter:card" content="summary_large_image" />
  <link rel="icon" type="image/png" sizes="192x192" href="https://manubot.org/favicon-192x192.png" />
  <link rel="mask-icon" href="https://manubot.org/safari-pinned-tab.svg" color="#ad1457" />
  <meta name="theme-color" content="#ad1457" />
  <!-- end Manubot generated metadata -->
bibliography:
- content/manual-references.json
manubot-output-bibliography: output/references.json
manubot-output-citekeys: output/citations.tsv
manubot-requests-cache-path: ci/cache/requests-cache
manubot-clear-requests-cache: false
...






<small><em>
This manuscript
([permalink](https://greenelab.github.io/optimizer-manuscript/v/c5471d9871088ab81646ad28a6f0b6cbfedb322c/))
was automatically generated
from [greenelab/optimizer-manuscript@c5471d9](https://github.com/greenelab/optimizer-manuscript/tree/c5471d9871088ab81646ad28a6f0b6cbfedb322c)
on May 1, 2023.
</em></small>



## Authors



+ **Jake Crawford**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0000-0001-6207-0782](https://orcid.org/0000-0001-6207-0782)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [jjc2718](https://github.com/jjc2718)
    · ![Twitter icon](images/twitter.svg){.inline_icon width=16 height=16}
    [jjc2718](https://twitter.com/jjc2718)
    <br>
  <small>
     Genomics and Computational Biology Graduate Group, Perelman School of Medicine, University of Pennsylvania, Philadelphia, PA, USA
  </small>

+ **Casey S. Greene**
  ^[✉](#correspondence)^<br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0000-0001-8713-9213](https://orcid.org/0000-0001-8713-9213)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [cgreene](https://github.com/cgreene)
    · ![Twitter icon](images/twitter.svg){.inline_icon width=16 height=16}
    [GreeneScientist](https://twitter.com/GreeneScientist)
    <br>
  <small>
     Department of Biochemistry and Molecular Genetics, University of Colorado School of Medicine, Aurora, CO, USA; Center for Health AI, University of Colorado School of Medicine, Aurora, CO, USA
  </small>


::: {#correspondence}
✉ — Correspondence possible via [GitHub Issues](https://github.com/greenelab/optimizer-manuscript/issues)
or email to
Casey S. Greene \<casey.s.greene@cuanschutz.edu\>.


:::


## Abstract {.page_break_before}




## Introduction {.page_break_before}

Gene expression profiles are widely used to classify samples or patients into relevant groups or categories, both preclinically [@doi:10.1371/journal.pcbi.1009926; @doi:10.1093/bioinformatics/btaa150] and clinically [@doi:10.1200/JCO.2008.18.1370; @doi:10/bp4rtw].
To extract informative gene features and to perform classification, a diverse array of algorithms exist, and different algorithms perform well across varying datasets and tasks [@doi:10.1371/journal.pcbi.1009926].
Even within a given model class, multiple optimization methods can often be applied to find well-performing model parameters or to optimize a model's loss function.
One commonly used example is logistic regression.
The widely used scikit-learn Python package for machine learning [@url:https://jmlr.org/papers/v12/pedregosa11a.html] provides two modules for fitting logistic regression classifiers: `LogisticRegression`, which uses the `liblinear` coordinate descent method [@url:https://www.jmlr.org/papers/v9/fan08a.html] to find parameters that optimize the logistic loss function, and `SGDClassifier`, which uses stochastic gradient descent [@online-learning] to optimize the same loss function.

Using scikit-learn, we compared the `liblinear` (coordinate descent) and SGD optimization techniques for prediction of driver mutation status in tumor samples, across a wide variety of genes implicated in cancer initiation and development [@doi:10.1126/science.1235122].
We applied LASSO (L1-regularized) logistic regression, and tuned the strength of the regularization to compare model selection between optimizers.
We found that across a variety of models (i.e. varying regularization strengths), the training dynamics of the optimizers were considerably different: models fit using `liblinear` tended to perform best at fairly high regularization strengths (100-1000 nonzero features in the model) and overfit easily with low regularization strengths.
On the other hand, models fit using stochastic gradient descent tended to perform best at fairly low regularization strengths (10000+ nonzero features in the model), and overfitting was uncommon.

Our results caution against viewing optimizer choice as a "black box" component of machine learning modeling.
The observation that LASSO logistic regression models fit using SGD tended to perform best for low levels of regularization, across diverse driver genes, runs counter to conventional wisdom in machine learning for high-dimensional data which generally states that explicit regularization and/or feature selection is necessary.
Comparing optimizers/model implementations directly is rare in applications of machine learning for genomics, and our work shows that this choice can affect generalization and interpretation properties of the model significantly.
Based on our results, we recommend considering the appropriate optimization approach carefully based on the goals of each individual analysis.

## Methods {.page_break_before}

## Results {.page_break_before}

For each of the 125 driver genes from the Vogelstein et al. 2013 paper, we trained models to predict mutation status (presence or absence) from RNA-seq data, derived from the TCGA Pan-Cancer Atlas.
For each optimizer, we trained LASSO logistic regression models across a variety of regularization parameters (see Methods for parameter range details), for 4 cross-validation splits x 2 replicates (random seeds) for a total of 8 different models per parameter.
Cross-validation splits were stratified by cancer type.

Previous work has shown that pan-cancer classifiers of Ras mutation status are accurate and biologically informative [@doi:10.1016/j.celrep.2018.03.046].
As model complexity increases (more nonzero coefficients) for the `liblinear` optimizer, we observe that performance increases then decreases, corresponding to overfitting for high model complexities/numbers of nonzero coefficients (Figure {@fig:optimizer_compare_mutations}A).
.
On the other hand, for the SGD optimizer, we observe an increase in performance as model complexity increases, with models having no nonzero coefficients performing the best (Figure {@fig:optimizer_compare_mutations}B).
In this case, top performance for SGD (the largest bin, i.e. furthest right on the x-axis) is slightly worse than top performance for `liblinear` (the third smallest bin): we observed a mean test AUPR of 0.618 for SGD vs. mean AUPR of 0.688 for `liblinear`.
As model complexity varies, similar performance trends tend to hold across a variety of driver genes in the Vogelstein dataset, and for a variety of approaches to quantifying model complexity (see Supplementary Data).

To determine if the relative performance improvement with `liblinear` tends to hold across the genes in the Vogelstein dataset at large, we compared performance for the best-performing models for each gene, between optimizers.
Figure {@fig:optimizer_compare_mutations}C shows the distribution of differences in performance across genes.
The distribution is generally shifted to the right, suggesting that `liblinear` generally tends to outperform SGD.
We saw that for 71/84 genes, performance for the best-performing model was better using `liblinear` than SGD, and for the other 13 genes performance was better using SGD.

![
**A.** Performance vs. model complexity (number of nonzero coefficients) for KRAS mutation status prediction, for `liblinear` optimizer. Bins are derived from deciles of coefficient count distribution across optimizers; additional detail in Methods. "Holdout" dataset is used in panel C and following figures for best-performing model selection, "test" data is completely held out from model selection and used for evaluation in panel C and following figures.
**B.** Performance vs. model complexity (number of nonzero coefficients) for KRAS mutation status prediction, for SGD optimizer.
**C.** Distribution of performance difference between best-performing model for `liblinear` and SGD optimizers, across all 84 genes in Vogelstein driver gene set. Positive numbers on the x-axis indicate better performance using `liblinear`, and negative numbers indicate better performance using SGD.
](images/figure_1.png){#fig:optimizer_compare_mutations width="100%"}

We next sought to determine whether there was a difference in the magnitudes of coefficients in the models resulting from the different optimization schemes.
Following up on the trend in Figure {@fig:optimizer_compare_mutations}, where we saw that the best-performing SGD model had many nonzero coefficients, we also see that in general across all genes, the best-performing SGD models tend to be bimodal, sometimes having few nonzero coefficients but often having many/all nonzero coefficients (Figure {@fig:optimizer_coefs}A).
By contrast, the `liblinear` models are almost always much sparser with fewer than 2500 nonzero coefficients, out of ~16100 total input features.

Despite the SGD models performing best with many nonzero coefficients, it could be the case that many of the coefficients could be "effectively" 0, or uninformative to the final model.
However, Figure {@fig:optimizer_coefs}B provides evidence that this is not the case, with most coefficients in the best-performing KRAS mutation prediction model using SGD being considerably larger than the coefficients in the best-performing model using `liblinear`, and very few close to 0.
This emphasizes that the different optimization methods result in fundamentally different models, relying on different numbers of features with nonzero coefficients in different magnitudes, rather than converging to similar models.

![
**A.** Distribution across genes of the number of nonzero coefficients included in best-performing LASSO logistic regression models. Violin plot density estimations are clipped at the ends of the observed data range, and boxes show the median/IQR.
**B.** Distribution of coefficient magnitudes for a single KRAS mutation prediction model (random seed 42, first cross-validation split), colored by optimizer. The x-axis shows the base-10 logarithm of the absolute value of each coefficient + 1 (since some coefficients are exactly 0), and the y-axis shows the base-10 log of the count of coefficients in each bin. Other random seeds and cross-validation splits are similar.
](images/figure_2.png){#fig:optimizer_coefs width="80%"}

## Discussion {.page_break_before}

Our results suggest that even for the same model, LASSO logistic regression, optimizer choice can affect model selection and performance.
Existing gene expression prediction benchmarks and pipelines typically use a single model implementation (and thus a single optimizer).
To our knowledge, the phenomenon we observed with SGD has not been documented in other applications of ML to genomic or transcriptomic data.
In the broader machine learning research community, however, similar patterns have been observed for both linear models and deep neural networks (e.g. [@doi:10.1073/pnas.1907378117; @arxiv:1611.03530]).
This is often termed "benign overfitting": the idea that "overfit" models, in the sense that they fit the training data perfectly and perform worse on the test data, can still outperform models that do not fit the training data as well or that have stronger explicit regularization.
Benign overfitting has been observed with, and attributed to, optimization using SGD, which is thought to provide a form of implicit regularization [@doi:10.1145/3446776; @url:http://proceedings.mlr.press/v134/zou21a.html].

We recommend thinking critically about optimizer choice, but this can be challenging for users that are inexperienced with machine learning or unfamiliar with how certain models are fit under the hood.
For example, R's `glmnet` package uses a cyclical coordinate descent algorithm to fit logistic regression models [@doi:10.18637/jss.v033.i01], which would presumably behave similarly to `liblinear`, but this is somewhat opaque in the `glmnet` documentation itself.
LASSO logistic regression is a convex optimization problem, meaning there is a single unique optimum of the loss function in contrast to more complex models such as neural networks, but this optimum can be computationally intensive to find in practice and there is no closed-form solution [@l1-logistic-regression].
Increased transparency and documentation in popular machine learning packages with respect to optimization, especially for models that are challenging to fit, would benefit new and unfamiliar users.


Similar to what we see in our SGD-optimized models, there exist other problems in gene expression analysis where using all available features is better than using a subset.
For example, using the full gene set improves correlations between preclinical cancer models and their tissue of origin, as compared to selecting genes based on variability or tissue-specificity [@doi:10.1101/2023.04.11.536431].
On the other hand, when predicting cell line viability from gene expression profiles, selecting features by Pearson correlation improves performance over using all features, similar to our `liblinear` classifiers [@doi:10.1101/2020.02.21.959627].
An avenue of future work for our SGD classifiers would be to interpret the coefficients and compare them systematically to the coefficients found using `liblinear`.
It could be useful to understand if the two optimization methods emphasize the same pathways or functional gene sets, or if there are patterns to which driver mutations perform better with more/fewer nonzero coefficients.



## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>

