---
title: Optimizers manuscript
keywords:
- markdown
- publishing
- manubot
lang: en-US
date-meta: '2023-04-25'
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
  <meta name="dc.date" content="2023-04-25" />
  <meta name="citation_publication_date" content="2023-04-25" />
  <meta property="article:published_time" content="2023-04-25" />
  <meta name="dc.modified" content="2023-04-25T21:20:55+00:00" />
  <meta property="article:modified_time" content="2023-04-25T21:20:55+00:00" />
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
  <link rel="alternate" type="text/html" href="https://greenelab.github.io/optimizer-manuscript/v/b2b08344404e818e79dae4d525dbbb4484c8c59a/" />
  <meta name="manubot_html_url_versioned" content="https://greenelab.github.io/optimizer-manuscript/v/b2b08344404e818e79dae4d525dbbb4484c8c59a/" />
  <meta name="manubot_pdf_url_versioned" content="https://greenelab.github.io/optimizer-manuscript/v/b2b08344404e818e79dae4d525dbbb4484c8c59a/manuscript.pdf" />
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
([permalink](https://greenelab.github.io/optimizer-manuscript/v/b2b08344404e818e79dae4d525dbbb4484c8c59a/))
was automatically generated
from [greenelab/optimizer-manuscript@b2b0834](https://github.com/greenelab/optimizer-manuscript/tree/b2b08344404e818e79dae4d525dbbb4484c8c59a)
on April 25, 2023.
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
The observation that LASSO logistic regression models fit using SGD tended to perform best for low levels of regularization, across diverse driver genes, runs counter to conventional wisdom in statistics and machine learning for high-dimensional data which generally states that explicit regularization and/or feature selection is necessary.
Comparing optimizers/model implementations directly is rare in applications of machine learning for genomics, and our work shows that this choice can affect generalization and interpretation properties of the model significantly.
Based on our results, we recommend considering the appropriate optimization approach carefully based on the goals of each individual analysis.

## Methods {.page_break_before}

## Results {.page_break_before}

## Discussion {.page_break_before}


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>

