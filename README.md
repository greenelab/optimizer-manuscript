# Optimizer’s dilemma: optimization strongly influences model selection in transcriptomic prediction

[![HTML Manuscript](https://img.shields.io/badge/manuscript-HTML-blue.svg)](https://greenelab.github.io/optimizer-manuscript/)
[![PDF Manuscript](https://img.shields.io/badge/manuscript-PDF-blue.svg)](https://greenelab.github.io/optimizer-manuscript/manuscript.pdf)
[![GitHub Actions Status](https://github.com/greenelab/optimizer-manuscript/workflows/Manubot/badge.svg)](https://github.com/greenelab/optimizer-manuscript/actions)

## Project description

We applied two different implementations of LASSO logistic regression implemented in Python’s scikit-learn package, using two different optimization approaches (coordinate descent and stochastic gradient descent), to predict driver mutation presence or absence from gene expression across 84 pan-cancer driver genes.
For varying levels of regularization, we compared performance and model sparsity between optimizers.

After model selection and tuning, we found that coordinate descent (implemented in the `liblinear` library) and SGD tended to perform comparably.
`liblinear` models required more extensive tuning of regularization strength, performing best for high model sparsities (more nonzero coefficients), but did not require selection of a learning rate parameter.
SGD models required tuning of the learning rate to perform well, but generally performed more robustly across different model sparsities as regularization strength decreased.
Given these tradeoffs, we believe that the choice of optimizers should be clearly reported as a part of the model selection and validation process, to allow readers and reviewers to better understand the context in which results have been generated.

The Github repository with the source code for the analyses can be found [here.](https://github.com/greenelab/pancancer-evaluation/tree/master/01_stratified_classification)

## Manubot

<!-- usage note: do not edit this section -->

Manubot is a system for writing scholarly manuscripts via GitHub.
Manubot automates citations and references, versions manuscripts using git, and enables collaborative writing via GitHub.
An [overview manuscript](https://greenelab.github.io/meta-review/ "Open collaborative writing with Manubot") presents the benefits of collaborative writing with Manubot and its unique features.
The [rootstock repository](https://git.io/fhQH1) is a general purpose template for creating new Manubot instances, as detailed in [`SETUP.md`](SETUP.md).
See [`USAGE.md`](USAGE.md) for documentation how to write a manuscript.

Please open [an issue](https://git.io/fhQHM) for questions related to Manubot usage, bug reports, or general inquiries.

### Repository directories & files

The directories are as follows:

+ [`content`](content) contains the manuscript source, which includes markdown files as well as inputs for citations and references.
  See [`USAGE.md`](USAGE.md) for more information.
+ [`output`](output) contains the outputs (generated files) from Manubot including the resulting manuscripts.
  You should not edit these files manually, because they will get overwritten.
+ [`webpage`](webpage) is a directory meant to be rendered as a static webpage for viewing the HTML manuscript.
+ [`build`](build) contains commands and tools for building the manuscript.
+ [`ci`](ci) contains files necessary for deployment via continuous integration.

### Local execution

The easiest way to run Manubot is to use [continuous integration](#continuous-integration) to rebuild the manuscript when the content changes.
If you want to build a Manubot manuscript locally, install the [conda](https://conda.io) environment as described in [`build`](build).
Then, you can build the manuscript on POSIX systems by running the following commands from this root directory.

```sh
# Activate the manubot conda environment (assumes conda version >= 4.4)
conda activate manubot

# Build the manuscript, saving outputs to the output directory
bash build/build.sh

# At this point, the HTML & PDF outputs will have been created. The remaining
# commands are for serving the webpage to view the HTML manuscript locally.
# This is required to view local images in the HTML output.

# Configure the webpage directory
manubot webpage

# You can now open the manuscript webpage/index.html in a web browser.
# Alternatively, open a local webserver at http://localhost:8000/ with the
# following commands.
cd webpage
python -m http.server
```

Sometimes it's helpful to monitor the content directory and automatically rebuild the manuscript when a change is detected.
The following command, while running, will trigger both the `build.sh` script and `manubot webpage` command upon content changes:

```sh
bash build/autobuild.sh
```

### Continuous Integration

Whenever a pull request is opened, CI (continuous integration) will test whether the changes break the build process to generate a formatted manuscript.
The build process aims to detect common errors, such as invalid citations.
If your pull request build fails, see the CI logs for the cause of failure and revise your pull request accordingly.

When a commit to the `main` branch occurs (for example, when a pull request is merged), CI builds the manuscript and writes the results to the [`gh-pages`](https://github.com/greenelab/optimizer-manuscript/tree/gh-pages) and [`output`](https://github.com/greenelab/optimizer-manuscript/tree/output) branches.
The `gh-pages` branch uses [GitHub Pages](https://pages.github.com/) to host the following URLs:

+ **HTML manuscript** at https://greenelab.github.io/optimizer-manuscript/
+ **PDF manuscript** at https://greenelab.github.io/optimizer-manuscript/manuscript.pdf

For continuous integration configuration details, see [`.github/workflows/manubot.yaml`](.github/workflows/manubot.yaml).

## License

<!--
usage note: edit this section to change the license of your manuscript or source code changes to this repository.
We encourage users to openly license their manuscripts, which is the default as specified below.
-->

[![License: CC BY 4.0](https://img.shields.io/badge/License%20All-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)
[![License: CC0 1.0](https://img.shields.io/badge/License%20Parts-CC0%201.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

Except when noted otherwise, the entirety of this repository is licensed under a CC BY 4.0 License ([`LICENSE.md`](LICENSE.md)), which allows reuse with attribution.
Please attribute by linking to https://github.com/greenelab/optimizer-manuscript.

Since CC BY is not ideal for code and data, certain repository components are also released under the CC0 1.0 public domain dedication ([`LICENSE-CC0.md`](LICENSE-CC0.md)).
All files matched by the following glob patterns are dual licensed under CC BY 4.0 and CC0 1.0:

+ `*.sh`
+ `*.py`
+ `*.yml` / `*.yaml`
+ `*.json`
+ `*.bib`
+ `*.tsv`
+ `.gitignore`

All other files are only available under CC BY 4.0, including:

+ `*.md`
+ `*.html`
+ `*.pdf`
+ `*.docx`

Please open [an issue](https://github.com/greenelab/optimizer-manuscript/issues) for any question related to licensing.
