# allennlp-semparse
[![Build status](https://github.com/allenai/allennlp-semparse/workflows/CI/badge.svg)](https://github.com/allenai/allennlp-semparse/actions?workflow=CI)
[![PyPI](https://img.shields.io/pypi/v/allennlp-semparse)](https://pypi.org/project/allennlp-semparse/)
[![codecov](https://codecov.io/gh/allenai/allennlp-semparse/branch/master/graph/badge.svg)](https://codecov.io/gh/allenai/allennlp-semparse)

A framework for building semantic parsers (including neural module networks) with AllenNLP, built by the authors of AllenNLP

## Installing

`allennlp-semparse` is available on PyPI. You can install through `pip` with

```
pip install allennlp-semparse
```

## Supported datasets

- ATIS
- Text2SQL
- NLVR
- WikiTableQuestions

## Supported models

- Grammar-based decoding models, following the parser originally introduced in [Neural
Semantic Parsing with Type Constraints for Semi-Structured
Tables](https://www.semanticscholar.org/paper/Neural-Semantic-Parsing-with-Type-Constraints-for-Krishnamurthy-Dasigi/8c6f58ed0ebf379858c0bbe02c53ee51b3eb398a).
The models that are currently checked in are all based on this parser, applied to various datasets.
- Neural module networks.  We don't have models checked in for this yet, but `DomainLanguage`
  supports defining them, and we will add some models to the repo once papers go through peer
review.  The code is slow (batching is hard), but it works.


## Tutorials

Coming sometime in the future...  You can look at [this old
tutorial](https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/semantic_parsing.md),
but the part about using NLTK to define a grammar is outdated.  Now you can use `DomainLanguage` to
define a python executor, and we analyze the type annotations in the functions in that executor to
automatically infer a grammar for you.  It is much easier to use than it used to be.  Until we get
around to writing a better tutorial for this, the best way to get started using this is to look at
some examples.  The simplest is the [Arithmetic
language](https://github.com/allenai/allennlp-semparse/blob/master/tests/domain_languages/domain_language_test.py)
in the `DomainLanguage` test (there's also a bit of description in the [`DomainLanguage`
docstring](https://github.com/allenai/allennlp-semparse/blob/bbc8fde3a354ee1708ae900f09be9aa2adc8177f/allennlp_semparse/domain_languages/domain_language.py#L204-L270)).
After looking at those, you can look at more complex (real) examples in the [`domain_languages`
module](https://github.com/allenai/allennlp-semparse/tree/master/allennlp_semparse/domain_languages).
Note that the executor you define can have _learned parameters_, making it a neural module network.
The best place to get an example of that is currently [this unfinished implementation of N2NMNs on
the CLEVR
dataset](https://github.com/matt-gardner/allennlp/blob/neural_module_networks/allennlp/semparse/domain_languages/visual_reasoning_language.py).
We'll have more examples of doing this in the not-too-distant future.
