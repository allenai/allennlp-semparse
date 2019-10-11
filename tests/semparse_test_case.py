import pathlib

from allennlp.common.testing import AllenNlpTestCase, ModelTestCase as AllenNlpModelTestCase

# These imports are to get all of the items registered that we need.
from allennlp_semparse import models, dataset_readers, predictors

ROOT = (pathlib.Path(__file__).parent / "..").resolve()


class SemparseTestCase(AllenNlpTestCase):
    PROJECT_ROOT = ROOT
    MODULE_ROOT = PROJECT_ROOT / "allennlp_semparse"
    TOOLS_ROOT = None  # just removing the reference from super class
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"


class ModelTestCase(AllenNlpModelTestCase):
    PROJECT_ROOT = ROOT
    MODULE_ROOT = PROJECT_ROOT / "allennlp_semparse"
    TOOLS_ROOT = None  # just removing the reference from super class
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"
