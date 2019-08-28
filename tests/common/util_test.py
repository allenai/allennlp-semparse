# pylint: disable=no-self-use,invalid-name
from .. import SemparseTestCase

from allennlp_semparse.common import util


class TestSemparseUtil(SemparseTestCase):
    def test_lisp_to_nested_expression(self):
        logical_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = util.lisp_to_nested_expression(logical_form)
        assert expression == [['reverse', 'fb:row.row.year'], ['fb:row.row.league', 'fb:cell.usl_a_league']]
        logical_form = "(count (and (division 1) (tier (!= null))))"
        expression = util.lisp_to_nested_expression(logical_form)
        assert expression == ['count', ['and', ['division', '1'], ['tier', ['!=', 'null']]]]
