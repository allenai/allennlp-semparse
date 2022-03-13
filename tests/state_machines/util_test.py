import torch

from .. import SemparseTestCase

from allennlp_semparse.state_machines import util


class TestStateMachinesUtil(SemparseTestCase):
    def test_create_allowed_transitions(self):
        targets = torch.Tensor(
            [[[2, 3, 4], [1, 3, 4], [1, 2, 4]], [[3, 4, 0], [2, 3, 4], [0, 0, 0]]]
        )
        target_mask = torch.tensor(
            [
                [[True, True, True], [True, True, True], [True, True, True]],
                [[True, True, False], [True, True, True], [False, False, False]],
            ]
        )
        prefix_tree = util.construct_prefix_tree(targets, target_mask)

        # There were two instances in this batch.
        assert len(prefix_tree) == 2

        # The first instance had six valid action sequence prefixes.
        assert len(prefix_tree[0]) == 6
        assert prefix_tree[0][()] == {1, 2}
        assert prefix_tree[0][(1,)] == {2, 3}
        assert prefix_tree[0][(1, 2)] == {4}
        assert prefix_tree[0][(1, 3)] == {4}
        assert prefix_tree[0][(2,)] == {3}
        assert prefix_tree[0][(2, 3)] == {4}

        # The second instance had four valid action sequence prefixes.
        assert len(prefix_tree[1]) == 4
        assert prefix_tree[1][()] == {2, 3}
        assert prefix_tree[1][(2,)] == {3}
        assert prefix_tree[1][(2, 3)] == {4}
        assert prefix_tree[1][(3,)] == {4}
