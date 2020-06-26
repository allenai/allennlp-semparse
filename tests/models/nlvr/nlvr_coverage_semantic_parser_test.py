from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from ... import ModelTestCase
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive


class NlvrCoverageSemanticParserTest(ModelTestCase):
    def setup_method(self):
        super(NlvrCoverageSemanticParserTest, self).setup_method()
        self.set_up_model(
            self.FIXTURES_ROOT / "nlvr_coverage_semantic_parser" / "experiment.json",
            self.FIXTURES_ROOT / "data" / "nlvr" / "sample_grouped_data.jsonl",
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_ungrouped_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.FIXTURES_ROOT / "nlvr_coverage_semantic_parser" / "ungrouped_experiment.json"
        )

    def test_mml_initialized_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.FIXTURES_ROOT / "nlvr_coverage_semantic_parser" / "mml_init_experiment.json"
        )

    def test_get_checklist_info(self):
        # Creating a fake all_actions field where actions 0, 2 and 4 are terminal productions.
        all_actions = [
            ("<Set[Object]:Set[Object]> -> top", True, None),
            ("fake_action", True, None),
            ("Color -> color_black", True, None),
            ("fake_action2", True, None),
            ("int -> 6", True, None),
        ]
        # Of the actions above, those at indices 0 and 4 are on the agenda, and there are padding
        # indices at the end.
        test_agenda = torch.Tensor([[0], [4], [-1], [-1]])
        checklist_info = self.model._get_checklist_info(test_agenda, all_actions)
        target_checklist, terminal_actions, checklist_mask = checklist_info
        assert_almost_equal(target_checklist.data.numpy(), [[1], [0], [1]])
        assert_almost_equal(terminal_actions.data.numpy(), [[0], [2], [4]])
        assert_almost_equal(checklist_mask.data.numpy(), [[1], [1], [1]])

    def test_initialize_weights_from_archive(self):
        original_model_parameters = self.model.named_parameters()
        original_model_weights = {
            name: parameter.data.clone().numpy() for name, parameter in original_model_parameters
        }
        mml_model_archive_file = (
            self.FIXTURES_ROOT / "nlvr_direct_semantic_parser" / "serialization" / "model.tar.gz"
        )
        archive = load_archive(mml_model_archive_file)
        archived_model_parameters = archive.model.named_parameters()
        self.model._initialize_weights_from_archive(archive)
        changed_model_parameters = dict(self.model.named_parameters())
        for name, archived_parameter in archived_model_parameters:
            archived_weight = archived_parameter.data.numpy()
            original_weight = original_model_weights[name]
            changed_weight = changed_model_parameters[name].data.numpy()
            # We want to make sure that the weights in the original model have indeed been changed
            # after a call to ``_initialize_weights_from_archive``.
            with self.assertRaises(AssertionError, msg=f"{name} has not changed"):
                assert_almost_equal(original_weight, changed_weight)
            # This also includes the sentence token embedder. Those weights will be the same
            # because the two models have the same vocabulary.
            assert_almost_equal(archived_weight, changed_weight)

    def test_get_vocab_index_mapping(self):
        mml_model_archive_file = (
            self.FIXTURES_ROOT / "nlvr_direct_semantic_parser" / "serialization" / "model.tar.gz"
        )
        archive = load_archive(mml_model_archive_file)
        mapping = self.model._get_vocab_index_mapping(archive.model.vocab)
        expected_mapping = [(i, i) for i in range(16)]
        assert mapping == expected_mapping

        new_vocab = Vocabulary()

        def copy_token_at_index(i):
            token = self.vocab.get_token_from_index(i, "tokens")
            new_vocab.add_token_to_namespace(token, "tokens")

        copy_token_at_index(5)
        copy_token_at_index(7)
        copy_token_at_index(10)
        mapping = self.model._get_vocab_index_mapping(new_vocab)
        # Mapping of indices from model vocabulary to new vocabulary. 0 and 1 are padding and unk
        # tokens.
        assert mapping == [(0, 0), (1, 1), (5, 2), (7, 3), (10, 4)]
