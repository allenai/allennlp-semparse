from flaky import flaky

from ... import ModelTestCase


class TestWikiTablesVariableFreeErm(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        config_path = self.FIXTURES_ROOT / "wikitables" / "experiment-erm.json"
        data_path = self.FIXTURES_ROOT / "data" / "wikitables" / "sample_data.examples"
        self.set_up_model(config_path, data_path)

    @flaky
    def test_model_can_train_save_and_load(self):
        # We have very few embedded actions on our agenda, and so it's rare that this parameter
        # actually gets used.  We know this parameter works from our NLVR ERM test, so it's easier
        # to just ignore it here than to try to finagle the test to make it so this has a non-zero
        # gradient.
        ignore = {"_decoder_step._checklist_multiplier"}
        self.ensure_model_can_train_save_and_load(self.param_file, gradients_to_ignore=ignore)
