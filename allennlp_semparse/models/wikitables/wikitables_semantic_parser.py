from typing import Any, Dict, List, Mapping, Sequence, Tuple


import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (
    Embedding,
    Seq2SeqEncoder,
    Seq2VecEncoder,
    TextFieldEmbedder,
    TimeDistributed,
)
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util
from allennlp.training.metrics import Average

from allennlp_semparse.common import ParsingError
from allennlp_semparse.domain_languages import WikiTablesLanguage, START_SYMBOL
from allennlp_semparse.domain_languages.domain_language import ExecutionError
from allennlp_semparse.fields.production_rule_field import ProductionRuleArray
from allennlp_semparse.state_machines.states import GrammarBasedState, GrammarStatelet, RnnStatelet


class WikiTablesSemanticParser(Model):
    """
    A ``WikiTablesSemanticParser`` is a :class:`Model` which takes as input a table and a question,
    and produces a logical form that answers the question when executed over the table.  The
    logical form is generated by a `type-constrained`, `transition-based` parser. This is an
    abstract class that defines most of the functionality related to the transition-based parser. It
    does not contain the implementation for actually training the parser. You may want to train it
    using a learning-to-search algorithm, in which case you will want to use
    ``WikiTablesErmSemanticParser``, or if you have a set of approximate logical forms that give the
    correct denotation, you will want to use ``WikiTablesMmlSemanticParser``.

    Parameters
    ----------
    vocab : ``Vocabulary``
    question_embedder : ``TextFieldEmbedder``
        Embedder for questions.
    action_embedding_dim : ``int``
        Dimension to use for action embeddings.
    encoder : ``Seq2SeqEncoder``
        The encoder to use for the input question.
    entity_encoder : ``Seq2VecEncoder``
        The encoder to used for averaging the words of an entity.
    max_decoding_steps : ``int``
        When we're decoding with a beam search, what's the maximum number of steps we should take?
        This only applies at evaluation time, not during training.
    add_action_bias : ``bool``, optional (default=True)
        If ``True``, we will learn a bias weight for each action that gets used when predicting
        that action, in addition to its embedding.
    use_neighbor_similarity_for_linking : ``bool``, optional (default=False)
        If ``True``, we will compute a max similarity between a question token and the `neighbors`
        of an entity as a component of the linking scores.  This is meant to capture the same kind
        of information as the ``related_column`` feature.
    dropout : ``float``, optional (default=0)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    num_linking_features : ``int``, optional (default=10)
        We need to construct a parameter vector for the linking features, so we need to know how
        many there are.  The default of 8 here matches the default in the ``KnowledgeGraphField``,
        which is to use all eight defined features. If this is 0, another term will be added to the
        linking score. This term contains the maximum similarity value from the entity's neighbors
        and the question.
    rule_namespace : ``str``, optional (default=rule_labels)
        The vocabulary namespace to use for production rules.  The default corresponds to the
        default used in the dataset reader, so you likely don't need to modify this.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        question_embedder: TextFieldEmbedder,
        action_embedding_dim: int,
        encoder: Seq2SeqEncoder,
        entity_encoder: Seq2VecEncoder,
        max_decoding_steps: int,
        add_action_bias: bool = True,
        use_neighbor_similarity_for_linking: bool = False,
        dropout: float = 0.0,
        num_linking_features: int = 10,
        rule_namespace: str = "rule_labels",
    ) -> None:
        super().__init__(vocab)
        self._question_embedder = question_embedder
        self._encoder = encoder
        self._entity_encoder = TimeDistributed(entity_encoder)
        self._max_decoding_steps = max_decoding_steps
        self._add_action_bias = add_action_bias
        self._use_neighbor_similarity_for_linking = use_neighbor_similarity_for_linking
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace
        self._denotation_accuracy = Average()
        self._action_sequence_accuracy = Average()
        self._has_logical_form = Average()

        self._action_padding_index = -1  # the padding value used by IndexField
        num_actions = vocab.get_vocab_size(self._rule_namespace)
        if self._add_action_bias:
            self._action_biases = Embedding(num_embeddings=num_actions, embedding_dim=1)
        self._action_embedder = Embedding(
            num_embeddings=num_actions, embedding_dim=action_embedding_dim
        )
        self._output_action_embedder = Embedding(
            num_embeddings=num_actions, embedding_dim=action_embedding_dim
        )

        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action, or a previous question attention.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        self._first_attended_question = torch.nn.Parameter(
            torch.FloatTensor(encoder.get_output_dim())
        )
        torch.nn.init.normal_(self._first_action_embedding)
        torch.nn.init.normal_(self._first_attended_question)

        check_dimensions_match(
            entity_encoder.get_output_dim(),
            question_embedder.get_output_dim(),
            "entity word average embedding dim",
            "question embedding dim",
        )

        self._num_entity_types = 5  # TODO(mattg): get this in a more principled way somehow?
        self._embedding_dim = question_embedder.get_output_dim()
        self._entity_type_encoder_embedding = Embedding(
            num_embeddings=self._num_entity_types, embedding_dim=self._embedding_dim
        )
        self._entity_type_decoder_embedding = Embedding(
            num_embeddings=self._num_entity_types, embedding_dim=action_embedding_dim
        )
        self._neighbor_params = torch.nn.Linear(self._embedding_dim, self._embedding_dim)

        if num_linking_features > 0:
            self._linking_params = torch.nn.Linear(num_linking_features, 1)
        else:
            self._linking_params = None

        if self._use_neighbor_similarity_for_linking:
            self._question_entity_params = torch.nn.Linear(1, 1)
            self._question_neighbor_params = torch.nn.Linear(1, 1)
        else:
            self._question_entity_params = None
            self._question_neighbor_params = None

    def _get_initial_rnn_and_grammar_state(
        self,
        question: Dict[str, torch.LongTensor],
        table: Dict[str, torch.LongTensor],
        world: List[WikiTablesLanguage],
        actions: List[List[ProductionRuleArray]],
        outputs: Dict[str, Any],
    ) -> Tuple[List[RnnStatelet], List[GrammarStatelet]]:
        """
        Encodes the question and table, computes a linking between the two, and constructs an
        initial RnnStatelet and GrammarStatelet for each batch instance to pass to the
        decoder.

        We take ``outputs`` as a parameter here and `modify` it, adding things that we want to
        visualize in a demo.
        """
        table_text = table["text"]
        # (batch_size, question_length, embedding_dim)
        embedded_question = self._question_embedder(question)
        question_mask = util.get_text_field_mask(question)
        # (batch_size, num_entities, num_entity_tokens, embedding_dim)
        embedded_table = self._question_embedder(table_text, num_wrapping_dims=1)
        table_mask = util.get_text_field_mask(table_text, num_wrapping_dims=1)

        batch_size, num_entities, num_entity_tokens, _ = embedded_table.size()
        num_question_tokens = embedded_question.size(1)

        # (batch_size, num_entities, embedding_dim)
        encoded_table = self._entity_encoder(embedded_table, table_mask)

        # entity_types: tensor with shape (batch_size, num_entities), where each entry is the
        # entity's type id.
        # entity_type_dict: Dict[int, int], mapping flattened_entity_index -> type_index
        # These encode the same information, but for efficiency reasons later it's nice
        # to have one version as a tensor and one that's accessible on the cpu.
        entity_types, entity_type_dict = self._get_type_vector(world, num_entities, encoded_table)

        entity_type_embeddings = self._entity_type_encoder_embedding(entity_types)

        # (batch_size, num_entities, num_neighbors) or None
        neighbor_indices = self._get_neighbor_indices(world, num_entities, encoded_table)

        if neighbor_indices is not None:
            # Neighbor_indices is padded with -1 since 0 is a potential neighbor index.
            # Thus, the absolute value needs to be taken in the index_select, and 1 needs to
            # be added for the mask since that method expects 0 for padding.
            # (batch_size, num_entities, num_neighbors, embedding_dim)
            embedded_neighbors = util.batched_index_select(
                encoded_table, torch.abs(neighbor_indices)
            )

            neighbor_mask = util.get_text_field_mask(
                {"ignored": {"ignored": neighbor_indices + 1}}, num_wrapping_dims=1
            )

            # Encoder initialized to easily obtain a masked average.
            neighbor_encoder = TimeDistributed(
                BagOfEmbeddingsEncoder(self._embedding_dim, averaged=True)
            )
            # (batch_size, num_entities, embedding_dim)
            embedded_neighbors = neighbor_encoder(embedded_neighbors, neighbor_mask)
            projected_neighbor_embeddings = self._neighbor_params(embedded_neighbors.float())

            # (batch_size, num_entities, embedding_dim)
            entity_embeddings = torch.tanh(entity_type_embeddings + projected_neighbor_embeddings)
        else:
            # (batch_size, num_entities, embedding_dim)
            entity_embeddings = torch.tanh(entity_type_embeddings)

        # Compute entity and question word similarity.  We tried using cosine distance here, but
        # because this similarity is the main mechanism that the model can use to push apart logit
        # scores for certain actions (like "n -> 1" and "n -> -1"), this needs to have a larger
        # output range than [-1, 1].
        question_entity_similarity = torch.bmm(
            embedded_table.view(batch_size, num_entities * num_entity_tokens, self._embedding_dim),
            torch.transpose(embedded_question, 1, 2),
        )

        question_entity_similarity = question_entity_similarity.view(
            batch_size, num_entities, num_entity_tokens, num_question_tokens
        )

        # (batch_size, num_entities, num_question_tokens)
        question_entity_similarity_max_score, _ = torch.max(question_entity_similarity, 2)

        # (batch_size, num_entities, num_question_tokens, num_features)
        linking_features = table["linking"]

        linking_scores = question_entity_similarity_max_score

        if self._use_neighbor_similarity_for_linking:
            # The linking score is computed as a linear projection of two terms. The first is the
            # maximum similarity score over the entity's words and the question token. The second
            # is the maximum similarity over the words in the entity's neighbors and the question
            # token.
            #
            # The second term, projected_question_neighbor_similarity, is useful when a column
            # needs to be selected. For example, the question token might have no similarity with
            # the column name, but is similar with the cells in the column.
            #
            # Note that projected_question_neighbor_similarity is intended to capture the same
            # information as the related_column feature.
            #
            # Also note that this block needs to be _before_ the `linking_params` block, because
            # we're overwriting `linking_scores`, not adding to it.

            # (batch_size, num_entities, num_neighbors, num_question_tokens)
            question_neighbor_similarity = util.batched_index_select(
                question_entity_similarity_max_score, torch.abs(neighbor_indices)
            )
            # (batch_size, num_entities, num_question_tokens)
            question_neighbor_similarity_max_score, _ = torch.max(question_neighbor_similarity, 2)
            projected_question_entity_similarity = self._question_entity_params(
                question_entity_similarity_max_score.unsqueeze(-1)
            ).squeeze(-1)
            projected_question_neighbor_similarity = self._question_neighbor_params(
                question_neighbor_similarity_max_score.unsqueeze(-1)
            ).squeeze(-1)
            linking_scores = (
                projected_question_entity_similarity + projected_question_neighbor_similarity
            )

        feature_scores = None
        if self._linking_params is not None:
            feature_scores = self._linking_params(linking_features).squeeze(3)
            linking_scores = linking_scores + feature_scores

        # (batch_size, num_question_tokens, num_entities)
        linking_probabilities = self._get_linking_probabilities(
            world, linking_scores.transpose(1, 2), question_mask, entity_type_dict
        )

        # (batch_size, num_question_tokens, embedding_dim)
        link_embedding = util.weighted_sum(entity_embeddings, linking_probabilities)
        encoder_input = torch.cat([link_embedding, embedded_question], 2)

        # (batch_size, question_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._encoder(encoder_input, question_mask))

        # This will be our initial hidden state and memory cell for the decoder LSTM.
        final_encoder_output = util.get_final_encoder_states(
            encoder_outputs, question_mask, self._encoder.is_bidirectional()
        )
        memory_cell = encoder_outputs.new_zeros(batch_size, self._encoder.get_output_dim())

        # To make grouping states together in the decoder easier, we convert the batch dimension in
        # all of our tensors into an outer list.  For instance, the encoder outputs have shape
        # `(batch_size, question_length, encoder_output_dim)`.  We need to convert this into a list
        # of `batch_size` tensors, each of shape `(question_length, encoder_output_dim)`.  Then we
        # won't have to do any index selects, or anything, we'll just do some `torch.cat()`s.
        encoder_output_list = [encoder_outputs[i] for i in range(batch_size)]
        question_mask_list = [question_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(
                RnnStatelet(
                    final_encoder_output[i],
                    memory_cell[i],
                    self._first_action_embedding,
                    self._first_attended_question,
                    encoder_output_list,
                    question_mask_list,
                )
            )
        initial_grammar_state = [
            self._create_grammar_state(world[i], actions[i], linking_scores[i], entity_types[i])
            for i in range(batch_size)
        ]
        if not self.training:
            # We add a few things to the outputs that will be returned from `forward` at evaluation
            # time, for visualization in a demo.
            outputs["linking_scores"] = linking_scores
            if feature_scores is not None:
                outputs["feature_scores"] = feature_scores
            outputs["similarity_scores"] = question_entity_similarity_max_score
        return initial_rnn_state, initial_grammar_state

    @staticmethod
    def _get_neighbor_indices(
        worlds: List[WikiTablesLanguage], num_entities: int, tensor: torch.Tensor
    ) -> torch.LongTensor:
        """
        This method returns the indices of each entity's neighbors. A tensor
        is accepted as a parameter for copying purposes.

        Parameters
        ----------
        worlds : ``List[WikiTablesLanguage]``
        num_entities : ``int``
        tensor : ``torch.Tensor``
            Used for copying the constructed list onto the right device.

        Returns
        -------
        A ``torch.LongTensor`` with shape ``(batch_size, num_entities, num_neighbors)``. It is padded
        with -1 instead of 0, since 0 is a valid neighbor index. If all the entities in the batch
        have no neighbors, None will be returned.
        """

        num_neighbors = 0
        for world in worlds:
            for entity in world.table_graph.entities:
                if len(world.table_graph.neighbors[entity]) > num_neighbors:
                    num_neighbors = len(world.table_graph.neighbors[entity])

        batch_neighbors = []
        no_entities_have_neighbors = True
        for world in worlds:
            # Each batch instance has its own world, which has a corresponding table.
            entities = world.table_graph.entities
            entity2index = {entity: i for i, entity in enumerate(entities)}
            entity2neighbors = world.table_graph.neighbors
            neighbor_indexes = []
            for entity in entities:
                entity_neighbors = [entity2index[n] for n in entity2neighbors[entity]]
                if entity_neighbors:
                    no_entities_have_neighbors = False
                # Pad with -1 instead of 0, since 0 represents a neighbor index.
                padded = pad_sequence_to_length(entity_neighbors, num_neighbors, lambda: -1)
                neighbor_indexes.append(padded)
            neighbor_indexes = pad_sequence_to_length(
                neighbor_indexes, num_entities, lambda: [-1] * num_neighbors
            )
            batch_neighbors.append(neighbor_indexes)
        # It is possible that none of the entities has any neighbors, since our definition of the
        # knowledge graph allows it when no entities or numbers were extracted from the question.
        if no_entities_have_neighbors:
            return None
        return tensor.new_tensor(batch_neighbors, dtype=torch.long)

    @staticmethod
    def _get_type_vector(
        worlds: List[WikiTablesLanguage], num_entities: int, tensor: torch.Tensor
    ) -> Tuple[torch.LongTensor, Dict[int, int]]:
        """
        Produces a tensor with shape ``(batch_size, num_entities)`` that encodes each entity's
        type. In addition, a map from a flattened entity index to type is returned to combine
        entity type operations into one method.

        Parameters
        ----------
        worlds : ``List[WikiTablesLanguage]``
        num_entities : ``int``
        tensor : ``torch.Tensor``
            Used for copying the constructed list onto the right device.

        Returns
        -------
        A ``torch.LongTensor`` with shape ``(batch_size, num_entities)``.
        entity_types : ``Dict[int, int]``
            This is a mapping from ((batch_index * num_entities) + entity_index) to entity type id.
        """
        entity_types = {}
        batch_types = []
        for batch_index, world in enumerate(worlds):
            types = []
            for entity_index, entity in enumerate(world.table_graph.entities):
                # We need numbers to be first, then date columns, then number columns, strings, and
                # string columns, in that order, because our entities are going to be sorted.  We do
                # a split by type and then a merge later, and it relies on this sorting.
                if entity.startswith("date_column:"):
                    entity_type = 1
                elif entity.startswith("number_column:"):
                    entity_type = 2
                elif entity.startswith("string:"):
                    entity_type = 3
                elif entity.startswith("string_column:"):
                    entity_type = 4
                else:
                    entity_type = 0
                types.append(entity_type)

                # For easier lookups later, we're actually using a _flattened_ version
                # of (batch_index, entity_index) for the key, because this is how the
                # linking scores are stored.
                flattened_entity_index = batch_index * num_entities + entity_index
                entity_types[flattened_entity_index] = entity_type
            padded = pad_sequence_to_length(types, num_entities, lambda: 0)
            batch_types.append(padded)
        return tensor.new_tensor(batch_types, dtype=torch.long), entity_types

    def _get_linking_probabilities(
        self,
        worlds: List[WikiTablesLanguage],
        linking_scores: torch.FloatTensor,
        question_mask: torch.BoolTensor,
        entity_type_dict: Dict[int, int],
    ) -> torch.FloatTensor:
        """
        Produces the probability of an entity given a question word and type. The logic below
        separates the entities by type since the softmax normalization term sums over entities
        of a single type.

        Parameters
        ----------
        worlds : ``List[WikiTablesLanguage]``
        linking_scores : ``torch.FloatTensor``
            Has shape (batch_size, num_question_tokens, num_entities).
        question_mask: ``torch.BoolTensor``
            Has shape (batch_size, num_question_tokens).
        entity_type_dict : ``Dict[int, int]``
            This is a mapping from ((batch_index * num_entities) + entity_index) to entity type id.

        Returns
        -------
        batch_probabilities : ``torch.FloatTensor``
            Has shape ``(batch_size, num_question_tokens, num_entities)``.
            Contains all the probabilities for an entity given a question word.
        """
        _, num_question_tokens, num_entities = linking_scores.size()
        batch_probabilities = []

        for batch_index, world in enumerate(worlds):
            all_probabilities = []
            num_entities_in_instance = 0

            # NOTE: The way that we're doing this here relies on the fact that entities are
            # implicitly sorted by their types when we sort them by name, and that numbers come
            # before "date_column:", followed by "number_column:", "string:", and "string_column:".
            # This is not a great assumption, and could easily break later, but it should work for now.
            for type_index in range(self._num_entity_types):
                # This index of 0 is for the null entity for each type, representing the case where a
                # word doesn't link to any entity.
                entity_indices = [0]
                entities = world.table_graph.entities
                for entity_index, _ in enumerate(entities):
                    if entity_type_dict[batch_index * num_entities + entity_index] == type_index:
                        entity_indices.append(entity_index)

                if len(entity_indices) == 1:
                    # No entities of this type; move along...
                    continue

                # We're subtracting one here because of the null entity we added above.
                num_entities_in_instance += len(entity_indices) - 1

                # We separate the scores by type, since normalization is done per type.  There's an
                # extra "null" entity per type, also, so we have `num_entities_per_type + 1`.  We're
                # selecting from a (num_question_tokens, num_entities) linking tensor on _dimension 1_,
                # so we get back something of shape (num_question_tokens,) for each index we're
                # selecting.  All of the selected indices together then make a tensor of shape
                # (num_question_tokens, num_entities_per_type + 1).
                indices = linking_scores.new_tensor(entity_indices, dtype=torch.long)
                entity_scores = linking_scores[batch_index].index_select(1, indices)

                # We used index 0 for the null entity, so this will actually have some values in it.
                # But we want the null entity's score to be 0, so we set that here.
                entity_scores[:, 0] = 0

                # No need for a mask here, as this is done per batch instance, with no padding.
                type_probabilities = torch.nn.functional.softmax(entity_scores, dim=1)
                all_probabilities.append(type_probabilities[:, 1:])

            # We need to add padding here if we don't have the right number of entities.
            if num_entities_in_instance != num_entities:
                zeros = linking_scores.new_zeros(
                    num_question_tokens, num_entities - num_entities_in_instance
                )
                all_probabilities.append(zeros)

            # (num_question_tokens, num_entities)
            probabilities = torch.cat(all_probabilities, dim=1)
            batch_probabilities.append(probabilities)
        batch_probabilities = torch.stack(batch_probabilities, dim=0)
        return batch_probabilities * question_mask.unsqueeze(-1)

    @staticmethod
    def _action_history_match(predicted: List[int], targets: torch.LongTensor) -> int:
        # TODO(mattg): this could probably be moved into a FullSequenceMatch metric, or something.
        # Check if target is big enough to cover prediction (including start/end symbols)
        if len(predicted) > targets.size(1):
            return 0
        predicted_tensor = targets.new_tensor(predicted)
        targets_trimmed = targets[:, : len(predicted)]
        # Return 1 if the predicted sequence is anywhere in the list of targets.
        return torch.max(torch.min(targets_trimmed.eq(predicted_tensor), dim=1)[0]).item()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        We track three metrics here:

            1. lf_retrieval_acc, which is the percentage of the time that our best output action
            sequence is in the set of action sequences provided by offline search.  This is an
            easy-to-compute lower bound on denotation accuracy for the set of examples where we
            actually have offline output.  We only score lf_retrieval_acc on that subset.

            2. denotation_acc, which is the percentage of examples where we get the correct
            denotation.  This is the typical "accuracy" metric, and it is what you should usually
            report in an experimental result.  You need to be careful, though, that you're
            computing this on the full data, and not just the subset that has DPD output (make sure
            you pass "keep_if_no_dpd=True" to the dataset reader, which we do for validation data,
            but not training data).

            3. lf_percent, which is the percentage of time that decoding actually produces a
            finished logical form.  We might not produce a valid logical form if the decoder gets
            into a repetitive loop, or we're trying to produce a super long logical form and run
            out of time steps, or something.
        """
        return {
            "lf_retrieval_acc": self._action_sequence_accuracy.get_metric(reset),
            "denotation_acc": self._denotation_accuracy.get_metric(reset),
            "lf_percent": self._has_logical_form.get_metric(reset),
        }

    def _create_grammar_state(
        self,
        world: WikiTablesLanguage,
        possible_actions: List[ProductionRuleArray],
        linking_scores: torch.Tensor,
        entity_types: torch.Tensor,
    ) -> GrammarStatelet:
        """
        This method creates the GrammarStatelet object that's used for decoding.  Part of
        creating that is creating the `valid_actions` dictionary, which contains embedded
        representations of all of the valid actions.  So, we create that here as well.

        The way we represent the valid expansions is a little complicated: we use a
        dictionary of `action types`, where the key is the action type (like "global", "linked", or
        whatever your model is expecting), and the value is a tuple representing all actions of
        that type.  The tuple is (input tensor, output tensor, action id).  The input tensor has
        the representation that is used when `selecting` actions, for all actions of this type.
        The output tensor has the representation that is used when feeding the action to the next
        step of the decoder (this could just be the same as the input tensor).  The action ids are
        a list of indices into the main action list for each batch instance.

        The inputs to this method are for a `single instance in the batch`; none of the tensors we
        create here are batched.  We grab the global action ids from the input
        ``ProductionRuleArrays``, and we use those to embed the valid actions for every
        non-terminal type.  We use the input ``linking_scores`` for non-global actions.

        Parameters
        ----------
        world : ``WikiTablesLanguage``
            From the input to ``forward`` for a single batch instance.
        possible_actions : ``List[ProductionRuleArray]``
            From the input to ``forward`` for a single batch instance.
        linking_scores : ``torch.Tensor``
            Assumed to have shape ``(num_entities, num_question_tokens)`` (i.e., there is no batch
            dimension).
        entity_types : ``torch.Tensor``
            Assumed to have shape ``(num_entities,)`` (i.e., there is no batch dimension).
        """
        # TODO(mattg): Move the "valid_actions" construction to another method.
        action_map = {}
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            action_map[action_string] = action_index
        entity_map = {}
        for entity_index, entity in enumerate(world.table_graph.entities):
            entity_map[entity] = entity_index

        valid_actions = world.get_nonterminal_productions()
        translated_valid_actions: Dict[
            str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]
        ] = {}
        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.  We'll first split those productions by global vs.
            # linked action.
            action_indices = [action_map[action_string] for action_string in action_strings]
            production_rule_arrays = [(possible_actions[index], index) for index in action_indices]
            global_actions = []
            linked_actions = []
            for production_rule_array, action_index in production_rule_arrays:
                if production_rule_array[1]:
                    global_actions.append((production_rule_array[2], action_index))
                else:
                    linked_actions.append((production_rule_array[0], action_index))

            # Then we get the embedded representations of the global actions if any.
            if global_actions:
                global_action_tensors, global_action_ids = zip(*global_actions)
                global_action_tensor = torch.cat(global_action_tensors, dim=0)
                global_input_embeddings = self._action_embedder(global_action_tensor)
                if self._add_action_bias:
                    global_action_biases = self._action_biases(global_action_tensor)
                    global_input_embeddings = torch.cat(
                        [global_input_embeddings, global_action_biases], dim=-1
                    )
                global_output_embeddings = self._output_action_embedder(global_action_tensor)
                translated_valid_actions[key]["global"] = (
                    global_input_embeddings,
                    global_output_embeddings,
                    list(global_action_ids),
                )

            # Then the representations of the linked actions.
            if linked_actions:
                linked_rules, linked_action_ids = zip(*linked_actions)
                entities = [rule.split(" -> ")[1] for rule in linked_rules]
                entity_ids = [entity_map[entity] for entity in entities]
                # (num_linked_actions, num_question_tokens)
                entity_linking_scores = linking_scores[entity_ids]
                # (num_linked_actions,)
                entity_type_tensor = entity_types[entity_ids]
                # (num_linked_actions, entity_type_embedding_dim)
                entity_type_embeddings = self._entity_type_decoder_embedding(entity_type_tensor)
                translated_valid_actions[key]["linked"] = (
                    entity_linking_scores,
                    entity_type_embeddings,
                    list(linked_action_ids),
                )
        return GrammarStatelet([START_SYMBOL], translated_valid_actions, world.is_nonterminal)

    def _compute_validation_outputs(
        self,
        actions: List[List[ProductionRuleArray]],
        best_final_states: Mapping[int, Sequence[GrammarBasedState]],
        world: List[WikiTablesLanguage],
        target_list: List[List[str]],
        metadata: List[Dict[str, Any]],
        outputs: Dict[str, Any],
    ) -> None:
        """
        Does common things for validation time: computing logical form accuracy (which is expensive
        and unnecessary during training), adding visualization info to the output dictionary, etc.

        This doesn't return anything; instead it `modifies` the given ``outputs`` dictionary, and
        calls metrics on ``self``.
        """
        batch_size = len(actions)
        action_mapping = {}
        for batch_index, batch_actions in enumerate(actions):
            for action_index, action in enumerate(batch_actions):
                action_mapping[(batch_index, action_index)] = action[0]
        outputs["action_mapping"] = action_mapping
        outputs["best_action_sequence"] = []
        outputs["debug_info"] = []
        outputs["entities"] = []
        outputs["logical_form"] = []
        outputs["answer"] = []
        for i in range(batch_size):
            # Decoding may not have terminated with any completed logical forms, if `num_steps`
            # isn't long enough (or if the model is not trained enough and gets into an
            # infinite action loop).
            outputs["logical_form"].append([])
            if i in best_final_states:
                all_action_indices = [
                    best_final_states[i][j].action_history[0]
                    for j in range(len(best_final_states[i]))
                ]
                found_denotation = False
                for action_indices in all_action_indices:
                    action_strings = [
                        action_mapping[(i, action_index)] for action_index in action_indices
                    ]
                    has_logical_form = False
                    try:
                        logical_form = world[i].action_sequence_to_logical_form(action_strings)
                        has_logical_form = True
                    except ParsingError:
                        logical_form = "Error producing logical form"
                    if target_list is not None:
                        denotation_correct = world[i].evaluate_logical_form(
                            logical_form, target_list[i]
                        )
                    else:
                        denotation_correct = False
                    if not found_denotation:
                        try:
                            denotation = world[i].execute(logical_form)
                            if denotation:
                                outputs["answer"].append(denotation)
                                found_denotation = True
                        except ExecutionError:
                            pass
                        if found_denotation:
                            if has_logical_form:
                                self._has_logical_form(1.0)
                            else:
                                self._has_logical_form(0.0)
                            if target_list:
                                self._denotation_accuracy(1.0 if denotation_correct else 0.0)
                            outputs["best_action_sequence"].append(action_strings)
                    outputs["logical_form"][-1].append(logical_form)
                if not found_denotation:
                    outputs["answer"].append(None)
                    self._denotation_accuracy(0.0)
                outputs["debug_info"].append(best_final_states[i][0].debug_info[0])  # type: ignore
                outputs["entities"].append(world[i].table_graph.entities)
            else:
                self._has_logical_form(0.0)
                self._denotation_accuracy(0.0)

        if metadata is not None:
            outputs["question_tokens"] = [x["question_tokens"] for x in metadata]

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions.  This is (confusingly) a separate notion from the "decoder"
        in "encoder/decoder", where that decoder logic lives in the ``TransitionFunction``.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        action_mapping = output_dict["action_mapping"]
        best_actions = output_dict["best_action_sequence"]
        debug_infos = output_dict["debug_info"]
        batch_action_info = []
        for batch_index, (predicted_actions, debug_info) in enumerate(
            zip(best_actions, debug_infos)
        ):
            instance_action_info = []
            for predicted_action, action_debug_info in zip(predicted_actions, debug_info):
                action_info = {}
                action_info["predicted_action"] = predicted_action
                considered_actions = action_debug_info["considered_actions"]
                probabilities = action_debug_info["probabilities"]
                actions = []
                for action, probability in zip(considered_actions, probabilities):
                    if action != -1:
                        actions.append((action_mapping[(batch_index, action)], probability))
                actions.sort()
                considered_actions, probabilities = zip(*actions)
                action_info["considered_actions"] = considered_actions
                action_info["action_probabilities"] = probabilities
                action_info["question_attention"] = action_debug_info.get("question_attention", [])
                instance_action_info.append(action_info)
            batch_action_info.append(instance_action_info)
        output_dict["predicted_actions"] = batch_action_info
        return output_dict
