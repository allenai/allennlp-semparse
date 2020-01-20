"""
``KnowledgeGraphField`` is a ``Field`` which stores a knowledge graph representation.
"""
from typing import Callable, Dict, List, Set

import editdistance
from overrides import overrides
import torch

from allennlp.common import util
from allennlp.common.checks import ConfigurationError
from allennlp.data.fields import Field, ListField, TextField
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from allennlp.data.vocabulary import Vocabulary

from allennlp_semparse.common.knowledge_graph import KnowledgeGraph


class KnowledgeGraphField(Field[Dict[str, torch.Tensor]]):
    """
    A ``KnowledgeGraphField`` represents a ``KnowledgeGraph`` as a ``Field`` that can be used in a
    ``Model``.  For each entity in the graph, we output two things: a text representation of the
    entity, handled identically to a ``TextField``, and a list of linking features for each token
    in some input utterance.

    The output of this field is a dictionary::

        {
          "text": Dict[str, torch.Tensor],  # each tensor has shape (batch_size, num_entities, num_entity_tokens)
          "linking": torch.Tensor  # shape (batch_size, num_entities, num_utterance_tokens, num_features)
        }

    The ``text`` component of this dictionary is suitable to be passed into a
    ``TextFieldEmbedder`` (which handles the additional ``num_entities`` dimension without any
    issues).  The ``linking`` component of the dictionary can be used however you want to decide
    which tokens in the utterance correspond to which entities in the knowledge graph.

    In order to create the ``text`` component, we use the same dictionary of ``TokenIndexers``
    that's used in a ``TextField`` (as we're just representing the text corresponding to each
    entity).  For the ``linking`` component, we use a set of hard-coded feature extractors that
    operate between the text corresponding to each entity and each token in the utterance.

    Parameters
    ----------
    knowledge_graph : ``KnowledgeGraph``
        The knowledge graph that this field stores.
    utterance_tokens : ``List[Token]``
        The tokens in some utterance that is paired with the ``KnowledgeGraph``.  We compute a set
        of features for linking tokens in the utterance to entities in the graph.
    tokenizer : ``Tokenizer``, optional (default=``SpacyTokenizer()``)
        We'll use this ``Tokenizer`` to tokenize the text representation of each entity.
    token_indexers : ``Dict[str, TokenIndexer]``
        Token indexers that convert entities into arrays, similar to how text tokens are treated in
        a ``TextField``.  These might operate on the name of the entity itself, its type, its
        neighbors in the graph, etc.
    feature_extractors : ``List[str]``, optional
        Names of feature extractors to use for computing linking features.  These must be
        attributes of this object, without the first underscore.  The feature extraction functions
        are listed as the last methods in this class.  For example, to use
        :func:`_exact_token_match`, you would pass the string ``exact_token_match``.  We will add
        an underscore and look for a function matching that name.  If this list is omitted, we will
        use all available feature functions.
    entity_tokens : ``List[List[Token]]``, optional
        If you have pre-computed the tokenization of the table text, you can pass it in here.  The
        must be a list of the tokens in the entity text, for each entity in the knowledge graph, in
        the same order in which the knowledge graph returns entities.
    linking_features : ``List[List[List[float]]]``, optional
        If you have pre-computed the linking features between the utterance and the table text, you
        can pass it in here.
    include_in_vocab : ``bool``, optional (default=True)
        If this is ``False``, we will skip the ``count_vocab_items`` logic, leaving out all table
        entity text from the vocabulary computation.  You might want to do this if you have a lot
        of rare entities in your tables, and you see the same table in multiple training instances,
        so your vocabulary counts get skewed and include too many rare entities.
    max_table_tokens : ``int``, optional
        If given, we will only keep this number of total table tokens.  This bounds the memory
        usage of the table representations, truncating cells with really long text.  We specify a
        total number of tokens, not a max cell text length, because the number of table entities
        varies.
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        utterance_tokens: List[Token],
        token_indexers: Dict[str, TokenIndexer],
        tokenizer: Tokenizer = None,
        feature_extractors: List[str] = None,
        entity_tokens: List[List[Token]] = None,
        linking_features: List[List[List[float]]] = None,
        include_in_vocab: bool = True,
        max_table_tokens: int = None,
    ) -> None:

        self.knowledge_graph = knowledge_graph
        self._tokenizer = tokenizer or SpacyTokenizer(pos_tags=True)
        self._token_indexers = token_indexers
        if not entity_tokens:
            entity_texts = [
                knowledge_graph.entity_text[entity].lower() for entity in knowledge_graph.entities
            ]
            # TODO(mattg): Because we do tagging on each of these entities in addition to just
            # tokenizations, this is quite slow, and about half of our data processing time just
            # goes to this (~15 minutes when there are 7k instances).  The reason we do tagging is
            # so that we can add lemma features.  If we can remove the need for lemma / other
            # hand-written features, like with a CNN, we can cut down our data processing time by a
            # factor of 2.
            self.entity_texts = self._tokenizer.batch_tokenize(entity_texts)
        else:
            self.entity_texts = entity_tokens
        entity_text_fields = []
        max_entity_tokens = None
        if max_table_tokens:
            num_entities = len(self.entity_texts)
            num_entity_tokens = max(len(entity_text) for entity_text in self.entity_texts)
            # This truncates the number of entity tokens used, enabling larger tables (either in
            # the number of entities in the table, or the number of tokens per entity) to fit in
            # memory, particularly when using ELMo.
            if num_entities * num_entity_tokens > max_table_tokens:
                max_entity_tokens = int(max_table_tokens / num_entities)
        for entity_text in self.entity_texts:
            if max_entity_tokens:
                entity_text = entity_text[:max_entity_tokens]
            entity_text_fields.append(TextField(entity_text, token_indexers))
        if self.entity_texts:
            self._entity_text_field = ListField(entity_text_fields)
        else:
            empty_text_field = TextField([], self._token_indexers).empty_field()
            self._entity_text_field = ListField([empty_text_field]).empty_field()
        self.utterance_tokens = utterance_tokens
        self._include_in_vocab = include_in_vocab

        feature_extractors = (
            feature_extractors
            if feature_extractors is not None
            else [
                "number_token_match",
                "exact_token_match",
                "contains_exact_token_match",
                "lemma_match",
                "contains_lemma_match",
                "edit_distance",
                "related_column",
                "related_column_lemma",
                "span_overlap_fraction",
                "span_lemma_overlap_fraction",
            ]
        )
        self._feature_extractors: List[
            Callable[[str, List[Token], Token, int, List[Token]], float]
        ] = []
        for feature_extractor_name in feature_extractors:
            extractor = getattr(self, "_" + feature_extractor_name, None)
            if not extractor:
                raise ConfigurationError(
                    f"Invalid feature extractor name: {feature_extractor_name}"
                )
            self._feature_extractors.append(extractor)

        if not linking_features:
            # For quicker lookups in our feature functions, we'll additionally store some
            # dictionaries that map entity strings to useful information about the entity.
            self._entity_text_map: Dict[str, List[Token]] = {}
            for entity, entity_text in zip(knowledge_graph.entities, self.entity_texts):
                self._entity_text_map[entity] = entity_text

            self._entity_text_exact_text: Dict[str, Set[str]] = {}
            for entity, entity_text in zip(knowledge_graph.entities, self.entity_texts):
                self._entity_text_exact_text[entity] = set(e.text for e in entity_text)

            self._entity_text_lemmas: Dict[str, Set[str]] = {}
            for entity, entity_text in zip(knowledge_graph.entities, self.entity_texts):
                self._entity_text_lemmas[entity] = set(e.lemma_ for e in entity_text)
            self.linking_features = self._compute_linking_features()
        else:
            self.linking_features = linking_features

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._include_in_vocab:
            self._entity_text_field.count_vocab_items(counter)

    @overrides
    def index(self, vocab: Vocabulary):
        self._entity_text_field.index(vocab)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        padding_lengths = {
            "num_entities": len(self.entity_texts),
            "num_utterance_tokens": len(self.utterance_tokens),
        }
        padding_lengths.update(self._entity_text_field.get_padding_lengths())
        return padding_lengths

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        text_tensors = self._entity_text_field.as_tensor(padding_lengths)
        padded_linking_features = util.pad_sequence_to_length(
            self.linking_features, padding_lengths["num_entities"], default_value=lambda: []
        )
        padded_linking_arrays = []

        def default_feature_value():
            return [0.0] * len(self._feature_extractors)

        for linking_features in padded_linking_features:
            padded_features = util.pad_sequence_to_length(
                linking_features,
                padding_lengths["num_utterance_tokens"],
                default_value=default_feature_value,
            )
            padded_linking_arrays.append(padded_features)
        linking_features_tensor = torch.FloatTensor(padded_linking_arrays)
        return {"text": text_tensors, "linking": linking_features_tensor}

    def _compute_linking_features(self) -> List[List[List[float]]]:
        linking_features = []
        for entity, entity_text in zip(self.knowledge_graph.entities, self.entity_texts):
            entity_features = []
            for token_index, token in enumerate(self.utterance_tokens):
                token_features = []
                for feature_extractor in self._feature_extractors:
                    token_features.append(
                        feature_extractor(
                            entity, entity_text, token, token_index, self.utterance_tokens
                        )
                    )
                entity_features.append(token_features)
            linking_features.append(entity_features)
        return linking_features

    @overrides
    def empty_field(self) -> "KnowledgeGraphField":
        return KnowledgeGraphField(KnowledgeGraph(set(), {}), [], self._token_indexers)

    @overrides
    def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        text_tensors = [tensor["text"] for tensor in tensor_list]
        batched_text = self._entity_text_field.batch_tensors(text_tensors)
        batched_linking = torch.stack([tensor["linking"] for tensor in tensor_list])
        return {"text": batched_text, "linking": batched_linking}

    # Below here we have feature extractor functions.  To keep a consistent API for easy logic
    # above, some of these functions have unused arguments.

    # These feature extractors are generally pretty specific to the logical form language and
    # problem setting in WikiTableQuestions.  This whole notion of feature extraction should
    # eventually be made more general (or just removed, if we can replace it with CNN features...).
    # For the feature functions used in the original parser written in PNP, see here:
    # https://github.com/allenai/pnp/blob/wikitables2/src/main/scala/org/allenai/wikitables/SemanticParserFeatureGenerator.scala

    # One notable difference between how the features work here and how they worked in PNP is that
    # we're using the table text when computing string matches, while PNP used the _entity name_.
    # It turns out that the entity name is derived from the table text, so this should be roughly
    # equivalent, except in the case of some numbers.  If there are cells with different text that
    # normalize to the same name, you could get `_2` or similar appended to the name, so the way we
    # do it here should just be better.  But it's a possible minor source of variation from the
    # original parser.

    # Another difference between these features and the PNP features is that the span overlap used
    # a weighting scheme to downweight matches on frequent words (like "the"), and the lemma
    # overlap feature value was calculated a little differently.  I'm guessing that doesn't make a
    # huge difference...

    def _number_token_match(
        self,
        entity: str,
        entity_text: List[Token],
        token: Token,
        token_index: int,
        tokens: List[Token],
    ) -> float:
        # PNP had a "spanFeatures" function that said whether an entity was a-priori known to link
        # to a token or set of tokens in the question.  This was only used for numbers, and it's
        # not totally clear to me how this number feature overlapped with the token match features
        # in the original implementation (I think in most cases it was the same, except for things
        # like "four million", because the token match is derived from the entity name, which would
        # be 4000000, and wouldn't match "four million").
        #
        # Our implementation basically just adds a duplicate token match feature that's specific to
        # numbers.  It'll break in some rare cases (e.g., "Which four had four million ..."), but
        # those shouldn't be a big deal.
        if ":" in entity:
            # This check works because numbers are the only entities that don't contain ":". All
            # others in both WikiTables languages do (e.g.: fb:row.row.column_name,
            # date_column:year, string:usl_a_league etc.).
            return 0.0
        return self._contains_exact_token_match(entity, entity_text, token, token_index, tokens)

    def _exact_token_match(
        self,
        entity: str,
        entity_text: List[Token],
        token: Token,
        token_index: int,
        tokens: List[Token],
    ) -> float:
        if len(entity_text) != 1:
            return 0.0
        return self._contains_exact_token_match(entity, entity_text, token, token_index, tokens)

    def _contains_exact_token_match(
        self,
        entity: str,
        entity_text: List[Token],
        token: Token,
        token_index: int,
        tokens: List[Token],
    ) -> float:
        if token.text in self._entity_text_exact_text[entity]:
            return 1.0
        return 0.0

    def _lemma_match(
        self,
        entity: str,
        entity_text: List[Token],
        token: Token,
        token_index: int,
        tokens: List[Token],
    ) -> float:
        if len(entity_text) != 1:
            return 0.0
        return self._contains_lemma_match(entity, entity_text, token, token_index, tokens)

    def _contains_lemma_match(
        self,
        entity: str,
        entity_text: List[Token],
        token: Token,
        token_index: int,
        tokens: List[Token],
    ) -> float:
        if token.text in self._entity_text_exact_text[entity]:
            return 1.0
        if token.lemma_ in self._entity_text_lemmas[entity]:
            return 1.0
        return 0.0

    def _edit_distance(
        self,
        entity: str,
        entity_text: List[Token],
        token: Token,
        token_index: int,
        tokens: List[Token],
    ) -> float:
        edit_distance = float(editdistance.eval(" ".join(e.text for e in entity_text), token.text))
        return 1.0 - edit_distance / len(token.text)

    def _related_column(
        self,
        entity: str,
        entity_text: List[Token],
        token: Token,
        token_index: int,
        tokens: List[Token],
    ) -> float:
        # Check if the entity is a column name in one of the two WikiTables languages.
        if not entity.startswith("fb:row.row") and "_column:" not in entity:
            return 0.0
        for neighbor in self.knowledge_graph.neighbors[entity]:
            if token.text in self._entity_text_exact_text[neighbor]:
                return 1.0
        return 0.0

    def _related_column_lemma(
        self,
        entity: str,
        entity_text: List[Token],
        token: Token,
        token_index: int,
        tokens: List[Token],
    ) -> float:
        # Check if the entity is a column name in one of the two WikiTables languages.
        if not entity.startswith("fb:row.row") and "_column:" not in entity:
            return 0.0
        for neighbor in self.knowledge_graph.neighbors[entity]:
            if token.text in self._entity_text_exact_text[neighbor]:
                return 1.0
            if token.lemma_ in self._entity_text_lemmas[neighbor]:
                return 1.0
        return 0.0

    def _span_overlap_fraction(
        self,
        entity: str,
        entity_text: List[Token],
        token: Token,
        token_index: int,
        tokens: List[Token],
    ) -> float:
        entity_words = set(entity_token.text for entity_token in entity_text)
        if not entity_words:
            # Some tables have empty cells.
            return 0
        seen_entity_words = set()
        token_index_left = token_index
        while token_index < len(tokens) and tokens[token_index].text in entity_words:
            seen_entity_words.add(tokens[token_index].text)
            token_index += 1
        while token_index_left >= 0 and tokens[token_index_left].text in entity_words:
            seen_entity_words.add(tokens[token_index_left].text)
            token_index_left -= 1
        return len(seen_entity_words) / len(entity_words)

    def _span_lemma_overlap_fraction(
        self,
        entity: str,
        entity_text: List[Token],
        token: Token,
        token_index: int,
        tokens: List[Token],
    ) -> float:
        entity_lemmas = set(entity_token.lemma_ for entity_token in entity_text)
        if not entity_lemmas:
            # Some tables have empty cells.
            return 0
        seen_entity_lemmas = set()
        token_index_left = token_index
        while token_index < len(tokens) and tokens[token_index].lemma_ in entity_lemmas:
            seen_entity_lemmas.add(tokens[token_index].lemma_)
            token_index += 1
        while token_index_left >= 0 and tokens[token_index_left].lemma_ in entity_lemmas:
            seen_entity_lemmas.add(tokens[token_index_left].lemma_)
            token_index_left -= 1
        return len(seen_entity_lemmas) / len(entity_lemmas)
