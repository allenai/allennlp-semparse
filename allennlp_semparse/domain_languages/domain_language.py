from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union
import inspect
import logging
import sys
import traceback
import types

from nltk import Tree

from allennlp.common.util import START_SYMBOL

from allennlp_semparse.common import util, ParsingError, ExecutionError

logger = logging.getLogger(__name__)


# We rely heavily on the typing module and its type annotations for our grammar induction code.
# Unfortunately, the behavior of the typing module changed somewhat substantially between python
# 3.6 and 3.7, so we need to do some gymnastics to get some of our checks to work with both.
# That's what these three methods are about.


def is_callable(type_: Type) -> bool:
    if sys.version_info < (3, 7):
        from typing import CallableMeta  # type: ignore

        return isinstance(type_, CallableMeta)  # type: ignore
    else:
        return getattr(type_, "_name", None) == "Callable"


def is_generic(type_: Type) -> bool:
    if sys.version_info < (3, 7):
        from typing import GenericMeta  # type: ignore

        return isinstance(type_, GenericMeta)  # type: ignore
    else:
        from typing import _GenericAlias  # type: ignore

        return isinstance(type_, _GenericAlias)  # type: ignore


def get_generic_name(type_: Type) -> str:
    if sys.version_info < (3, 7):
        origin = type_.__origin__.__name__
    else:
        # In python 3.7, type_.__origin__ switched to the built-in class, instead of the typing
        # class.
        origin = type_._name
    args = type_.__args__
    return f'{origin}[{",".join(arg.__name__ for arg in args)}]'


def infer_collection_type(collection: Any) -> Type:
    instance_types = set([type(instance) for instance in collection])
    if len(instance_types) != 1:
        raise ValueError(f"Inconsistent types in collection: {instance_types}, {collection}")
    subtype = list(instance_types)[0]
    if isinstance(collection, list):
        return List[subtype]  # type: ignore
    elif isinstance(collection, set):
        return Set[subtype]  # type: ignore
    else:
        raise ValueError(f"Unsupported top-level generic type: {collection}")


class PredicateType:
    """
    A base class for `types` in a domain language.  This serves much the same purpose as
    ``typing.Type``, but we add a few conveniences to these types, so we construct separate classes
    for them and group them together under ``PredicateType`` to have a good type annotation for
    these types.
    """

    @staticmethod
    def get_type(type_: Type) -> "PredicateType":
        """
        Converts a python ``Type`` (as you might get from a type annotation) into a
        ``PredicateType``.  If the ``Type`` is callable, this will return a ``FunctionType``;
        otherwise, it will return a ``BasicType``.

        ``BasicTypes`` have a single ``name`` parameter - we typically get this from
        ``type_.__name__``.  This doesn't work for generic types (like ``List[str]``), so we handle
        those specially, so that the ``name`` for the ``BasicType`` remains ``List[str]``, as you
        would expect.
        """
        if is_callable(type_):
            callable_args = type_.__args__
            argument_types = [PredicateType.get_type(t) for t in callable_args[:-1]]
            return_type = PredicateType.get_type(callable_args[-1])
            return FunctionType(argument_types, return_type)
        elif is_generic(type_):
            # This is something like List[int].  type_.__name__ doesn't do the right thing (and
            # crashes in python 3.7), so we need to do some magic here.
            name = get_generic_name(type_)
        else:
            name = type_.__name__
        return BasicType(name)

    @staticmethod
    def get_function_type(
        arg_types: Sequence["PredicateType"], return_type: "PredicateType"
    ) -> "PredicateType":
        """
        Constructs an NLTK ``ComplexType`` representing a function with the given argument and
        return types.
        """
        # TODO(mattg): We might need to generalize this to just `get_type`, so we can handle
        # functions as arguments correctly in the logic below.
        if not arg_types:
            # Functions with no arguments are basically constants whose type match their return
            # type.
            return return_type
        return FunctionType(arg_types, return_type)


class BasicType(PredicateType):
    """
    A ``PredicateType`` representing a zero-argument predicate (which could technically be a
    function with no arguments or a constant; both are treated the same here).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.name == other.name
        return NotImplemented


class FunctionType(PredicateType):
    """
    A ``PredicateType`` representing a function with arguments.  When seeing this as a string, it
    will be in angle brackets, with argument types separated by commas, and the return type
    separated from argument types with a colon.  For example, ``def f(a: str) -> int:`` would look
    like ``<str:int>``, and ``def g(a: int, b: int) -> int`` would look like ``<int,int:int>``.
    """

    def __init__(self, argument_types: Sequence[PredicateType], return_type: PredicateType) -> None:
        self.argument_types = argument_types
        self.return_type = return_type
        self.name = f'<{",".join(str(arg) for arg in argument_types)}:{return_type}>'

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.name == other.name
        return NotImplemented


def predicate(function: Callable) -> Callable:
    """
    This is intended to be used as a decorator when you are implementing your ``DomainLanguage``.
    This marks a function on a ``DomainLanguage`` subclass as a predicate that can be used in the
    language.  See the :class:`DomainLanguage` docstring for an example usage, and for what using
    this does.
    """
    setattr(function, "_is_predicate", True)
    return function


def predicate_with_side_args(side_arguments: List[str]) -> Callable:
    """
    Like :func:`predicate`, but used when some of the arguments to the function are meant to be
    provided by the decoder or other state, instead of from the language.  For example, you might
    want to have a function use the decoder's attention over some input text when a terminal was
    predicted.  That attention won't show up in the language productions.  Use this decorator, and
    pass in the required state to :func:`DomainLanguage.execute_action_sequence`, if you need to
    ignore some arguments when doing grammar induction.

    In order for this to work out, the side arguments `must` be after any non-side arguments.  This
    is because we use ``*args`` to pass the non-side arguments, and ``**kwargs`` to pass the side
    arguments, and python requires that ``*args`` be before ``**kwargs``.
    """

    def decorator(function: Callable) -> Callable:
        setattr(function, "_side_arguments", side_arguments)
        return predicate(function)

    return decorator


def nltk_tree_to_logical_form(tree: Tree) -> str:
    """
    Given an ``nltk.Tree`` representing the syntax tree that generates a logical form, this method
    produces the actual (lisp-like) logical form, with all of the non-terminal symbols converted
    into the correct number of parentheses.

    This is used in the logic that converts action sequences back into logical forms.  It's very
    unlikely that you will need this anywhere else.
    """
    # nltk.Tree actually inherits from `list`, so you use `len()` to get the number of children.
    # We're going to be explicit about checking length, instead of using `if tree:`, just to avoid
    # any funny business nltk might have done (e.g., it's really odd if `if tree:` evaluates to
    # `False` if there's a single leaf node with no children).
    if len(tree) == 0:
        return tree.label()
    if len(tree) == 1:
        return tree[0].label()
    return "(" + " ".join(nltk_tree_to_logical_form(child) for child in tree) + ")"


class DomainLanguage:
    """
    A ``DomainLanguage`` specifies the functions available to use for a semantic parsing task.  You
    write execution code for these functions, and we will automatically induce a grammar from those
    functions and give you a lisp interpreter that can use those functions.  For example:

    .. code-block:: python

        class Arithmetic(DomainLanguage):
            @predicate
            def add(self, num1: int, num2: int) -> int:
                return num1 + num2

            @predicate
            def halve(self, num: int) -> int:
                return num / 2

            ...

    Instantiating this class now gives you a language object that can parse and execute logical
    forms, can convert logical forms to action sequences (linearized abstract syntax trees) and
    back again, and can list all valid production rules in a grammar induced from the specified
    functions.

    .. code-block:: python

        >>> l = Arithmetic()
        >>> l.execute("(add 2 3)")
        5
        >>> l.execute("(halve (add 12 4))")
        8
        >>> l.logical_form_to_action_sequence("(add 2 3)")
        # See the docstring for this function for an description of what these strings mean.
        ['@start@ -> int', 'int -> [<int,int:int>, int, int]', '<int,int:int> -> add',
         'int -> 2', 'int -> 3']
        >>> l.action_sequence_to_logical_form(l.logical_form_to_action_sequence('(add 2 3)'))
        '(add 2 3)'
        >>> l.get_nonterminal_productions()
        {'<int,int:int>': ['add', 'divide', 'multiply', 'subtract'], '<int:int>': ['halve'], ...}

    This is done with some reflection magic, with the help of the ``@predicate`` decorator and type
    annotations.  For a method you define on a ``DomainLanguage`` subclass to be included in the
    language, it *must* be decorated with ``@predicate``, and it *must* have type annotations on
    all arguments and on its return type.  You can also add predicates and constants to the
    language using the :func:`add_predicate` and :func:`add_constant` functions, if you choose
    (minor point: constants with generic types (like ``Set[int]``) must currently be specified as
    predicates, as the ``allowed_constants`` dictionary doesn't pass along the generic type
    information).

    By default, the language we construct is purely functional - no defining variables or using
    lambda functions, or anything like that.  There are options to allow two extensions to the
    default language behavior, which together allow for behavior that is essentially equivalent to
    lambda functions: (1) function currying, and (2) function composition.  Currying is still
    functional, but allows only giving some of the arguments to a function, with a functional return
    type.  For example, if you allow currying, you can convert a two-argument function like
    ``(multiply 4 5)`` into a one-argument function like ``(multiply 4)`` (which would then multiply
    its single argument by 4).  Without being able to save variables, currying isn't `that` useful,
    so it is not enabled by default, but it can be used in conjunction with function composition to
    get the behavior of lambda functions without needing to explicitly deal with lambdas.  Function
    composition calls two functions in succession, passing the output of one as the input to
    another.  The difference between this and regular nested function calls is that it happens
    `outside` the nesting, so the input type of the outer function is the input type of the first
    function, not the second, as would be the case with nesting.  As a simple example, with function
    composition you could change the nested expression ``(sum (list1 8))`` into the equivalent
    expression ``((* sum list1) 8)``.  As a more useful example, consider taking an argmax over a
    list: ``(argmax (list3 5 9 2) sin)``, where this will call the ``sin`` function on each element
    of the list and return the element with the highest value.  If you want a more complex function
    when computing a value, say ``sin(3x)``, this would typically be done with lambda functions.  We
    can accomplish this with currying and function composition: ``(argmax (list3 5 9 2) (* sin
    (mutiply 3)))``.  In this way we do not need to introduce variables into the language, which are
    tricky from a modeling perspective.  All of the actual terminal productions in this version
    should have a reasonably strong correspondence with the words in the input utterance.

    Two important notes on currying and composition: first, in order to perform type inference on
    curried functions (to know which argument is being ommitted), we currently rely on `executing`
    the subexpressions.  This should be ok for simple, determinstic languages, but this is very much
    not recommended for things like NMNs at this point.  We'd need to implement smarter type
    inference for that to work.  Second, the grammar induction that we do for currying and
    composition is very permissive and quite likely overgenerates productions.  If you use this, you
    probably should double check all productions that were induced and make sure you really want
    them in your grammar, manually removing any that you don't want in your subclass after the
    grammar induction step (i.e., in your constructor, after calling `super().__init__()` and
    `self.get_nonterminal_productions()`, modify `self._nonterminal_productions` directly).

    We have rudimentary support for class hierarchies in the types that you provide.  This is done
    through adding constants multiple times with different types.  For example, say you have a
    ``Column`` class with ``NumberColumn`` and ``StringColumn`` subclasses.  You can have functions
    that take the base class ``Column`` as an argument, and other functions that take the
    subclasses.  These will get types like ``<List[Row],Column:List[str]>`` (for a "select"
    function that returns whatever cell text is in that column for the given rows), and
    ``<List[Row],NumberColumn,Number:List[Row]>`` (for a "greater_than" function that returns rows
    with a value in the column greater than the given number).  These will generate argument types
    of ``Column`` and ``NumberColumn``, respectively.  ``NumberColumn`` is a subclass of
    ``Column``, so we want the ``Column`` production to include all ``NumberColumns`` as options.
    This is done by calling ``add_constant()`` with each ``NumberColumn`` twice: once without a
    ``type_`` argument (which infers the type as ``NumberColumn``), and once with ``type_=Column``.
    You can see a concrete example of how this works in the
    :class:`~allennlp_semparse.domain_languages.wikitables_language.WikiTablesLanguage`.

    Parameters
    ----------
    allowed_constants : ``Dict[str, Any]``, optional (default=None)
        If given, we add all items in this dictionary as constants (instances of non-functional
        types) in the language.  You can also add them manually by calling ``add_constant`` in the
        constructor of your ``DomainLanguage``.
    start_types : ``Set[Type]``, optional (default=None)
        If given, we will constrain the set of start types in the grammar to be this set.
        Otherwise, we allow any type that we can get as a return type in the functions in the
        language.
    allow_function_currying : ``bool``, optional (default=False)
        If ``True``, we will add production rules to the grammar (and support in function execution,
        etc.) to curry all two-or-more-argument functions into one-argument functions.  See the
        above for a discussion of what this means and when you might want to do it.  If you set this
        to ``True``, you likely also want to set ``allow_function_composition`` to ``True``.
    allow_function_composition : ``bool``, optional (default=False)
        If ``True``, function composition as described above will be enabled in the language,
        including support for parsing expressions with function composition, for executing these
        expressions, and for converting them to and from action sequences.  If you set this to
        ``True``, you likely also want to set ``allow_function_composition`` to ``True``.
    """

    def __init__(
        self,
        allowed_constants: Dict[str, Any] = None,
        start_types: Set[Type] = None,
        allow_function_currying: bool = False,
        allow_function_composition: bool = False,
    ) -> None:
        self._allow_currying = allow_function_currying
        self._allow_composition = allow_function_composition
        self._functions: Dict[str, Callable] = {}
        self._function_types: Dict[str, List[PredicateType]] = defaultdict(list)
        self._start_types: Set[PredicateType] = {
            PredicateType.get_type(type_) for type_ in start_types
        }
        for name in dir(self):
            if isinstance(getattr(self, name), types.MethodType):
                function = getattr(self, name)
                if getattr(function, "_is_predicate", False):
                    side_arguments = getattr(function, "_side_arguments", None)
                    self.add_predicate(name, function, side_arguments)
        if allowed_constants:
            for name, value in allowed_constants.items():
                self.add_constant(name, value)
        # Caching this to avoid recomputing it every time `get_nonterminal_productions` is called.
        self._nonterminal_productions: Dict[str, List[str]] = None

    def execute(self, logical_form: str):
        """Executes a logical form, using whatever predicates you have defined."""
        if not hasattr(self, "_functions"):
            raise RuntimeError("You must call super().__init__() in your Language constructor")
        logical_form = logical_form.replace(",", " ")
        expression = util.lisp_to_nested_expression(logical_form)
        return self._execute_expression(expression)

    def execute_action_sequence(
        self, action_sequence: List[str], side_arguments: List[Dict] = None
    ):
        """
        Executes the program defined by an action sequence directly, without needing the overhead
        of translating to a logical form first.  For any given program, :func:`execute` and this
        function are equivalent, they just take different representations of the program, so you
        can use whichever is more efficient.

        Also, if you have state or side arguments associated with particular production rules
        (e.g., the decoder's attention on an input utterance when a predicate was predicted), you
        `must` use this function to execute the logical form, instead of :func:`execute`, so that
        we can match the side arguments with the right functions.
        """
        # We'll strip off the first action, because it doesn't matter for execution.
        first_action = action_sequence[0]
        left_side = first_action.split(" -> ")[0]
        if left_side != "@start@":
            raise ExecutionError("invalid action sequence")
        remaining_actions = action_sequence[1:]
        remaining_side_args = side_arguments[1:] if side_arguments else None
        return self._execute_sequence(remaining_actions, remaining_side_args)[0]

    def get_nonterminal_productions(self) -> Dict[str, List[str]]:
        """
        Induces a grammar from the defined collection of predicates in this language and returns
        all productions in that grammar, keyed by the non-terminal they are expanding.

        This includes terminal productions implied by each predicate as well as productions for the
        `return type` of each defined predicate.  For example, defining a "multiply" predicate adds
        a "<int,int:int> -> multiply" terminal production to the grammar, and `also` a "int ->
        [<int,int:int>, int, int]" non-terminal production, because I can use the "multiply"
        predicate to produce an int.
        """
        if not self._nonterminal_productions:
            actions: Dict[Union[str, PredicateType], Set[str]] = defaultdict(set)
            # If you didn't give us a set of valid start types, we'll assume all types we know
            # about (including functional types) are valid start types.
            if self._start_types:
                start_types = self._start_types
            else:
                start_types = set()
                for type_list in self._function_types.values():
                    start_types.update(type_list)
            for start_type in start_types:
                actions[START_SYMBOL].add(f"{START_SYMBOL} -> {start_type}")
            for name, function_type_list in self._function_types.items():
                for function_type in function_type_list:
                    actions[function_type].add(f"{function_type} -> {name}")
                    if isinstance(function_type, FunctionType):
                        return_type = function_type.return_type
                        arg_types = function_type.argument_types
                        right_side = f"[{function_type}, {', '.join(str(arg_type) for arg_type in arg_types)}]"
                        actions[return_type].add(f"{return_type} -> {right_side}")

            if self._allow_currying:
                function_types = [t for t in actions if isinstance(t, FunctionType)]
                for function_type in function_types:
                    if len(function_type.argument_types) > 1:
                        argument_types = list(set(function_type.argument_types))
                        for uncurried_arg_type in argument_types:
                            curried_arg_types = list(
                                reversed([t for t in function_type.argument_types])
                            )
                            curried_arg_types.remove(uncurried_arg_type)
                            curried_arg_types.reverse()
                            curried_function_type = PredicateType.get_function_type(
                                [uncurried_arg_type], function_type.return_type
                            )
                            right_side = f'[{function_type}, {", ".join(str(arg) for arg in curried_arg_types)}]'
                            actions[curried_function_type].add(
                                f"{curried_function_type} -> {right_side}"
                            )

            if self._allow_composition:
                function_types = [t for t in actions if isinstance(t, FunctionType)]
                for type1 in function_types:
                    for type2 in function_types:
                        if len(type1.argument_types) != 1:
                            continue
                        if type1.argument_types[0] != type2.return_type:
                            continue
                        composed_type = PredicateType.get_function_type(
                            type2.argument_types, type1.return_type
                        )
                        right_side = f"[*, {type1}, {type2}]"
                        actions[composed_type].add(f"{composed_type} -> {right_side}")

            self._nonterminal_productions = {
                str(key): sorted(value) for key, value in actions.items()
            }
        return self._nonterminal_productions

    def all_possible_productions(self) -> List[str]:
        """
        Returns a sorted list of all production rules in the grammar induced by
        :func:`get_nonterminal_productions`.
        """
        all_actions = set()
        for action_set in self.get_nonterminal_productions().values():
            all_actions.update(action_set)
        return sorted(all_actions)

    def logical_form_to_action_sequence(self, logical_form: str) -> List[str]:
        """
        Converts a logical form into a linearization of the production rules from its abstract
        syntax tree.  The linearization is top-down, depth-first.

        Each production rule is formatted as "LHS -> RHS", where "LHS" is a single non-terminal
        type, and RHS is either a terminal or a list of non-terminals (other possible values for
        RHS in a more general context-free grammar are not produced by our grammar induction
        logic).

        Non-terminals are `types` in the grammar, either basic types (like ``int``, ``str``, or
        some class that you define), or functional types, represented with angle brackets with a
        colon separating arguments from the return type.  Multi-argument functions have commas
        separating their argument types.  For example, ``<int:int>`` is a function that takes an
        integer and returns an integer, and ``<int,int:int>`` is a function that takes two integer
        arguments and returns an integer.

        As an example translation from logical form to complete action sequence, the logical form
        ``(add 2 3)`` would be translated to ``['@start@ -> int', 'int -> [<int,int:int>, int, int]',
        '<int,int:int> -> add', 'int -> 2', 'int -> 3']``.
        """
        expression = util.lisp_to_nested_expression(logical_form)
        try:
            transitions, start_type = self._get_transitions(expression, expected_type=None)
            if self._start_types and start_type not in self._start_types:
                raise ParsingError(
                    f"Expression had unallowed start type of {start_type}: {expression}"
                )
        except ParsingError as error:
            logger.error(f"Error parsing logical form: {logical_form}: {error}")
            raise
        transitions.insert(0, f"@start@ -> {start_type}")
        return transitions

    def action_sequence_to_logical_form(self, action_sequence: List[str]) -> str:
        """
        Takes an action sequence as produced by :func:`logical_form_to_action_sequence`, which is a
        linearization of an abstract syntax tree, and reconstructs the logical form defined by that
        abstract syntax tree.
        """
        # Basic outline: we assume that the bracketing that we get in the RHS of each action is the
        # correct bracketing for reconstructing the logical form.  This is true when there is no
        # currying in the action sequence.  Given this assumption, we just need to construct a tree
        # from the action sequence, then output all of the leaves in the tree, with brackets around
        # the children of all non-terminal nodes.

        remaining_actions = [action.split(" -> ") for action in action_sequence]
        tree = Tree(remaining_actions[0][1], [])

        try:
            remaining_actions = self._construct_node_from_actions(tree, remaining_actions[1:])
        except ParsingError:
            logger.error("Error parsing action sequence: %s", action_sequence)
            raise

        if remaining_actions:
            logger.error("Error parsing action sequence: %s", action_sequence)
            logger.error("Remaining actions were: %s", remaining_actions)
            raise ParsingError("Extra actions in action sequence")
        return nltk_tree_to_logical_form(tree)

    def add_predicate(self, name: str, function: Callable, side_arguments: List[str] = None):
        """
        Adds a predicate to this domain language.  Typically you do this with the ``@predicate``
        decorator on the methods in your class.  But, if you need to for whatever reason, you can
        also call this function yourself with a (type-annotated) function to add it to your
        language.

        Parameters
        ----------
        name : ``str``
            The name that we will use in the induced language for this function.
        function : ``Callable``
            The function that gets called when executing a predicate with the given name.
        side_arguments : ``List[str]``, optional
            If given, we will ignore these arguments for the purposes of grammar induction.  This
            is to allow passing extra arguments from the decoder state that are not explicitly part
            of the language the decoder produces, such as the decoder's attention over the question
            when a terminal was predicted.  If you use this functionality, you also `must` use
            ``language.execute_action_sequence()`` instead of ``language.execute()``, and you must
            pass the additional side arguments needed to that function.  See
            :func:`execute_action_sequence` for more information.
        """
        side_arguments = side_arguments or []
        signature = inspect.signature(function)
        argument_types = [
            param.annotation
            for name, param in signature.parameters.items()
            if name not in side_arguments
        ]
        return_type = signature.return_annotation
        argument_nltk_types: List[PredicateType] = [
            PredicateType.get_type(arg_type) for arg_type in argument_types
        ]
        return_nltk_type = PredicateType.get_type(return_type)
        function_nltk_type = PredicateType.get_function_type(argument_nltk_types, return_nltk_type)
        self._functions[name] = function
        self._function_types[name].append(function_nltk_type)

    def add_constant(self, name: str, value: Any, type_: Type = None):
        """
        Adds a constant to this domain language.  You would typically just pass in a list of
        constants to the ``super().__init__()`` call in your constructor, but you can also call
        this method to add constants if it is more convenient.

        Because we construct a grammar over this language for you, in order for the grammar to be
        finite we cannot allow arbitrary constants.  Having a finite grammar is important when
        you're doing semantic parsing - we need to be able to search over this space, and compute
        normalized probability distributions.
        """
        value_type = type_ if type_ else type(value)
        constant_type = PredicateType.get_type(value_type)
        self._functions[name] = lambda: value
        self._function_types[name].append(constant_type)

    def is_nonterminal(self, symbol: str) -> bool:
        """
        Determines whether an input symbol is a valid non-terminal in the grammar.
        """
        nonterminal_productions = self.get_nonterminal_productions()
        return symbol in nonterminal_productions

    def _execute_expression(self, expression: Any):
        """
        This does the bulk of the work of executing a logical form, recursively executing a single
        expression.  Basically, if the expression is a function we know about, we evaluate its
        arguments then call the function.  If it's a list, we evaluate all elements of the list.
        If it's a constant (or a zero-argument function), we evaluate the constant.
        """
        if isinstance(expression, list):
            if isinstance(expression[0], list):
                function = self._execute_expression(expression[0])
            elif expression[0] in self._functions:
                function = self._functions[expression[0]]
            elif self._allow_composition and expression[0] == "*":
                function = "*"
            else:
                if isinstance(expression[0], str):
                    raise ExecutionError(f"Unrecognized function: {expression[0]}")
                else:
                    raise ExecutionError(f"Unsupported expression type: {expression}")
            arguments = [self._execute_expression(arg) for arg in expression[1:]]
            try:
                if self._allow_composition and function == "*":
                    return self._create_composed_function(arguments[0], arguments[1])
                return function(*arguments)
            except (TypeError, ValueError):
                if self._allow_currying:
                    # If we got here, then maybe the error is because this should be a curried
                    # function.  We'll check for that and return the curried function if possible.
                    curried_function = self._get_curried_function(function, arguments)
                    if curried_function:
                        return curried_function
                traceback.print_exc()
                raise ExecutionError(
                    f"Error executing expression {expression} (see stderr for stack trace)"
                )
        elif isinstance(expression, str):
            if expression not in self._functions:
                raise ExecutionError(f"Unrecognized constant: {expression}")
            # This is a bit of a quirk in how we represent constants and zero-argument functions.
            # For consistency, constants are wrapped in a zero-argument lambda.  So both constants
            # and zero-argument functions are callable in `self._functions`, and are `BasicTypes`
            # in `self._function_types`.  For these, we want to return
            # `self._functions[expression]()` _calling_ the zero-argument function.  If we get a
            # `FunctionType` in here, that means we're referring to the function as a first-class
            # object, instead of calling it (maybe as an argument to a higher-order function).  In
            # that case, we return the function _without_ calling it.
            # Also, we just check the first function type here, because we assume you haven't
            # registered the same function with both a constant type and a `FunctionType`.
            if isinstance(self._function_types[expression][0], FunctionType):
                return self._functions[expression]
            else:
                return self._functions[expression]()
            return self._functions[expression]
        else:
            raise ExecutionError(
                "Not sure how you got here. Please open a github issue with details."
            )

    def _execute_sequence(
        self, action_sequence: List[str], side_arguments: List[Dict]
    ) -> Tuple[Any, List[str], List[Dict]]:
        """
        This does the bulk of the work of :func:`execute_action_sequence`, recursively executing
        the functions it finds and trimming actions off of the action sequence.  The return value
        is a tuple of (execution, remaining_actions), where the second value is necessary to handle
        the recursion.
        """
        if not action_sequence:
            raise ExecutionError("invalid action sequence")
        first_action = action_sequence[0]
        remaining_actions = action_sequence[1:]
        remaining_side_args = side_arguments[1:] if side_arguments else None
        right_side = first_action.split(" -> ")[1]
        if right_side in self._functions:
            function = self._functions[right_side]
            # mypy doesn't like this check, saying that Callable isn't a reasonable thing to pass
            # here.  But it works just fine; I'm not sure why mypy complains about it.
            if isinstance(function, Callable):  # type: ignore
                function_arguments = inspect.signature(function).parameters
                if not function_arguments:
                    # This was a zero-argument function / constant that was registered as a lambda
                    # function, for consistency of execution in `execute()`.
                    execution_value = function()
                elif side_arguments:
                    kwargs = {}
                    non_kwargs = []
                    for argument_name in function_arguments:
                        if argument_name in side_arguments[0]:
                            kwargs[argument_name] = side_arguments[0][argument_name]
                        else:
                            non_kwargs.append(argument_name)
                    if kwargs and non_kwargs:
                        # This is a function that has both side arguments and logical form
                        # arguments - we curry the function so only the logical form arguments are
                        # left.
                        def curried_function(*args):
                            return function(*args, **kwargs)

                        execution_value = curried_function
                    elif kwargs:
                        # This is a function that _only_ has side arguments - we just call the
                        # function and return a value.
                        execution_value = function(**kwargs)
                    else:
                        # This is a function that has logical form arguments, but no side arguments
                        # that match what we were given - just return the function itself.
                        execution_value = function
                else:
                    execution_value = function
            return execution_value, remaining_actions, remaining_side_args
        else:
            # This is a non-terminal expansion, like 'int -> [<int:int>, int, int]'.  We need to
            # get the function and its arguments, then call the function with its arguments.
            # Because we linearize the abstract syntax tree depth first, left-to-right, we can just
            # recursively call `_execute_sequence` for the function and all of its arguments, and
            # things will just work.
            right_side_parts = right_side.split(", ")

            if right_side_parts[0] == "[*" and self._allow_composition:
                # This one we need to handle differently, because the "function" is a function
                # composition which doesn't show up in the action sequence.
                function = "*"  # type: ignore
            else:
                # Otherwise, we grab the function itself by executing the next self-contained action
                # sequence (if this is a simple function call, that will be exactly one action; if
                # it's a higher-order function, it could be many actions).
                function, remaining_actions, remaining_side_args = self._execute_sequence(
                    remaining_actions, remaining_side_args
                )
            # We don't really need to know what the types of the arguments are, just how many of them
            # there are, so we recurse the right number of times.
            arguments = []
            for _ in right_side_parts[1:]:
                argument, remaining_actions, remaining_side_args = self._execute_sequence(
                    remaining_actions, remaining_side_args
                )
                arguments.append(argument)
            if self._allow_composition and function == "*":
                # In this case, both arguments should be functions, and we want to compose them, by
                # calling the second argument first, and passing the result to the first argument.
                def composed_function(*args):
                    function_argument, is_curried = self._execute_function(arguments[1], list(args))
                    if is_curried:
                        # If the inner function ended up curried, we have to curry the outer
                        # function too.
                        return_type = inspect.signature(arguments[0]).return_annotation
                        inner_signature = inspect.signature(function_argument)
                        arg_type = list(inner_signature.parameters.values())[0].annotation

                        # Pretty cool that you can give runtime types to a function defined at
                        # runtime, but mypy has no idea what to do with this.
                        def curried_function(x: arg_type) -> return_type:  # type: ignore
                            return arguments[0](function_argument(x))

                        function_value = curried_function
                    else:
                        function_value, _ = self._execute_function(
                            arguments[0], [function_argument]
                        )
                    return function_value

                return composed_function, remaining_actions, remaining_side_args
            function_value, _ = self._execute_function(function, arguments)
            return function_value, remaining_actions, remaining_side_args

    def _execute_function(self, function: Callable, arguments: List[Any]) -> Any:
        """
        Calls `function` with the given `arguments`, allowing for the possibility of currying the
        `function`.
        """
        is_curried = False
        try:
            function_value = function(*arguments)
        except TypeError:
            if not self._allow_currying:
                raise
            # If we got here, then maybe the error is because this should be a curried
            # function.  We'll check for that and return the curried function if possible.
            curried_function = self._get_curried_function(function, arguments)
            if curried_function:
                function_value = curried_function
                is_curried = True
            else:
                raise
        return function_value, is_curried

    def _get_transitions(
        self, expression: Any, expected_type: PredicateType
    ) -> Tuple[List[str], PredicateType]:
        """
        This is used when converting a logical form into an action sequence.  This piece
        recursively translates a lisp expression into an action sequence, making sure we match the
        expected type (or using the expected type to get the right type for constant expressions).
        """
        if isinstance(expression, (list, tuple)):
            function_transitions, return_type, argument_types = self._get_function_transitions(
                expression, expected_type
            )
            if len(argument_types) != len(expression[1:]):
                raise ParsingError(f"Wrong number of arguments for function in {expression}")
            argument_transitions = []
            for argument_type, subexpression in zip(argument_types, expression[1:]):
                argument_transitions.extend(self._get_transitions(subexpression, argument_type)[0])
            return function_transitions + argument_transitions, return_type
        elif isinstance(expression, str):
            if expression not in self._functions:
                raise ParsingError(f"Unrecognized constant: {expression}")
            constant_types = self._function_types[expression]
            if len(constant_types) == 1:
                constant_type = constant_types[0]
                # This constant had only one type; that's the easy case.
                if expected_type and expected_type != constant_type:
                    raise ParsingError(
                        f"{expression} did not have expected type {expected_type} "
                        f"(found {constant_type})"
                    )
                return [f"{constant_type} -> {expression}"], constant_type
            else:
                if not expected_type:
                    raise ParsingError(
                        "With no expected type and multiple types to pick from "
                        f"I don't know what type to use (constant was {expression})"
                    )
                if expected_type not in constant_types:
                    raise ParsingError(
                        f"{expression} did not have expected type {expected_type} "
                        f"(found these options: {constant_types}; none matched)"
                    )
                return [f"{expected_type} -> {expression}"], expected_type

        else:
            raise ParsingError(
                "Not sure how you got here. Please open an issue on github with details."
            )

    def _get_function_transitions(
        self, expression: Sequence, expected_type: PredicateType
    ) -> Tuple[List[str], PredicateType, Sequence[PredicateType]]:
        """
        A helper method for ``_get_transitions``.  This gets the transitions for the predicate
        itself in a function call.  If we only had simple functions (e.g., "(add 2 3)"), this would
        be pretty straightforward and we wouldn't need a separate method to handle it.  We split it
        out into its own method because handling higher-order functions and currying is complicated
        (e.g., something like "((negate add) 2 3)" or "((multiply 3) 2)").
        """
        function_expression = expression[0]
        # This first block handles getting the transitions and function type (and some error
        # checking) _just for the function itself_.  If this is a simple function, this is easy; if
        # it's a higher-order function, it involves some recursion.
        if isinstance(function_expression, list):
            # This is a higher-order function.  TODO(mattg): we'll just ignore type checking on
            # higher-order functions, for now.
            # Some annoying redirection here to make mypy happy; need to specify the type of
            # function_type.
            result = self._get_transitions(function_expression, None)
            transitions = result[0]
            function_type: FunctionType = result[1]  # type: ignore
            # This is a bit of an unfortunate hack. In order to handle currying, we currently rely
            # on executing the function, for which we need actual function code (see the class
            # docstring).  I want to avoid executing the function prematurely, though, so this still
            # works in cases where you don't need to handle currying higher-order functions.  So,
            # we'll leave this as `None` and handle it below, if indeed you are currying this
            # function.
            function = None
        elif function_expression in self._functions:
            name = function_expression
            function_types = self._function_types[function_expression]
            if len(function_types) != 1:
                raise ParsingError(
                    f"{function_expression} had multiple types; this is not yet supported for functions"
                )
            function_type = function_types[0]  # type: ignore

            transitions = [f"{function_type} -> {name}"]
            function = self._functions[function_expression]

        elif self._allow_composition and function_expression == "*":
            outer_function_expression = expression[1]
            if not isinstance(outer_function_expression, list):
                outer_function_expression = [outer_function_expression]
            inner_function_expression = expression[2]
            if not isinstance(inner_function_expression, list):
                inner_function_expression = [inner_function_expression]

            # This is unfortunately a bit complex.  What we really what is the _type_ of these
            # expressions.  We don't have a function that will give us that.  Instead, we have a
            # function that will give us return types and argument types.  If we have a bare
            # function name, like "sum", this works fine.  But if it's a higher-order function
            # (including a curried function), then the return types and argument types from
            # _get_function_transitions aren't what we're looking for here, because that function is
            # designed for something else.  We need to hack our way around that a bit, by grabbing
            # the return type from the inner return type (confusing, I know).
            _, outer_return_type, outer_arg_types = self._get_function_transitions(
                outer_function_expression, None
            )
            if isinstance(expression[1], list):
                outer_function_type: FunctionType = outer_return_type  # type: ignore
            else:
                outer_function_type = PredicateType.get_function_type(  # type: ignore
                    outer_arg_types, outer_return_type
                )

            _, inner_return_type, inner_arg_types = self._get_function_transitions(
                inner_function_expression, None
            )
            if isinstance(expression[2], list):
                inner_function_type: FunctionType = inner_return_type  # type: ignore
            else:
                inner_function_type = PredicateType.get_function_type(  # type: ignore
                    inner_arg_types, inner_return_type
                )

            composition_argument_types = [outer_function_type, inner_function_type]
            composition_type = PredicateType.get_function_type(
                inner_function_type.argument_types, outer_function_type.return_type
            )
            right_side = f'[*, {", ".join(str(arg) for arg in composition_argument_types)}]'
            composition_transition = f"{composition_type} -> {right_side}"
            return [composition_transition], composition_type, composition_argument_types

        else:
            if isinstance(function_expression, str):
                raise ParsingError(f"Unrecognized function: {function_expression[0]}")
            else:
                raise ParsingError(f"Unsupported function_expression type: {function_expression}")

        if not isinstance(function_type, FunctionType):
            raise ParsingError(f"Zero-arg function or constant called with arguments: {name}")

        # Now that we have the transitions for the function itself, and the function's type, we can
        # get argument types and do the rest of the transitions.  The first thing we need to do is
        # check if we need to curry this function, because we're missing an argument.
        if (
            self._allow_currying
            # This check means we're missing an argument to the function.
            and len(expression) > 1
            and len(function_type.argument_types) - 1 == len(expression) - 1
        ):
            # If we're currying this function, we need to add a transition that encodes the
            # currying, and change the function_type accordingly.
            arguments = [self._execute_expression(e) for e in expression[1:]]
            if function is None:
                function = self._execute_expression(function_expression)
            curried_function = self._get_curried_function(function, arguments)

            # Here we get the FunctionType corresponding to the new, curried function.
            signature = inspect.signature(curried_function)
            return_type = PredicateType.get_type(signature.return_annotation)
            uncurried_arg_type = PredicateType.get_type(
                list(signature.parameters.values())[0].annotation
            )
            curried_function_type = PredicateType.get_function_type(
                [uncurried_arg_type], return_type
            )

            # To fit in with the logic below, we need to basically make a fake `curry`
            # FunctionType, with the arguments being the function we're currying and all of the
            # curried arguments, and the return type being the one-argument function.  Then we
            # can re-use all of the existing logic without modification.
            curried_arg_types = list(reversed([t for t in function_type.argument_types]))
            curried_arg_types.remove(uncurried_arg_type)
            curried_arg_types.reverse()
            right_side = f'[{function_type}, {", ".join(str(arg) for arg in curried_arg_types)}]'
            curry_transition = f"{curried_function_type} -> {right_side}"
            transitions.insert(0, curry_transition)
            return transitions, curried_function_type, curried_arg_types

        argument_types = function_type.argument_types
        return_type = function_type.return_type
        right_side = f'[{function_type}, {", ".join(str(arg) for arg in argument_types)}]'
        first_transition = f"{return_type} -> {right_side}"
        transitions.insert(0, first_transition)
        if expected_type and expected_type != return_type:
            raise ParsingError(
                f"{function_expression} did not have expected type {expected_type} "
                f"(found {return_type})"
            )
        return transitions, return_type, argument_types

    def _construct_node_from_actions(
        self, current_node: Tree, remaining_actions: List[List[str]]
    ) -> List[List[str]]:
        """
        Given a current node in the logical form tree, and a list of actions in an action sequence,
        this method fills in the children of the current node from the action sequence, then
        returns whatever actions are left.

        For example, we could get a node with type ``c``, and an action sequence that begins with
        ``c -> [<r,c>, r]``.  This method will add two children to the input node, consuming
        actions from the action sequence for nodes of type ``<r,c>`` (and all of its children,
        recursively) and ``r`` (and all of its children, recursively).  This method assumes that
        action sequences are produced `depth-first`, so all actions for the subtree under ``<r,c>``
        appear before actions for the subtree under ``r``.  If there are any actions in the action
        sequence after the ``<r,c>`` and ``r`` subtrees have terminated in leaf nodes, they will be
        returned.
        """
        if not remaining_actions:
            logger.error("No actions left to construct current node: %s", current_node)
            raise ParsingError("Incomplete action sequence")
        left_side, right_side = remaining_actions.pop(0)
        if left_side != current_node.label():
            logger.error("Current node: %s", current_node)
            logger.error("Next action: %s -> %s", left_side, right_side)
            logger.error("Remaining actions were: %s", remaining_actions)
            raise ParsingError("Current node does not match next action")
        if right_side[0] == "[":
            # This is a non-terminal expansion, with more than one child node.
            for child_type in right_side[1:-1].split(", "):
                child_node = Tree(child_type, [])
                current_node.append(child_node)  # you add a child to an nltk.Tree with `append`
                # For now, we assume that all children in a list like this are non-terminals, so we
                # recurse on them.  I'm pretty sure that will always be true for the way our
                # grammar induction works.  We can revisit this later if we need to.
                if self._allow_composition and child_type == "*":
                    # One exception to the comment above is when we are doing function composition.
                    # The function composition operator * does not have a corresponding action, so
                    # the recursion on constructing that node doesn't work.
                    continue
                remaining_actions = self._construct_node_from_actions(child_node, remaining_actions)
        else:
            # The current node is a pre-terminal; we'll add a single terminal child.  By
            # construction, the right-hand side of our production rules are only ever terminal
            # productions or lists of non-terminals.
            current_node.append(
                Tree(right_side, [])
            )  # you add a child to an nltk.Tree with `append`
        return remaining_actions

    def _get_curried_function(self, function: Callable, arguments: List[Any]) -> Optional[Callable]:
        signature = inspect.signature(function)
        parameters = signature.parameters
        if len(parameters) != len(arguments) + 1:
            # We only allow currying that makes a function into a one-argument function.  This is to
            # simplify both the complexity of the `DomainLanguage` code and the complexity of
            # whatever model might use the resulting grammar.  For all currently-envisioned uses of
            # currying, we only need to make one-argument functions.  These are predominantly for
            # replacing lambda functions in argmaxes and the like.
            return None
        # Now we have to decide where the missing argument goes in the list of arguments.  We will
        # look at types to figure that out, and arbitrarily say that if there are multiple matching
        # types, the missing one comes last.
        missing_arg_index = 0
        parameter_types = list(parameters.values())
        for parameter in parameter_types:
            argument = arguments[missing_arg_index]
            if isinstance(argument, (list, set)):
                arg_type = infer_collection_type(argument)
            else:
                arg_type = type(argument)
            if parameter.annotation == arg_type:
                missing_arg_index += 1
                if missing_arg_index == len(parameters) - 1:
                    break
            else:
                break

        arg_type = parameter_types[missing_arg_index].annotation

        # Pretty cool that you can give runtime types to a function defined at runtime, but mypy has
        # no idea what to do with this.
        def curried_function(x: arg_type) -> signature.return_annotation:  # type: ignore
            new_arguments = arguments[:missing_arg_index] + [x] + arguments[missing_arg_index:]
            return function(*new_arguments)

        return curried_function

    def _create_composed_function(
        self, outer_function: Callable, inner_function: Callable
    ) -> Callable:
        """
        Creating a composed function is easy; just do a `def` and call the functions in order.  This
        function exists because we need the composed function to have _correct type annotations_,
        which is harder.  We can't use `*args` for the lambda function that we construct, so we need
        to switch on how many arguments the inner function takes, and create functions with the
        right argument type annotations.

        And, as in other places where we assign types at runtime, mypy has no idea what's going on,
        so we tell it to ignore this code.  `inspect` will do the right thing, even if mypy can't
        analyze it.
        """
        inner_signature = inspect.signature(inner_function)
        outer_signature = inspect.signature(outer_function)
        argument_types = [arg.annotation for arg in inner_signature.parameters.values()]
        return_type = outer_signature.return_annotation

        if len(argument_types) == 1:

            def composed_function(arg1: argument_types[0]) -> return_type:  # type: ignore
                return outer_function(inner_function(arg1))

        elif len(argument_types) == 2:

            def composed_function(  # type: ignore
                arg1: argument_types[0], arg2: argument_types[1]  # type:ignore
            ) -> return_type:  # type: ignore
                return outer_function(inner_function(arg1, arg2))

        elif len(argument_types) == 3:

            def composed_function(  # type:ignore
                arg1: argument_types[0],  # type: ignore
                arg2: argument_types[1],  # type: ignore
                arg3: argument_types[2],  # type:ignore
            ) -> return_type:  # type: ignore
                return outer_function(inner_function(arg1, arg2, arg3))

        elif len(argument_types) == 4:

            def composed_function(  # type:ignore
                arg1: argument_types[0],  # type:ignore
                arg2: argument_types[1],  # type:ignore
                arg3: argument_types[2],  # type:ignore
                arg4: argument_types[3],  # type:ignore
            ) -> return_type:  # type: ignore
                return outer_function(inner_function(arg1, arg2, arg3, arg4))

        else:
            raise ValueError(
                f"Inner function has a type signature that's not currently handled: {inner_function}"
            )

        return composed_function

    def __len__(self):
        # This method exists just to make it easier to use this in a MetadataField.  Kind of
        # annoying, but oh well.
        return 0
