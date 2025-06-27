import copy
import re
from anytree import NodeMixin, RenderTree
from easydict import EasyDict as edict

# This function is a direct dependency for the .render() method in ProdBase
def Substitution(template, lang=None):
    def apply_to_all_args(f):
        def decorator(func):
            def wrapper(*args, **kwargs):
                new_args = [f(arg) for arg in args]
                return func(*new_args, **kwargs)
            return wrapper
        return decorator

    def replace_template(template, a):
        # Make numbers formattable 0 -> {0}
        wrap = lambda s: (re.sub(r'(\d+)', r'{\1}', s) if isinstance(s, str) else s)

        # Function to safely replace only unescaped '?'
        def replace_match(m):
            slot_idx = int(m.group(1))
            replacement = m.group(2)
            # Replace '?' that is not preceded by a backslash
            return re.sub(r'(?<!\\)\?', replacement, a[slot_idx])

        inner_replaced = re.sub(r"(\d+)\[\?←(.+?)\]", replace_match, template)
        output = wrap(inner_replaced).format(*a)

        # Convert escaped "\?" back to "?" after processing
        return output.replace(r'\?', '?')

    """i[?←expr] replaces ? (but not \?) in slot i with expr"""
    @apply_to_all_args(lambda x: x.render(lang) if hasattr(x, 'render') and not isinstance(x, str) else x)
    def sub(*a, **ka):
        return replace_template(template, a)

    return sub


class ProdBase:
    """
    A base class for production nodes, containing shared logic
    to avoid code duplication between Production and LightProduction.
    """
    def __init__(self, rule=None, type=None, state=dict()):
        self.rule = rule or ''
        # If type is not given, infer it from the rule's name
        self.type = type or (self.rule.name if self.rule else None)
        self.state = {"parents": [], **state}
        self.cache = dict()
        self.children = []
        # Parent and children list are managed by the subclasses' __init__

    def _create_children(self, production_class):
        """
        A helper method for subclasses to correctly instantiate children
        of their own type, establishing the parent-child relationship.
        """
        if self.rule:
            self.state.update(self.rule.state)
            self.children = [production_class(type=c, state=self.state, parent=self)
                             for c in self.rule.args]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __eq__(self, other):
        lang = self.rule.langs[0]
        return self @ lang == other @ lang

    def __getitem__(self, key):
        return self.children[key]

    def __len__(self):
        return len(self.children)

    def check(self, mode='args'):
        """Checks if the node and its descendants satisfy grammar constraints."""
        if not self.rule:
            return True

        if mode == 'args':
            arguments = self.children
            if not all(x.rule for x in arguments):
                return True
            return all(constraint(arguments) for constraint in self.rule.constraint)

        if mode == 'state':
            nodes_to_check = [self] + self._get_ancestors()
            return all(constraint(x)
                       for x in nodes_to_check
                       for constraint in (x.rule.state_constraint if x.rule else []))
        return True

    def render(self, lang):
        """
        Renders the production to a string in the specified language.
        The case for `lang=None` is handled by subclasses.
        """
        if lang in self.cache:
            return self.cache[lang]

        try:
            template = self.rule.templates[lang]
        except (AttributeError, KeyError):
            return "#" + self.type

        args = [child.render(lang) for child in self.children]

        if isinstance(template, str):
            if '?←' in template:
                template_func = Substitution(template, lang)
            else:
                template_func = template.format
        else:  # It's a function
            template_func = template

        out = template_func(*args)

        if "#" not in out:
            self.cache[lang] = out
        return out

    def dict(self, use_cls=False):
        return edict({l: self @ l for l in self.rule.langs} | (dict(cls=self) if use_cls else dict()))

    # Subclasses are expected to implement these
    def _get_ancestors(self):
        raise NotImplementedError("Subclasses must implement _get_ancestors.")

    def __repr__(self):
        raise NotImplementedError("Subclasses must implement __repr__.")


class Production(ProdBase, NodeMixin):
    """The original Production class, now inheriting from ProdBase and anytree.NodeMixin."""
    def __init__(self, rule=None, type=None, state=dict(), parent=None):
        ProdBase.__init__(self, rule=rule, type=type, state=state)
        # The parent argument is handled by NodeMixin to build the tree
        self.parent = parent
        self._create_children(Production)

    def _get_ancestors(self):
        # anytree.NodeMixin provides the .ancestors property
        return self.ancestors

    def render(self, lang=None):
        if lang is None:
            return str(RenderTree(self))
        # For a specific language, use the base class's render method
        return super().render(lang)

    __matmul__ = render

    def __repr__(self):
        return f"PROD:{self.type}" + (str(self.rule.args) if self.rule else '')

    def __deepcopy__(self, memo):
        # Custom deepcopy is needed due to the anytree structure
        new_obj = Production.__new__(Production)
        memo[id(self)] = new_obj
        new_obj.rule = self.rule
        new_obj.type = self.type
        new_obj.state = copy.deepcopy(self.state)
        new_obj.parent = None
        new_obj.cache = self.cache
        new_obj.children = [copy.deepcopy(child, memo) for child in self.children]
        return new_obj


class LightProduction(ProdBase):
    """
    A lightweight alternative to the 'Production' class, without anytree overhead,
    now inheriting common logic from ProdBase.
    """
    def __init__(self, rule=None, type=None, state=dict(), parent=None):
        super().__init__(rule=rule, type=type, state=state)
        self.parent = parent
        self._create_children(LightProduction)

    def _get_ancestors(self):
        """Manually traverses up the tree to get ancestors."""
        ancestors = []
        node = self.parent
        while node:
            ancestors.append(node)
            node = node.parent
        return ancestors

    @property
    def leaves(self):
        """Manually find all leaves of this node."""
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.leaves)
        return leaves

    def render(self, lang=None):
        if lang is None:
            # anytree.RenderTree is not available, so return the repr
            return self.__repr__()
        return super().render(lang)

    __matmul__ = render

    def __repr__(self):
        return f"LightPROD:{self.type}" + (str(self.rule.args) if self.rule else '')