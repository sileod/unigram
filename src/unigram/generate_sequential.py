import random
from collections import deque, ChainMap
from functools import lru_cache
import anytree
from easydict import EasyDict as edict


import re
def Substitution(template, lang=None):
    def replace_template(template, a):
        # Make numbers formattable 0 -> {0}
        wrap = lambda s: (re.sub(r'(\d+)', r'{\1}', s) if isinstance(s, str) else s)

        # Function to safely replace only unescaped '?'
        def replace_match(m):
            slot_idx = int(m.group(1))
            replacement = m.group(2)
            # The children args 'a' might be FastProduction nodes, so render them if needed
            arg_to_sub = a[slot_idx]
            if not isinstance(arg_to_sub, str):
                 arg_to_sub = arg_to_sub.render(lang)
            return re.sub(r'(?<!\\)\?', replacement, arg_to_sub)

        inner_replaced = re.sub(r"(\d+)\[\?←(.+?)\]", replace_match, template)
        
        # Render the remaining children before formatting
        rendered_args = [arg.render(lang) if not isinstance(arg, str) else arg for arg in a]
        output = wrap(inner_replaced).format(*rendered_args)

        # Convert escaped "\?" back to "?" after processing
        return output.replace(r'\?', '?')

    # The original Substitution returned a function that expected rendered strings.
    # The fix here is to make it robust enough to handle node objects too.
    def sub(*a, **ka):
        return replace_template(template, a)

    return sub


class FastProduction:
    """A unified, high-performance production node for all generation modes."""
    __slots__ = ('rule', 'type', 'parent', 'children', 'local_state', 'cache', '_depth', 'step', 'save', '_height')

    def __init__(self, rule=None, type=None, parent=None, children=None, state=None):
        self.rule, self.type, self.parent, self.cache = rule or '', type or (rule.name if rule else ''), parent, {}
        self.children = children if children is not None else []
        self._depth = (parent.depth + 1 if parent else 0)
        self._height = -1
        self.local_state = state or {}

        if self.rule and not children:
            if self.rule.state:
                self.local_state = {**self.rule.state, **self.local_state}
            self.children = [FastProduction(type=arg, parent=self) for arg in self.rule.args]
            
    @property
    def state(self):
        if self.parent:
            return ChainMap(self.local_state, self.parent.state)
        return self.local_state

    def clone(self):
        new_root = FastProduction(rule=self.rule, state=self.local_state.copy())
        if hasattr(self, 'step'): new_root.step = self.step
        if hasattr(self, 'save'): new_root.save = self.save
        q_orig, q_clone = deque([self]), deque([new_root])
        while q_orig:
            orig_node, clone_node = q_orig.popleft(), q_clone.popleft()
            clone_node.children = [FastProduction(rule=c.rule, type=c.type, parent=clone_node, state=c.local_state.copy()) for c in orig_node.children]
            q_orig.extend(orig_node.children); q_clone.extend(clone_node.children)
        return new_root
    
    def _find_first_unexpanded_leaf(self):
        q = deque([self])
        while q:
            node = q.popleft()
            if not node.rule:
                return node
            q.extendleft(reversed(node.children))
        return None

    def update_height(self):
        if self._height != -1: return self._height
        self._height = 0 if not self.children else 1 + max((c.update_height() for c in self.children), default=-1)
        return self._height

    @property
    def height(self):
        if self._height == -1: self.update_height()
        return self._height
        
    @property
    def depth(self): return self._depth
    @property
    def leaves(self):
        if not self.children: return [self]
        leaves_list, q = [], deque([self])
        while q:
            node = q.popleft()
            if not node.children: leaves_list.append(node)
            else: q.extendleft(reversed(node.children))
        return leaves_list
    @property
    def ancestors(self):
        anc, node = [], self.parent
        while node: anc.append(node); node = node.parent
        return tuple(anc)
    @property
    def siblings(self):
        return tuple(c for c in self.parent.children if c is not self) if self.parent else tuple()
    @property
    def descendants(self):
        if not self.children: return tuple()
        desc, q = [], deque(self.children)
        while q:
            node = q.popleft(); desc.append(node); q.extend(node.children)
        return tuple(desc)

    def check(self, mode='args'):
        if not self.rule: return True
        if mode == 'args':
            if not all(x.rule for x in self.children): return True
            return all(c(self.children) for c in self.rule.constraint)
        if mode == 'state':
            # A state constraint function `c` receives a node `x`,
            # and can now correctly access `x.state`.
            return all(c(x) for x in [self] + list(self.ancestors) for c in x.rule.state_constraint)
        return True

    def render(self, lang=None):
        if lang in self.cache: return self.cache[lang]
        if not self.rule: return "#" + self.type
        template = self.rule.templates.get(lang)
        if template is None: return "#" + self.type
        
        if callable(template):
            # A callable template receives children as *args. If that template
            # function needs state, it can now correctly access it via node.state
            # (e.g., args[0].parent.state).
            args, template_func = self.children, template
        else: # String template
            # For string templates, we decide which function to use based on content
            if '?←' in template:
                # The Substitution function needs the actual child nodes, not their rendered strings
                args = self.children
                # In the user's original setup, Substitution is imported from grammar.py
                # This call assumes it is available in the scope.
                template_func = Substitution(template, lang)
            else:
                # For regular .format() templates, we render the children first
                args = [c.render(lang) for c in self.children]
                template_func = template.format

        out = template_func(*args)
        if "#" not in out: self.cache[lang] = out
        return out

    def __getitem__(self, key): return self.children[key]
    __matmul__ = render

    def __repr__(self):
        if self.rule: return f"PROD:{self.type}({', '.join(self.rule.args)})"
        else: return f"PROD:{self.type}"
    

    def dict(self, use_cls=False):
        res = edict({l: self@l for l in self.rule.langs})
        if use_cls: res.cls = self
        return res


    def to_anytree(self):
        memo = {}
        def _convert_node(node):
            if id(node) in memo: return memo[id(node)]
            new_node = anytree.Node(node.type)
            new_node.rule = node.rule
            # Convert the ChainMap view to a regular dict for export
            new_node.state = dict(node.state) 
            for child in node.children:
                _convert_node(child).parent = new_node
            memo[id(node)] = new_node
            return new_node
        return _convert_node(self)

@lru_cache(maxsize=None)
def _precompute_height_bounds(Rule):
    non_terminals = list({r.name for r in Rule._instances})
    min_h, max_h = {nt: 999 for nt in non_terminals}, {nt: 0 for nt in non_terminals}
    for r in Rule._instances:
        if not r.args: min_h[r.name], max_h[r.name] = 0, 0
    for _ in range(len(non_terminals) + 1):
        for r in Rule._instances:
            if not r.args: continue
            child_mins = [min_h.get(arg, 999) for arg in r.args]
            child_maxs = [max_h.get(arg, 0) for arg in r.args]
            if all(h != 999 for h in child_mins): min_h[r.name] = min(min_h[r.name], 1 + max(child_mins))
            if all(h != 0 or max_h.get(arg, -1) != -1 for arg, h in zip(r.args, child_maxs)):
                max_h[r.name] = max(max_h[r.name], 1 + max(child_maxs))
    min_h = {k: v for k, v in min_h.items() if v != 999}
    return min_h, max_h


def generate_sequential(start, k=1, max_depth=12, min_depth=None, bushiness=1.0,
                        skip_check=False, max_steps=5000, save_prob=0.0125, **kwargs):
    """
    A high-yield, depth-first, stateful generator with relaxed topological control.

    Args:
        start: The starting rule object.
        k (int): The branching factor for exploration.
        max_depth (int): The absolute maximum depth for any leaf.
        min_depth (int): The minimum depth the *deepest* leaf in the tree must have.
        bushiness (float): A "pickiness" parameter from 0.0 to 1.0 controlling the
            trade-off between generation speed and topological strictness.
            - 1.0 ("Picky/Bushy"): Aggressively prunes branches that can't reach
              min_depth. This is slower but generates dense trees where all leaves
              are deep.
            - 0.0 ("Relaxed/Viney"): Never prunes short branches, maximizing generation
              speed and yield. Relies on the final check to ensure depth.
        skip_check (bool): If True, skips constraint checks for performance.
        max_steps (int): Safety limit to prevent infinite generation.
        save_prob (float): Probability of saving a state for backtracking.
    """
    min_depth = min_depth or 0
    if min_depth > max_depth: return []
        
    Rule = type(start)
    min_heights, max_heights = _precompute_height_bounds(Rule)

    def save(p, s, st): ckpt = p.clone(); ckpt.step, ckpt.save = st, 1; s.insert(([0] + [i for i, x in enumerate(s) if hasattr(x, 'save') and x.save])[-1], ckpt)
    def s_choices(seq, w, k):
        if not seq: return []
        if w and sum(w) > 0: return random.choices(seq, weights=w, k=k)
        return random.sample(seq, min(k, len(seq)))

    start_prod = FastProduction(start)
    start_prod.step, start_prod.save = 0, 0
    stack, step = [start_prod], 0
    while stack:
        step += 1
        if step > max_steps: return []
        prod = stack.pop()
        
        lv = prod._find_first_unexpanded_leaf()

        if not lv:
            all_leaf_depths = [leaf.depth for leaf in prod.leaves]
            deepest_leaf_depth = max(all_leaf_depths) if all_leaf_depths else 0
            
            if min_depth <= deepest_leaf_depth <= max_depth and \
               (skip_check or all(x.check() for x in [prod] + list(prod.descendants))):
                return [prod]
            continue
            
        potential_rules = Rule.get_rules(lv.type, shuffle=True)
        rules = []
        for r in potential_rules:
            min_h_rule = 0 if not r.args else 1 + max((min_heights.get(a, max_depth + 1) for a in r.args), default=0)
            max_h_rule = 0 if not r.args else 1 + max((max_heights.get(a, 0) for a in r.args), default=0)

            # --- HIGH-YIELD PRUNING LOGIC ---
            
            # 1. Absolute Safety Check: Prune if this rule *guarantees* exceeding max_depth.
            # This is the only non-negotiable hard constraint during generation.
            if lv.depth + min_h_rule > max_depth:
                continue

            # 2. Topological Heuristic: Check if this rule is "too short" to reach min_depth.
            is_too_short = (lv.depth + max_h_rule < min_depth)

            # 3. Probabilistic Pruning: If the rule is too short, we become "picky" based on
            # bushiness. High bushiness makes us likely to prune. Low bushiness makes us
            # likely to accept it, prioritizing yield over strict topology.
            if is_too_short and (random.random() < bushiness):
                continue
            
            rules.append(r)
        
        if not rules: continue
        
        rules_to_try = s_choices(rules, [r.weight for r in rules], k)
        if len(rules_to_try) > 1 and random.random() < save_prob: save(prod, stack, step)
        
        for i, rule in enumerate(rules_to_try):
            branch = prod if i == len(rules_to_try) - 1 else prod.clone()
            
            path, curr = [], lv
            while curr.parent: path.append(curr.parent.children.index(curr)); curr = curr.parent
            target_lv = branch
            for idx in reversed(path): target_lv = target_lv.children[idx]

            target_lv.rule = rule
            if rule.state:
                target_lv.local_state.update(rule.state)
            target_lv.children = [FastProduction(type=c, parent=target_lv) for c in rule.args]

            if not skip_check and (not target_lv.check('state') or (all(s.rule for s in target_lv.siblings) and not target_lv.parent.check('args'))):
                continue
            
            stack.append(branch)
            break
            
    return []