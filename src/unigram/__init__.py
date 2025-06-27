import os, sys, re, random
import pandas as pd
from dataclasses import dataclass
import itertools
from functools import lru_cache, partial
from tqdm.auto import tqdm
from typing import Any
from anytree import Node, NodeMixin, LightNodeMixin, RenderTree
import psutil
from tqdm.auto import tqdm
import copy
from collections import defaultdict
import numpy as np
from easydict import EasyDict as edict
from functools import wraps
import time

class FlatList(list):
    def __iadd__(self, other):
        if isinstance(other, list):
            self.extend(other)
        else:
            self.append(other)
        return self

    def __add__(self, other):
        result = FlatList(self)
        if isinstance(other, list):
            result.extend(other)
        else:
            result.append(other)
        return result


def Constraint(constraint_str):
    def generated_function(x):
        conditions = constraint_str.split(',')
        for cond in conditions:
            i, j = map(int, cond.split('∉'))
            if x[i].render('eng') in x[j].render('eng'):
                return False
        return True
    return generated_function

def apply_to_all_args(f):
    def decorator(func):
        def wrapper(*args, **kwargs):
            new_args = [f(arg) for arg in args]
            return func(*new_args, **kwargs)
        return wrapper
    return decorator

def Substitution(template,lang=None):
    def replace_template(template, a):
        # make numbers formattable 0 -> {0}
        wrap =  lambda s: (re.sub(r'(\d+)', r'{\1}', s) if type(s)==str else s)
        inner_replaced= re.sub(r"(\d+)\[\?←(.+?)\]",
            lambda m: a[int(m.group(1))].replace("?", (m.group(2))),
            template)
        output=wrap(inner_replaced).format(*a)
        return output

    """i[?←expr] replaces ? (but not \?) in slot i with expr"""
    @apply_to_all_args(lambda x:x.render(lang) if type(x)!=str else x)
    def sub(*a,**ka):
        y = replace_template(template,a)
        return y
    return sub



def Substitution(template, lang=None):
    def replace_template(template, a):
        # Make numbers formattable 0 -> {0}
        wrap = lambda s: (re.sub(r'(\d+)', r'{\1}', s) if isinstance(s, str) else s)

        # Function to safely replace only unescaped '?'
        def replace_match(m):
            slot_idx = int(m.group(1))
            replacement = m.group(2)
            return re.sub(r'(?<!\\)\?', replacement, a[slot_idx])  # Only replace unescaped '?'

        inner_replaced = re.sub(r"(\d+)\[\?←(.+?)\]", replace_match, template)
        output = wrap(inner_replaced).format(*a)

        # Convert escaped "\?" back to "?" after processing
        return output.replace(r'\?', '?')

    """i[?←expr] replaces ? (but not \?) in slot i with expr"""
    @apply_to_all_args(lambda x: x.render(lang) if not isinstance(x, str) else x)
    def sub(*a, **ka):
        return replace_template(template, a)

    return sub


# Pre-compile regex patterns
NUMBER_PATTERN = re.compile(r'(\d+)')
SUBSTITUTION_PATTERN = re.compile(r"(\d+)\[\?←(.+?)\]")


default_preprocess_template = lambda s: (re.sub(r'(\d+)', r'{\1}', s) if type(s)==str and '←' not in s else s)

def init_grammar(langs, name='', preprocess_template=default_preprocess_template):
    class Rule:
        _instances = []
    
        @classmethod
        def init(cls, langs, name='', preprocess_template=preprocess_template):
            Rule.langs = langs
            Rule._instances = []
            Rule.preprocess_template = preprocess_template
            return langs
    
        def __init__(self, signature, *args, constraint=[], state_constraint=[], vars=dict(), weight=1):
            self.signature=signature
            self.name, self.args = self.parse_signature(signature)
            self.constraint = FlatList() + constraint
            self.state_constraint = FlatList() + state_constraint
            self.templates = {}
            self.weight=weight
            self.state = vars
            for lang, template in zip(Rule.langs, args*2):
                self.templates[lang] = Rule.preprocess_template(template)
            self.index = len(Rule._instances)
            Rule._instances.append(self)
    
        #def __getattr__(self,lang):
        #    return self.templates[lang]
    
        def parse_signature(self, s):
            name, args = re.match(r'(\w+)(?:\((.*?)\))?', s).groups()
            args = args.split(',') if args else []
            args = [a.strip() for a in args]
            return name, args
    
        @classmethod
        def get_rules(cls, name, sort=False, terminals=False,shuffle=False):
            instances = [x for x in cls._instances if x.name == name]
            if terminals:
                instances = [x for x in instances if not x.args]
            if shuffle:
                random.shuffle(instances)
            return instances
    
        @classmethod
        def start(cls):
            starts = cls.get_rules('start')
            if starts:
                return starts[0]
            else:
                return cls._instances[0]
    
        def __repr__(self):
            return f"RULE:{self.name}{self.args}"



        def __reduce__(self):
            def rebuild_rule(signature, constraint, state_constraint, state, weight, templates):
                # Logic to recreate the rule
                rule = Rule(signature, constraint=constraint, state_constraint=state_constraint, 
                            vars=state, weight=weight)
                rule.templates = templates
                return rule
                    
            return (rebuild_rule, (self.signature, self.constraint, self.state_constraint, 
                                  self.state, self.weight, self.templates))
                    
    R = Rule
    R.init(langs,name)

    classname = f'Rule_{id(object())}'
    cls_dict = dict(Rule.__dict__)
    cls_dict.pop('__dict__', None)
    cls_dict.pop('__weakref__', None)
    NewRule = type(classname, (object,), cls_dict)
    globals()[classname] = NewRule
    module = sys.modules[__name__]
    setattr(module, classname, NewRule)
    return NewRule
    
Rule = init_grammar(None)

class Production(NodeMixin):
    def __init__(self, rule=None,type=None,state=dict(), parent=None):
        self.rule = rule or ''
        self.type= type or self.rule.name
        self.state = {"parents": [],**state}

        if parent:
            self.parent = parent
            
        if rule: 
            self.children = [Production(type=element) for element in self.rule.args]
            self.state = {**self.rule.state}
        self.cache=dict()

    
    def __setitem__(self, key, value):
        setattr(self,key,value)

    def distinct_constraint(self,x):
        self.pairs = itertools.combinations([idx for idx, a in enumerate(self.children) if self.children.count(a) > 1], 2)
        result= all(x[i] != x[j] for i, j in self.pairs)
        return result

    def __eq__(self,other):
        lang=self.rule.langs[0]
        return self@lang == other@lang

    def __getitem__(self,key):
        return self.children[key]

    def __len__(self):
        return len(self.children)

    def __deepcopy__(self, memo):
        new_obj = Production.__new__(Production)
        memo[id(self)] = new_obj
        new_obj.rule = self.rule
        new_obj.type = self.type
        new_obj.state = copy.deepcopy(self.state)
        new_obj.parent = None
        new_obj.cache = self.cache
        new_obj.children = [copy.deepcopy(child, memo) for child in self.children]
        return new_obj

    def check(self,mode='args'):
        if mode=='args':
            arguments=self.children
            if all(x.rule for x in arguments):
                return all(constraint(arguments) for constraint in self.rule.constraint+self.distinct_constraint)
            else:
                return True
        if mode=='state':
            return all(constraint(x) for x in [self,*self.ancestors] for constraint in x.rule.state_constraint)
            #constraints = [x.rule.state_constraint for x in [self,*self.ancestors]]
            #return all([constraint(self) for constraint in constraints])

    def render(self, lang=None):
        if lang==None:
            return str(RenderTree(self))
        if lang in self.cache:
            return self.cache[lang]
        try:
            template = self.rule.templates[lang]
        except:
            return "#"+self.type
        args = self.children
        if isinstance(template, str):
            args = [x.render(lang) for x in args]
            if '?←' in template:
                template = Substitution(template,lang)
            else:
                template_str = template
                template = template.format
        out = template(*args)
        if "#" not in out:
            self.cache[lang]=out
        return out
        
    __matmul__ = render


    def dict(self,use_cls=False):
        return edict({l:self@l for l in self.rule.langs}|(dict(cls=self) if use_cls else dict()))

    def __repr__(self):
        return f"PROD:{self.type}"+ (str(self.rule.args) if self.rule else '')


def safe_choices(sequence, weights=None, k=1):
    if not sequence:
        return sequence
    if weights is not None:
        weights = np.array(weights).flatten() / np.sum(weights)
        k = np.count_nonzero(weights)
    k = min(len(sequence), k)
    if weights is None or len(set(weights))==1:
        return random.sample(sequence, k)
    return np.random.choice(sequence, size=k, replace=False, p=weights)
    
def save(production,stack,step):
    ckpt=copy.deepcopy(production)
    ckpt.step=step
    ckpt.save=1
    ckpt_indices=[i for (i,x) in enumerate(stack) if x.save]
    #stack.insert(random.choice(ckpt_indices or [0]), ckpt)
    stack.insert(([0]+ckpt_indices)[-1], ckpt)

def generate_sequential(start,k=8,depth=14, skip_check=False,max_steps=1200,max_concentration_rate=0.8):
    start=Production(start)
    start.step=step=0
    start.save=0
    stack = [start]
    while stack:
        step+=1
        if step>max_steps:
            return []
        production = stack.pop()
        leaves=production.leaves
        if all(lv.rule for lv in leaves):
            if skip_check or all(x.check(mode='args') for x in (production,)+production.descendants):
                production.step=step
                return [production]
            continue
        lv = [lv for lv in leaves if not lv.rule][0]
        
        rules = start.rule.get_rules(lv.type,terminals=(lv.depth==depth),shuffle=False)
        if not rules:
            step+=5
            continue
        weights = [r.weight for r in rules]
        rules = safe_choices(rules,k=k, weights=weights)
        if len(rules)>1 and random.random()<(1/8):
            save(production,stack,step)
        for rule in rules:
            step+=0.25
            if step>max_steps:
                return []
            lv.rule=rule
            state={**rule.state,**lv.state}
            lv.children = [Production(type=c,state=state) for c in rule.args]
            if not lv.check(mode='state') and not skip_check:
                continue
            if all(x.rule for x in lv.siblings) and not lv.parent.check(mode='args') and not skip_check:
                continue #backtrack, chose another rule
            stack.append(production)
            break
    return []

def view(a,*la,**kwa):
    verbose=False
    if verbose:
        if type(a)==Production:
            try:
                if verbose>2:
                    clear_output()
                    print(a@eng,*la,**kwa)
                    print()
            except:
                pass
        else:
            print(a,*la,**kwa)



def generate_recursive(start, depth=14, max_steps=1200, production_class=Production):
    start_prod = production_class(start)
    
    def fill_tree(node, depth_budget=depth):
        empty_leaves = [lv for lv in node.leaves if not lv.rule]
        if not empty_leaves: return start_prod
        
        for lv in empty_leaves:
            # Use terminals when depth budget is exhausted
            use_terminals = (depth_budget <= 0)
            rules = node.rule.get_rules(lv.type, terminals=use_terminals, shuffle=True)
            if not rules: return None
            
            weights = [r.weight for r in rules]
            chosen_rule = random.choices(rules, weights=weights, k=1)[0]
            
            lv.rule = chosen_rule
            # Pass the reduced depth budget to children
            lv.children = [production_class(type=c, state={**chosen_rule.state, **lv.state}) 
                          for c in chosen_rule.args]
        
        # Reduce depth budget for the next recursive call
        return fill_tree(node, depth_budget - 1)
    
    result = fill_tree(start_prod, depth)
    return [result] if result else []

def generate_recursive_light(*args, **kwargs):
    return generate_recursive(*args, production_class=LightProduction, **kwargs)

def generate(start, n_iter=10_000, mode='recursive', seed=None, *args, **kwargs):
    random.seed(seed)
    if type(start)==type:
        start=start.start()

    modes = {'recursive': generate_recursive,
             'recursive_light': generate_recursive_light,
             'sequential': generate_sequential
            }
    generate = modes[mode]
    
    """Generate one production using specified mode."""
    for _ in range(n_iter):
        result = generate(start, *args, **kwargs)
        if result: return result[0]
    raise ValueError('Incomplete generation')



def G0():
    """Toy grammar for illustration"""
    R = init_grammar(['tptp','eng'])
    def render_branch(x):
        #print(x.root, x.root.leaves, x.root.descendants, x.leftsbling)
        """equivalent to '0', illustrate function-based rendering"""
        return x@"eng"

    R('start(branch)', render_branch)
    R('branch(formula)', 'A:0')
    R('branch(formula)', 'B:0')
    R('branch(formula)', 'C:0')
    R('formula(formula)', '~(0)', 'It is not the case that "0"')
    R('formula(atom,op,atom)', '0 1 2', '0 1 2')
    R('formula(atom)', '0')
    R('atom', 'p', 'it is daytime')
    R('atom', 'q', 'it is raining')
    R('op', '&', 'and')
    R('op', '|', 'or')
    return R


def LogicNLI():
    ADJECTIVES = ['rich', 'quiet', 'old', 'tall', 'kind', 'brave', 'wise',
                  'happy', 'strong', 'curious', 'patient', 'funny', 'generous', 'humble']
    # (We selected adjectives with no clear semantic interference)
    NAMES = ['mary', 'paul', 'fred', 'alice', 'john', 'susan', 'lucy']

    R = init_grammar(['tptp','eng'])
    R('start(' + ','.join(['rule']*16) + ',' + ','.join(['fact']*8) + ')',
      '&\n'.join([f'({i})' for i in range(24)]),
      '\n'.join([f'{i}' for i in range(24)]))

    R('hypothesis(person,a)', '1(0)', '0 is 1')
    for a in ADJECTIVES:
        R('adj', a)
        R('adj', f'~{a}', f'not {a}', weight=0.2)

    R('property(adj,adj)', '(0(?)&1(?))', 'both 0 and 1')
    R('property(adj,adj)', '(0(?)|1(?))', '0 or 1')
    R('property(adj,adj)', '(0(?)<~>1(?))', 'either 0 or 1', weight=0.5)
    R('property(adj)', '0(?)', '0')

    R('rule(property,property)', '![X]:(0[?←X]=>1[?←X])',
      'everyone who is 0 is 1')
    R('rule(property,property)', '![X]:(0[?←X]<=>1[?←X])',
      'everyone who is 0 is 1 and vice versa')

    for p in NAMES:
        R('person', p)

    R('fact(person,property)', '1[?←0]', '0 is 1')
    R('fact(property)', '?[X]:(0[?←X])', 'someone is 0', weight=0.2)
    R('rule(fact,fact)', '(0)=>(1)', 'if 0 then 1')
    R('rule(fact,fact)', '(0)<=>(1)', 'if 0 then 1 and vice versa')
    return R


def test_grammar(grammar=G0()):
    "Illustrate the generation process"
    production = generate(grammar)
    return production, production@"eng"




class LightProduction:
    """A lightweight alternative to the 'Production' class, without anytree overhead."""
    def __init__(self, rule=None, type=None, state=dict(), parent=None):
        self.rule = rule or ''
        self.type = type or self.rule.name
        self.parent = parent
        self.state = {"parents": [], **state}
        self.cache = dict()
        self.children = []
        if rule:
            self.state = {**self.rule.state, **state}
            # Manually create children and set their parent to self
            self.children = [LightProduction(type=element, state=self.state, parent=self) for element in self.rule.args]

    def get_ancestors(self):
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

    def check(self, mode='args'):
        """Re-implementation of the check logic without anytree."""
        if not self.rule: return True
        
        if mode == 'args':
            arguments = self.children
            if not all(x.rule for x in arguments):
                return True
            # The distinct_constraint is not used in the original code, but kept for parity.
            # You would need to implement it manually if needed.
            return all(constraint(arguments) for constraint in self.rule.constraint)
        
        if mode == 'state':
            nodes_to_check = [self] + self.get_ancestors()
            return all(constraint(x) 
                       for x in nodes_to_check 
                       for constraint in x.rule.state_constraint)
        return True

    def render(self, lang=None):
        if lang is None:
            # We can't use RenderTree, so we return a simplified representation.
            return self.__repr__()
            
        if lang in self.cache:
            return self.cache[lang]
        
        try:
            template = self.rule.templates[lang]
        except (AttributeError, KeyError):
            return "#" + self.type

        # Recursively render children
        args = [child.render(lang) for child in self.children]

        # Template application logic is identical to Production
        if isinstance(template, str):
            if '?←' in template:
                # Assuming Substitution function is globally available
                template_func = Substitution(template, lang)
            else:
                template_func = template.format
        else: # It's a function
            template_func = template
            
        out = template_func(*args)
        
        if "#" not in out:
            self.cache[lang] = out
        return out

    # Make it compatible with the @ operator
    __matmul__ = render

    def __repr__(self):
        return f"LightPROD:{self.type}" + (str(self.rule.args) if self.rule else '')