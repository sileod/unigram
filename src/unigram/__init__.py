import os, sys, re, random
import pandas as pd
from dataclasses import dataclass
import itertools
from functools import lru_cache, partial
from tqdm.auto import tqdm
from typing import Any
from anytree import Node, NodeMixin, LightNodeMixin, RenderTree
#from xpflow import FlatList
import psutil
from contextlib import contextmanager
from tqdm.auto import tqdm
import copy
from collections import defaultdict
import random
import numpy as np
from easydict import EasyDict as edict

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

    """i[?←expr] replaces ? in slot i with expr"""
    @apply_to_all_args(lambda x:x.render(lang) if type(x)!=str else x)
    def sub(*a,**ka):
            return replace_template(template,a)
    return sub


default_preprocess_template = lambda s: (re.sub(r'(\d+)', r'{\1}', s) if type(s)==str and '←' not in s else s)


class Rule:
    _instances = []

    @classmethod
    def init(cls, langs, setup, preprocess_template=default_preprocess_template):
        Rule.langs = langs
        Rule._instances = []
        Rule.preprocess_template = preprocess_template

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
        return cls.get_rules('start')[0]

    def __repr__(self):
        return f"RULE:{self.name}{self.args}"


class Production(NodeMixin):
    def __init__(self, rule=None,type=None,state=dict()):
        self.rule = rule or ''
        self.type= type or self.rule.name
        self.state = {"parents": [],**state}
        if rule: 
            self.children = [Production(type=element) for element in self.rule.args]
            self.state = {**self.rule.state}

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
                template = template.format
        return template(*args)
    __matmul__ = render


    def dict(self):
        return edict({l:self@l for l in self.rule.langs}|dict(cls=self))

    def __repr__(self):
        return f"PROD:{self.type}"+ (str(self.rule.args) if self.rule else '')


def safe_choices(sequence, weights=None, k=1):
    if not sequence:
        return sequence
    if weights is not None:
        weights = np.array(weights).flatten() / np.sum(weights)
        k = np.count_nonzero(weights)
    k = min(len(sequence), k)
    return np.random.choice(sequence, size=k, replace=False, p=weights)
    
def save(production,stack,step):
    ckpt=copy.deepcopy(production)
    ckpt.step=step
    ckpt.save=1
    ckpt_indices=[i for (i,x) in enumerate(stack) if x.save]
    #stack.insert(random.choice(ckpt_indices or [0]), ckpt)
    stack.insert(([0]+ckpt_indices)[-1], ckpt)

def generate(start,k=8,depth=14, skip_check=False,max_steps=1200,max_concentration_rate=0.8):
    start=Production(start)
    start.step=step=0
    start.save=0
    stack = [start]
    while stack:
        #print(len(stack),'\t',step,end='\r')
        step+=1
        if step>max_steps:
            return []
        production = stack.pop()
        leaves=production.leaves
        if all(lv.rule for lv in leaves):
            if skip_check or all(x.check(mode='args') for x in production.descendants):
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
            if not lv.check(mode='state'):
                continue
            if all(x.rule for x in lv.siblings) and not lv.parent.check(mode='args'):
                continue #backtrack, chose another rule
            stack.append(production)
            break
    return []


def R0():
    
    R=Rule
    R.init(['tptp','eng'], "fof")
    
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
    
    R.init(['tptp', 'eng'], "fof")
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
