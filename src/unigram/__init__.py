import os, sys, re, random
import pandas as pd
from dataclasses import dataclass
import copy
import itertools
from functools import lru_cache, partial
from tqdm.auto import tqdm
import inspect
import math, types
import funcy as fc
import networkx as nx
from anytree import Node, NodeMixin
from typing import Any


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

def sample_given_size(population, population_size, sample_size):
    for item in population:
        if random.random() < 0.99 * sample_size / population_size:
            yield item
            sample_size -= 1
        population_size -= 1

def replace_template(template, a):
    # make numbers formattable 0 -> {0}
    wrap =  lambda s: (re.sub(r'(\d)', r'{\1}', s) if type(s)==str else s)
    inner_replaced= re.sub(r"(\d)\[\?←(.+?)\]",
        lambda m: a[int(m.group(1))].replace("?", (m.group(2))),
        template)
    output=wrap(inner_replaced).format(*a)
    return output
def Substitution(template):
    """i[?←expr] replaces ? in slot i with expr"""
    @apply_to_all_args(lambda x:x.render())
    def sub(*a,**ka):
            return replace_template(template,a)
    return sub

@dataclass(slots=True)
class Slot:
    type:Any
    value:Any=None

cache={}
class Rule(NodeMixin):
    _instances = []
    _productions = []
    counter=0
    #renderer="tptp"

    def parse_signature(self, s):
        components = s.split(';')
        s = components[0]
        self.output_type = components[1] if len(components) > 1 else None
        name, args = re.match(r'(\w+)(?:\((.*?)\))?', s).groups()
        args=args.split(',') if args else []
        return name, args

    def distinct_constraint(self,x):
        self.pairs = itertools.combinations([idx for idx, a in enumerate(self.args) if self.args.count(a) > 1], 2)
        return all(x[i] != x[j] for i, j in self.pairs)

    def __init__(self, name, *args,
                 constraint=lambda x:True,
                 state_constraint=[],
                 tags=[],vars=dict()):

        wrap =  lambda s: (re.sub(r'(\d)', r'{\1}', s) if type(s)==str and '←' not in s else s)
        for k,v in zip(self.langs,args+(None,)*len(self.langs)):
            if v==None:
                v=prev_v
            setattr(self,k,wrap(v))
            prev_v=v
        self.signature=name
        self.max_depth=0
        self.background=False
        self.name, self.args = self.parse_signature(name)
        self.constraint = FlatList()+[self.distinct_constraint] +constraint
        self.state_constraint=FlatList()+state_constraint
        self.tags=tags
        self.parent=None
        self.verbose=True
        self.state={"parents":[],**vars}
        self.filled_args = [Slot(type=element) for element in self.args]
        self.renderer=self.langs[0]
        self.rules_filter = lambda x:True
        Rule._instances.append(self)

    def __repr__(self):
        return self.signature

    def __getitem__(self, key):
        return self.filled_args[key].value
        
    def __setitem__(self,key,value):
        self.filled_args[key].value=value

    def __len__(self):
        return len(self.filled_args)
        
    def __iter__(self):
        for x in self.filled_args:
            yield x.value

    def __str__(self):
        args=[x.value for x in self.filled_args]
        for x in args:
            x.renderer=self.renderer
        renderer = getattr(self,self.renderer)
        if type(renderer)==str:
            if '?←' in renderer:
                renderer = Substitution(renderer)
            else:
                renderer=renderer.format
        return renderer(*args)
            
    def __eq__(self,other):
        return str(self)==str(other)

    def __hash__(self):
        return hash(str(self))

    def __matmul__(self,lang):
        return self.render(lang)

    def __deepcopy__(self, memo):
        new_obj = copy.copy(self)
        new_obj.parent=copy.copy(self.parent)
        new_obj.filled_args = [Slot(type=element) for element in self.args]
        return new_obj
        
        
    @staticmethod
    def get_cycles(R):
        G = nx.DiGraph()
        for _, row in R.rules().iterrows():
            G.add_node(row.cls.signature)
        for _, row in R.rules().iterrows():
            for arg in row.args:
                for fill in R.get_instances(arg):
                    G.add_edge(row.cls.signature, fill.signature)
        return pd.value_counts(list(fc.flatten(nx.recursive_simple_cycles(G)))).to_dict()
      
    def render(self,format=None):
        if format:
            self.renderer=format
        return str(self)#.strip()

    def parent_constraints(self,l=FlatList()):
        if self.parent: return self.parent.parent_constraints(l+self.state_constraint)
        else: return l+self.state_constraint
        
    def check(self,mode='args'):
        if mode=='args':
            args=[x.value for x in self.filled_args]
            return all([constraint(args) for constraint in self.constraint])
        else:
            return all([constraint(self) for constraint in self.parent_constraints()])
            
    @classmethod
    def get_instances(cls,name,sort=False, terminals=False):
        instances = [x for x in cls._instances if x.name==name]
        instances = [x for x in instances if cls.rules_filter(x)]
        if terminals:
            instances = [x for x in instances if not x.args]
        if sort:
            instances=sorted(instances, key=lambda x: cls.cycles.get(x.signature,0))
        return copy.deepcopy(instances)

    def get_children(self,*args,**kwargs):

        parents=dict(parents=self.state.get('parents',[])+[self])
        children=self.get_instances(*args,**kwargs)
        output=[]
        for x in children:
            x.parent=copy.deepcopy(self) ###############
            x.state={**x.state,**self.state,**parents}
            if x.check(mode="state"):
                output+=[x]
        return output        
        
    @classmethod
    def start(cls):
        return cls.get_instances('start')[0]

    @classmethod
    def rules(cls,renderer="signature"):
        Rule.renderer=renderer
        df= pd.DataFrame([{c:getattr(x,c) for c in ('name','args')} for x in cls._instances])
        df['cls']=cls._instances
        df= copy.deepcopy(df)
        df.__hash__ = lambda df: str(df.signature)
        return df
        
    @classmethod
    def init(cls,langs,setup):
        Rule.langs=langs
        Rule._instances=[]
        Rule._productions=[]
        Rule.setup=setup

    def fill_args(self,args):
        for slot, value in zip(self.filled_args,args):
            slot.value=value
        self.children=args

    def slot_fillings(self,slot_type,depth,breadth):
        if depth<1:
            children = self.get_children(slot_type,sort=True,terminals=True)
            output = [x for x in children]
        else:
            children = self.get_children(slot_type,sort=True)
            if not children:
                return []
            breadth=math.ceil(breadth/len(children))
            output = [x.fillings(depth-1,breadth) for x in children]
        output =  [item for sublist in output for item in sublist]
        return output
    
    def fillings(self,depth,breadth,verbose=False):
        make_self = lambda:copy.deepcopy(self)
        n_args = max(1,len(self.filled_args))
        sub_breadth = math.ceil(breadth**(1.0/n_args)) if breadth else breadth
        possibilities = [self.slot_fillings(slot.type,depth,sub_breadth) for slot in self.filled_args]
        if verbose:
            print(depth,[len(x) for x in possibilities],' '*10,end='\r')
        n_possibilites=math.prod([len(x) for x in possibilities])
        fillings=[]
        for filling in sample_given_size(itertools.product(*possibilities),n_possibilites,breadth):
            _self=make_self()
            _self.fill_args(filling)
            if _self.check():
                fillings+=[_self]
        if breadth and breadth<len(fillings):
            fillings=sorted(random.sample(fillings,int(breadth)),key=fillings.index)
        return fillings
        
    @classmethod
    def generate(self,start=None,depth=2,breadth=sys.maxsize,pandas=True,
            constraint=[],state_constraint=[],verbose=False,
            rules_filter=lambda x:True):
        Rule.rules_filter=rules_filter
        Rule.cycles=Rule.get_cycles(Rule)
        start = (self.start() if not start else start)
        start.constraint+=constraint
        start.state_constraint+=state_constraint
        generations=start.fillings(depth=depth,breadth=breadth,verbose=verbose)
        out=[{'cls':x,**{lang:x.render(lang) for lang in self.langs}} for x in generations]
        return pd.DataFrame(out).drop_duplicates(subset=self.langs)


def apply_to_all_args(f):
    def decorator(func):
        def wrapper(*args, **kwargs):
            new_args = [f(arg) for arg in args]
            return func(*new_args, **kwargs)
        return wrapper
    return decorator

        
def Constraint(constraint_str):
    def generated_function(x):
        conditions = constraint_str.split(',')
        for cond in conditions:
            i, j = map(int, cond.split('∉'))
            if x[i].render('eng') in x[j].render('eng'):
                return False
        return True
    return generated_function
