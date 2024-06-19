from unigram import Rule, Substitution, Constraint, Production
import random
from xpflow import FlatList
from unigram.solver_utils.tptp import split_clauses, run, to_tptp, extract_inferences_and_formulas
import subprocess, copy
import itertools, re, random
import funcy as fc
from itertools import combinations
from collections import defaultdict
import numpy as np
import pandas as pd
import anytree

R,S,C=Rule,Substitution,Constraint
    
def pluralize(s):
    s=s.replace('does','do')
    s=s.replace('is a','are')
    s=s.replace('is ','are ')
    s=s.replace('es ','e ')
    s=s.replace('person','persons')
    s=s.replace('european','europeans')
    s=s.replace('tourist','tourists')
    return s


def polysyllogism(x):
    # P1-P1 P2-P2 P3-P3 P4|P4
    if x.parent is not x.parent.parent.children[0]: # lhs
        return # must be lhs
    if x.parent.parent.type!="cproperty":
        return 
    for a in x.ancestors:
        if a.type=='term':
            ls = leftsibling(a)
            if not ls:
                return
            cp=[d for d in ls.descendants if d.type=='cproperty']
            if not cp:
                return 
            return cp[-1][-1]


def get_concepts(text):
    return " ".join(tuple(re.findall(r'[a-z]+', text)))
    
R=Rule
lang=eng,tptp=['eng','tptp']
R.init(['tptp','eng'], "fof") #R(signature, fof_tptp, pseudo-english)

R('start(setup)', '0')# (x@l).replace('¿','?') for l in lang])


def B(*args, **kwargs):
    r = Rule("background",*args,**kwargs)
    r.background=True
    return r

x_y = lambda x:f"![X,Y]:(X!=Y=>({x}))"
x_y_z = lambda x: f"![X,Y,Z]:((X!=Y & Y!=Z & X!=Z)=>({x}))"


B(x_y("like(X,Y)=>~hate(X,Y)"))
B(x_y("hate(X,Y)=>~like(X,Y)"))
B("![X]:(european(X)=>person(X))")
B("![X]:(tourist(X)=>person(X))")

B("![X]:(anywhere(X))")

B(x_y("sibling(X,Y)<=>sibling(Y,X)"))
R('predicate','sibling','is a sibling of')

# neither symmetric nor transitive
R('predicate','client','is a client of')


adjs=['rich','quiet','old','tall','kind','brave','wise','happy','creative',
      'strong','curious','patient','funny','generous','calm','humble']

for a in adjs[:4]:
    
    B(x_y_z(f"({a}er(X,Y)&{a}er(Y,Z))=>{a}er(X,Z)")) #transitivity
    B(x_y(f"({a}er(X,Y)=>~{a}er(Y,X))")) #asymmetry
    B(x_y(f"({a}er(X,Y)&{a}(X))=>{a}(Y)")) #instanciation
    B(x_y(f"(~{a}(X)&{a}(Y))=>{a}er(X,Y)")) #reverse instanciation

    R('adj_comp',a)


R('predicate(adj_comp)','0er','is 0er than',weight=2)


R('term(property_toward_inside,outside_person)', '0[?←1]','1 0',constraint=C('1∉0,0∉1'))
R('property_toward_inside(predicate,person)', '0(1,?)','0 1')
R('property(predicate,outside_person)', '0(1,?)','0 1',weight=3)


def render_setup(x):
    persons=x.state['persons']
    room=x.state['room']
    persons_str = ', '.join([p.title() for p in persons])
    eng_setup=f"{persons_str} are the only persons in the {room}.\n{x[0]@eng}"
    if len(persons)==1:
        eng_setup=eng_setup.replace(' are ',' is ').replace('persons','person')
    in_room="&".join([f"room({x})" for x in persons])
    disj="|".join([f"X='{x}'" for x in persons])
    tptp_setup=in_room + '&' + f"(![X]:(room(X)=>({disj})))&\n{x[0]@tptp}"
    return tptp_setup,eng_setup

room = 'room'
names = 'mary', 'paul', 'fred', 'alice', 'john', 'susan', 'lucy'
psums = [list(names[:i]) for i in range(1, 5 + 1)]

for persons in psums:
    R('setup(valid_block)', lambda x:render_setup(x)[0], lambda x:render_setup(x)[1],
        vars=dict(persons=persons,room=room))

def render_setup(x):
    return f"(there_is_a_room)&\n{x[0]@tptp}", f"there is a room.\n{x[0]@eng}"
for persons in [names[:4],[],[]]:
    R('setup(valid_block)', lambda x:rs(x)[0], lambda x:render_setup(x)[1],
        vars=dict(persons=[],room=room))
    
def check_state(x):
    parents_=" ".join([a.signature for a in x.parent])
    pattern = r"\([^)]*term[^)]*\)"
    term_depth = len(re.findall(pattern, parents_))
    return term_depth<9
 
def no_free_var(x):
    s=x[0]@tptp
    return 'X' not in s or set('?!')&set(s)
    
R('valid_block(block)','0', constraint=no_free_var)

def neg_constraint(x):
    for d in x.descendants:
        if type(d.rule)==str:
            continue
        tptp=str(d.rule.templates['tptp'])
        if "=>" in tptp or "<=" in tptp:
            return False
        eng=str(d.rule.templates['eng'])
        if 'who' in eng:
            return False
    return True

nesting_limit = lambda x: x[0].render('eng').count('it is ') <= 2
R("term(term)", "~(0)", "it is not the case that “0”", constraint=nesting_limit,
  state_constraint=neg_constraint,weight=1)

R("term(term)", "0", "it is true that “0”",constraint=nesting_limit,weight=0.25)
R("term(term)", "(mary_dream=>(0))", "Mary dreamt that “0”",constraint=nesting_limit,weight=0.15)

def render_coref(coref,debug=""):
    c_tp = fc.rcompose(coref, lambda x: x@tptp if x else None)
    c_en = fc.rcompose(coref, lambda x: debug+x@eng  if x else None)
    return c_tp, c_en

class SeedContext:
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        random.seed(None)  # Reset to a random state

R('state','')
R("block(term)", "0", "0",weight=2)

def block_cond(x):
    for d in x.descendants:
        if type(d.rule)!=str and d.rule.name in ['cproperty','term']:
            if '=' in str(d.rule.templates['tptp']):
                return False
            if 'who' in str(d.rule.templates['eng']):
                return False
    return True


R('nterm(term)','0', state_constraint=block_cond)

R("cond(nterm,term)", "(0)=>(1)", "if “0” then “1”", weight=4)
R("cond(nterm,nterm,term)", "((0)=>(1))&\n((1)=>(2))", "if “0” then “1”\nif “1” then “2”")
R("cond(nterm,term)", "(0)=>(1)", "“1” if “0”")
R("cond(nterm,nterm)", "(0)<=>(1)", "“1” if “0” and vice versa")
R("cond(nterm,term)", "(0)=>(1)", "“0” only if “1”")
R("cond(nterm,nterm)", "(0)<~>(1)", "either “0” or “1” but not both")
R("cond(term,nterm)", "~(1)=>(0)", "“0” unless “1”")
R('cond(nterm,term,term)', '((0)=>(1))&((~(0))=>(2))', 'if “0” then “1” otherwise “2”')
#R("block(term,term)", "(0)&\n(1)", "0\n1")

conj_tptp = lambda i: '&\n'.join([f'({i})' for i in range(i)])
conj_eng  = lambda i: '\n'.join([f'{i}' for i in range(i)])
conj = lambda i: (conj_tptp(i), conj_eng(i))

for n in list(range(3, 32)): 
    for nconds in range(int(n//2)):
        nterms = n - nconds  
        arg1 = "block(" + "cond,"*nconds + ("term,"*(nterms-1)).rstrip(',') + ")"
        R(arg1, *conj(n-1),weight=w)

for n in names:
    R("all_persons",n,n.title())
B('&'.join('({}!={})'.format(*a) for a in itertools.product(names,names) if len(set(a))!=1))

def inside_person(x):
    return x[0]@tptp in x[0].state['persons']
R('person(all_persons)','0','0',constraint=inside_person)

def all_unique(x):
    seen = set()
    return not any(i in seen or seen.add(i) for i in x)

def block_constraint(x):
    filled = [x for x in x.children if x.rule]
    return all_unique(x@eng for x in filled if "#" not in x@eng)

for i in 'abcdefghijklmnopqrstuvwxyz'
    R('atomic_predicate',f"pred_{i}")
R('property(atomic_predicate)', '0(?)','0',weight=6)
R('property(atomic_predicate)', '~0(?)','~0',weight=1)

R('term(person,property)', '1[?←0]',
    fc.rcompose(S('0 1', lang='eng'), lambda x: x.replace('they', 'he/she')),
    constraint=[C('0∉1,1∉0'), lambda x: 'who' not in x[1]@eng], weight=8)

R('term(person,property,property)','(1[?←0])&(2[?←0])','0 who 1 2',weight=1)

R('term(property,person,person)', '0[?←1]&0[?←2]', (S('1 and 2 0',lang='eng')|pluralize),
    constraint=[
        C('0∉1,1∉2,0∉2'),
        lambda x:'pred' not in x[0]@tptp,
        lambda x:'who' not in x[0]@eng],
  weight=2)

R('quantifier', '!', 'everyone',weight=10), R('quantifier','~!','not everyone')
R('quantifier','~?','no one') 
R('group','room','in the room',weight=8), R('group','~room','outside the room')
R('group','anywhere','anywhere')

R('X_quantifier(quantifier,group)','0[X]:(1(X)=>(?))','0 1')

def block_property(x):
    for d in x.descendants:
        if d.rule and 'property' in d.rule.args:
            return False
    return True

kw=dict(weight=0.5,state_constraint=block_property)
R('property(property,property)', '((0)<~>(1))', 'either 0 or 1 but not both',**kw)
R('property(property,property)', '((0)|(1))', '0 or 1 or both',**kw)
R('property(property,property)', '((0)&(1))', '0 and 1',**kw)
R('property(property,property)', '~((0)|(1))', 'neither 0 nor 1',**kw)
R('property(property,property,property)', '((0)&(1)&(2))', '0, 1 and 2',**kw)
R('property(property,property,property)', '((0)|(1)|(2))', '0, 1 or 2',**kw)

R('property(state)',*render_coref(polysyllogism,debug=""),
  state_constraint=lambda x:bool(x@tptp), weight=14)

R('cproperty(property,property)', '((0)=>(1))', 'who 0 1',weight=4)
R('cproperty(property,property)', '((0)=>(1))', '1 if they 0',weight=4)
R('cproperty(property,property)', '((1)<=(0))', '0 only if they 1')
R('cproperty(property,property)', '((0)<=>(1))', '0 if they 1 and vice versa')
R('cproperty(property,property)', '((0)<=>(1))', '0 if and only if they 1')


R('X_property(cproperty)','0[?←X]','0')
R('X_property(property)','0[?←X]','0',weight=0.1)

R('term(X_quantifier,X_property)', '0[?←1]', '0 1', weight=12) 

R('E_quantifier(group)','?[X]:(0(X)&(?))','someone 0')
R('E_property(property)','0[?←X]','0')
R('term(E_quantifier,E_property)', '0[?←1]', '0 1') 

R('term(property,group)', '((?[X]:(1(X)&0[?←X])))', 'at least one person 1 0')
R('term(property,group)', '((?[X]:(1(X)&0[?←X]))&(![X,Y]:((1(X)&1(Y)&(0[?←X])&(0[?←Y]))=>(X=Y))))', 'only one person 1 0')
R('term(property,group)', '(?[X,Y]:(1(X)&1(Y)&(0[?←X]&0[?←Y])&(X!=Y)))', 'more than one person 1 0')

R('term(person,adj_comp,group)', '![X]:((2(X)&(X!=0))=>1er(X,0))', '0 is the 1est 2')
R('term(person,adj_comp,group)', '![X]:((2(X)&(X!=0))=>1er(0,X))', '0 is the least 1 2');

background="&\n".join([f"({x.templates['tptp']})" for x in R.get_rules('background')])
background=split_clauses(background,name_prefix='b')
