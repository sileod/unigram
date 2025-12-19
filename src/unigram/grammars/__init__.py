from .FOL import FOL_grammar
from .tinypy import tinypy_grammar
from ..grammar import init_grammar
import random
import string

def arith():
    g = init_grammar(['py'], name="arithmetics", preprocess_template=lambda s:s)
    g('start(expr)',        '{0}')
    g('expr(expr)',       '({0})',            weight=1)
    g('expr(expr,expr)',  '{0} + {1}', weight=2)
    g('expr(expr,expr)',  '{0} - {1}', weight=1)
    g('expr(expr,expr)',  '{0} * {1}')
    g('expr(expr,expr)',  '{0} / {1}')
    g('expr(expr)',       '({0})**2',         weight=.25)
    g('expr(value)',       '{0}',weight= 10)
    g('value',  'NUM')

    return g


def regex_grammar():

    from faker import Faker
    fake = Faker()
    wordlist = fake.words(nb=100,unique=True)
    
    R = init_grammar(['re'], preprocess_template=lambda x: x)

    R('start(regex)', '{0}')
    R('regex(regex,regex)', '{0}{1}', weight=2)
    R('regex(regex)', '({0})', weight=2)
    R('regex(regex,regex)', '{0}|{1}', weight=2)
    R('regex(char)', '{0}',weight=1)
    R('regex(word)', '{0}',weight=3)

    for w in random.sample(wordlist, 8):
        R('word', w)

    R('regex(regex)?', '{0}?')
    R('regex(regex)*', '{0}*')
    R('regex(regex)+', '{0}+')
    R('regex(rangechar,rangechar)', '[{0}-{1}]')
    R('regex(predef)', '{0}',weight=3)

    chars = string.ascii_letters + string.digits
    for c in chars:
        R('char', c)
        R('rangechar', c)

    for s in [r'\d', r'\w', r'\s', '.', r'\.']:
        R('predef', s, weight=1)

    for s in [r'\D', r'\W', r'\S', r'\\', r'\(', r'\)', r'\[', r'\]', r'\t', r'\n']:
        R('predef', s, weight=0.25)

    return R