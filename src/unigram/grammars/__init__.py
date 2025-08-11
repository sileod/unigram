from .FOL import FOL_grammar
from ..grammar import init_grammar

def arith():
    g = init_grammar(['py'], name="arithmetics", preprocess_template=lambda s:s)
    g('start(expr)',        '{0}')
    g('expr(expr)',       '({0})',            weight=0.5)
    g('expr(expr,expr)',  '{0} + {1}', weight=3)
    g('expr(expr,expr)',  '{0} - {1}', weight=2)
    g('expr(expr,expr)',  '{0} * {1}')
    g('expr(expr,expr)',  '{0} / {1}')
    g('expr(expr)',       '({0})**2',         weight=.25)
    g('expr(value)',       '{0}',weight=0.3) # Turn the weight to 0.smth in order to reach deeper depth (was 1)
    g('value',  'NUM')

    return g
