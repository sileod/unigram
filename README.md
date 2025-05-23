# unigram
Unigram is a library for random (depth first) generation with context-sensitive grammars (but also context free grammars) for synthetic data creation.

One particularity is the option to generate in multiple languages in parallel (for example, tptp and pseudo-english).

Example with LogicNLI grammar:

`pip install unigram`

```python
from unigram import init_grammar, generate
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


eng, tptp = "eng","tptp"
grammar = LogicNLI()
x=generate(grammar)
print(x@eng)
print(x@tptp)
```

Citation:
```
@inproceedings{sileo-2024-scaling,
    title = "Scaling Synthetic Logical Reasoning Datasets with Context-Sensitive Declarative Grammars",
    author = "Sileo, Damien",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.301/",
    doi = "10.18653/v1/2024.emnlp-main.301",
    pages = "5275--5283",
}
```
