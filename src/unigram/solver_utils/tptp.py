import os, subprocess, random
from functools import lru_cache


def to_tptp(x,background,problem='prem',neg='',mode='sat',use_hypothesis=True):
    mode={'sat':'axiom','proof':'conjecture'}[mode]
    premise = split_clauses(x.tptp,prefix=mode,name_prefix="p")
    if use_hypothesis:
        hypothesis = f"fof(hypothesis,{mode},{neg}({x.hyp_tptp}))."
    else:
        hypothesis=""
    return f"{background}\n{premise}\n{hypothesis}".replace('¿','?')

def split_clauses(x,prefix='axiom',name_prefix='',debug=False):
    clauses=x.split('&\n')
    if any(a.count('(')!=a.count(')') for a in clauses):
        clauses=[x]
    return '\n'.join([f"fof({name_prefix}{i},{prefix},{c})." for i,c in enumerate(clauses)])+"\n"

def run(expr,solver='vampire',proof=False,cache=None):
    #cache_dir="/mnt/nfs_share_magnet2/dsileo/.cache/tptp"
    cache_dir = "/mnt/tmpfs/"
    path = f'{cache_dir}/{abs(hash(expr))}{random.randint(0,1e6)}.p'
    print(expr,file=open(path,mode="w"))
    if solver=='vampire':
        exec = "/mnt/nfs_share_magnet2/dsileo/extlibs/solver/vampire"
        cmd=f"{exec} {path} --mode vampire --output_axiom_names on -t 20s"
    if solver=='z3':
        proof = ' -proof' if proof else ''
        exec = f"/mnt/nfs_share_magnet2/dsileo/extlibs/z3/build/z3_tptp {proof} -t:20 -file"
        cmd=f"{exec}:{path}"
    result = subprocess.run(cmd,
        shell=True, text=True, capture_output=True)
    output, error = result.stdout, result.stderr
    os.remove(path)
    output=output[output.find("% SZS"):]
    return output



def split_clauses(x,prefix='axiom',name_prefix='',do_split=True):
    clauses=x.split('&\n')
    if any(a.count('(')!=a.count(')') for a in clauses):
        clauses=[x]
    return '\n'.join([f"fof({name_prefix}{i},{prefix},{c})." for i,c in enumerate(clauses)])+"\n"


def extract_inferences_and_formulas(proof):
    inferences,inputs=[],[]
    for x in proof.split('\n'):
        if not x.endswith(']'): 
            continue
        x=x.split('[')[-1].strip(']')        
        if x.startswith('input'):
            inputs.append(x.replace('input ',''))
        inferences.append(x.rsplit(' ',-1)[0])
    return inferences,inputs

def extract_inferences_and_formulas_tff(tff_statements):
    inference_types = list()
    formula_names = list()
    for line in tff_statements.strip().split("\n"):
        if "inference" in line:
            inference_types.append(line.split("inference(")[1].split(",")[0])
        if "file" in line:
            formula_names.append(line.split("file")[1].split(",")[1].split("'")[1].split('/')[-1])
    return inference_types, formula_names