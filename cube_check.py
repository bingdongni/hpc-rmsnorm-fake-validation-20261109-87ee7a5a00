"""Exercise the instrumented Python module with the released PySCF integrals."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pytest
from pyscf import gto, scf, tools

def load_module(path):
    spec=importlib.util.spec_from_file_location("pyscf.tools.cubegen",path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    tools.cubegen=mod
    return mod

def measure(path):
    import resource
    cube=load_module(path)
    mol=gto.M(atom="H 0 0 0; H 0 0 .74",basis="sto-3g",verbose=0)
    dm=scf.RHF(mol).run().make_rdm1()
    with tempfile.TemporaryDirectory() as tmp:
        file=Path(tmp)/"molecule.cube"
        rho=cube.density(mol,str(file),dm,137,127,111)
        print(json.dumps({"maxrss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          "sum":float(rho.sum()),"output_bytes":file.stat().st_size}))

def validate(head, base):
    rel=Path("pyscf/tools/cubegen.py")
    head=head.resolve()
    sources=[base.resolve()/rel,head/rel]
    stats=[]
    for path in sources:
        log=subprocess.check_output([sys.executable,__file__,"--measure",str(path)],text=True)
        stats.append(json.loads(log.splitlines()[-1]))
    print("Density peak RSS",json.dumps(stats),flush=True)
    assert stats[1]["maxrss_kib"] < stats[0]["maxrss_kib"]
    np.testing.assert_allclose(stats[1]["sum"],stats[0]["sum"],atol=1e-9,rtol=1e-12)
    assert stats[1]["output_bytes"]==stats[0]["output_bytes"]
    mol=gto.M(atom="O 0 0 0; H 0 .76 .48; H 0 -.76 .48",basis="sto-3g",verbose=0)
    mf=scf.RHF(mol).run()
    with tempfile.TemporaryDirectory() as tmp:
        for operation,arg in [("orbital",mf.mo_coeff[:,0]),("density",mf.make_rdm1()),("mep",mf.make_rdm1())]:
            results=[]
            for k,path in enumerate(sources):
                mod=load_module(path)
                file=Path(tmp)/f"{operation}-{k}.cube"
                field=getattr(mod,operation)(mol,str(file),arg,23,21,19)
                results.append((field, mod.Cube(mol).read(str(file))))
            for ref,trial in zip(results[0],results[1]):
                np.testing.assert_allclose(ref,trial,rtol=1e-12,atol=1e-12)
            print(operation,"returned and serialized fields agree",flush=True)
    load_module(sources[1])
    raise SystemExit(pytest.main(["-q",str(head/"pyscf/tools/test/test_cubegen.py"),"--import-mode=importlib","--rootdir",os.getcwd(),"--noconftest"]))

if __name__=="__main__":
    if sys.argv[1]=="--measure": measure(Path(sys.argv[2]))
    else: validate(*map(Path,sys.argv[1:]))
