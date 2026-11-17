import os,sys,subprocess
from pathlib import Path
name=sys.argv[1];root=Path(name).resolve()
paths={'xt':['tests/test_x_transformers.py','-k','grouped_attention or grouped_query or multi_query'],'lin':['test/operators/test_toeplitz_linear_operator.py'],'quad':['tests/gauss_test.py','-k','lobatto or integrate_numpy']}
env={**os.environ,'PYTHONPATH':str(root),'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','OMP_NUM_THREADS':'2','TF_NUM_INTRAOP_THREADS':'2','TF_NUM_INTEROP_THREADS':'2'}
subprocess.run([sys.executable,'-m','pytest','-q','-o','addopts=',*paths[name]],cwd=root,env=env,check=True)

