from pathlib import Path
import sysconfig
import shutil
base=Path(sysconfig.get_paths()['purelib'])
for source in (base/'ot').rglob('*.pyd'):
    target=Path.cwd()/source.relative_to(base)
    if target.parent.is_dir():
        shutil.copyfile(source,target)

