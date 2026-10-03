"""Reject private manuscript/history trees in the public repository index."""
from pathlib import Path,PurePosixPath
import subprocess,sys
root=Path(__file__).resolve().parents[1]
tracked=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
forbidden={'paper','paper_tii','paper_zh','paper_zh_word','history','original_materials','materials_original'}
errors=[]
for name in filter(None,tracked):
 path=PurePosixPath(name)
 if set(path.parts)&forbidden:errors.append(name)
 p=root/name
 if p.is_symlink():errors.append(name+' (symlink is not allowed in public release)')
if errors:
 print('Private publication boundary violation:\n'+'\n'.join(errors));sys.exit(1)
print('Public boundary passed: no manuscript/history/private-material trees or symlinks in tracked files.')
