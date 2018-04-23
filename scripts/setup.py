import subprocess
import sys
import os

if sys.argv[1] == 'pytorch':
	subprocess.call("./scripts/install_pytorch.py")
elif sys.argv[1] == 'tensorboard':
	subprocess.call("./scripts/run_tensorboard.sh")