import subprocess
import sys
import os

if 'pytorch' in sys.argv[1:]:
	subprocess.call(['python3', '/content/Neural_Networks-demo/scripts/install_pytorch.py'])

if 'tensorboard' in sys.argv[1:]:
	subprocess.call(['chmod', '+x', '/content/Neural_Networks-demo/scripts/run_tensorboard.sh'])
	subprocess.call(['/content/Neural_Networks-demo/scripts/run_tensorboard.sh'])

if 'helper_funcs' in sys.argv[1:]:
	print('Getting helper functions...')
	subprocess.call(['cp', '/content/Neural_Networks-demo/helper_funcs.py', '/content/'])