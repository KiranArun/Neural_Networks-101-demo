import subprocess
import sys
import os

if 'pytorch' in sys.argv[1:]:
	subprocess.call(['python3', '/content/Neural_Networks-101-demo/scripts/install_pytorch.py'])

if 'tensorboard' in sys.argv[1:]:
	subprocess.call(['chmod', '+x', '/content/Neural_Networks-101-demo/scripts/install_ngrok.sh'])
	subprocess.call(['/content/Neural_Networks-101-demo/scripts/install_ngrok.sh'])
	

if 'helper_funcs' in sys.argv[1:]:
	print('Getting helper functions...')
	subprocess.call(['cp', '/content/Neural_Networks-101-demo/helper_funcs.py', '/content/'])