print("Installing Pytorch...")

from os import path
import subprocess
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

subprocess.call('pip3 install -q'.split() + ['http://download.pytorch.org/whl/%s/torch-0.3.0.post4-%s-linux_x86_64.whl' % (accelerator,platform)] + ['torchvision'])