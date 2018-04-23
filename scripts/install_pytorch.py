print("Installing Pytorch...")

from os import path
import subprocess
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'
subprocess.call('pip install -q http://download.pytorch.org/whl/{}/torch-0.3.0.post4-{}-linux_x86_64.whl torchvision'.format(accelerator,platform))