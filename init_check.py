"""
This code server to verify capabilities on training machine.
"""

import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
print(torch.cuda.device_count())

