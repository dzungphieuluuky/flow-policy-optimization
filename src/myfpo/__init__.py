import os

# Fewer graph breaks around built-in nn modules. Must be set before
# torch._dynamo is first imported; myfpo modules import torch lazily.
os.environ.setdefault("TORCHDYNAMO_INLINE_INBUILT_NN_MODULES", "1")
