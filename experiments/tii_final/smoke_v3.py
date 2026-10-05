"""Loader of the frozen MantisV2 encoder, imported by gpu_run.py, hyd/hyd_mantis_embed.py and the gpu_simproc.py runners.

Usage:  from smoke_v3 import load_mantis;  model = load_mantis(path_to_weights, 'cuda:0')
The package `mantis` and the published weights are obtained from the authors of MantisV2. The path of the weights
is passed to the runners in the environment variable V3_MANTIS. The runners look for this module in `envs/` under
their root directory and on the Python path, so a runner of a subfolder is started with PYTHONPATH set to this folder.
"""


def load_mantis(path, device):
    from mantis.architecture import MantisV2
    model = MantisV2(device=device, return_transf_layer=2, output_token="combined").from_pretrained(str(path))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
