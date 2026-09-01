import torch
ckpt = torch.load(
    "src/checkpoints_test/multi_task_model_stainacc99.16_vesselacc92.48.pth",
    map_location="cpu", weights_only=False
)
print(type(ckpt))
if isinstance(ckpt, dict):
    for k, v in ckpt.items():
        if hasattr(v, "keys"):
            print(k, "-> dict with", len(v), "keys, e.g.", list(v.keys())[:5])
        elif hasattr(v, "shape"):
            print(k, "-> tensor", v.shape)
        else:
            print(k, "->", v)
else:
    print("Not a dict — probably a raw state_dict or a full model object")

    