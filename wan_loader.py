import os
import torch
from pipelines.wan_video import WanVideoPipeline, ModelConfig
from pipelines.wan_video_face_swap import WanVideoPipeline_FaceSwap


def load_wan_pipe(
    base_path, torch_dtype=torch.bfloat16, face_swap=False, use_vace=False, device="cuda"
):
    if not use_vace:
        diffusion_model_files = [
            f"diffusion_pytorch_model-0000{i}-of-00006.safetensors" for i in range(1, 7)
        ]
    else:
        diffusion_model_files = [
            f"diffusion_pytorch_model-0000{i}-of-00007.safetensors" for i in range(1, 8)
        ]    
    
    diffusion_model_paths = [
        os.path.join(base_path, fname) for fname in diffusion_model_files
    ]

    pipe_cls = WanVideoPipeline_FaceSwap if face_swap else WanVideoPipeline

    pipe = pipe_cls.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(
                path=diffusion_model_paths,
                offload_device="cpu",
                skip_download=True,
            ),
            ModelConfig(
                path=os.path.join(base_path, "models_t5_umt5-xxl-enc-bf16.pth"),
                offload_device="cpu",
                skip_download=True,
            ),
            ModelConfig(
                path=os.path.join(base_path, "Wan2.1_VAE.pth"),
                offload_device="cpu",
                skip_download=True,
            ),
        ],
        tokenizer_config=ModelConfig(
            path=os.path.join(base_path, "google/umt5-xxl/"),
            offload_device="cpu",
            skip_download=True,
        ),
    )
    pipe.enable_vram_management()
    return pipe
