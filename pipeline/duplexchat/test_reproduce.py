import torch
from diffusers import DPMSolverMultistepScheduler

scheduler = DPMSolverMultistepScheduler(
    algorithm_type="dpmsolver++",
    solver_order=2,
)

def run_chunk(seq_len):
    print(f"Running chunk with seq_len={seq_len}")
    scheduler.set_timesteps(10)
    latents = torch.randn(1, seq_len, 32)
    for t in scheduler.timesteps:
        # Simulate model output (same shape as latents)
        model_output = torch.randn(1, seq_len, 32)
        latents = scheduler.step(model_output, t, latents).prev_sample
    print(f"Finished chunk with seq_len={seq_len}")

run_chunk(3000)
run_chunk(383)
