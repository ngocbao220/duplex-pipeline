from diffusers import DPMSolverMultistepScheduler
scheduler = DPMSolverMultistepScheduler()
print(hasattr(scheduler, "model_outputs"))
scheduler.set_timesteps(10)
print(getattr(scheduler, "model_outputs", "None"))
