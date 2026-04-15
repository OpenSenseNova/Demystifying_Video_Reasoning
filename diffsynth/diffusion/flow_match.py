import torch, math


class FlowMatchScheduler():

    def __init__(self, template="Wan"):
        self.set_timesteps_fn = {
            "Wan": FlowMatchScheduler.set_timesteps_wan,
            "LTX-2": FlowMatchScheduler.set_timesteps_ltx2,
        }.get(template, FlowMatchScheduler.set_timesteps_wan)
        self.num_train_timesteps = 1000

    @staticmethod
    def set_timesteps_wan(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 5 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def _calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    @staticmethod
    def set_timesteps_ltx2(num_inference_steps=100, denoising_strength=1.0, dynamic_shift_len=None, terminal=0.1, special_case=None):
        num_train_timesteps = 1000
        if special_case == "stage2":
            sigmas = torch.Tensor([0.909375, 0.725, 0.421875])
        elif special_case == "ditilled_stage1":
            sigmas = torch.Tensor([1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875])
        else:
            dynamic_shift_len = dynamic_shift_len or 4096
            sigma_shift = FlowMatchScheduler._calculate_shift(
                image_seq_len=dynamic_shift_len,
                base_seq_len=1024,
                max_seq_len=4096,
                base_shift=0.95,
                max_shift=2.05,
            )
            sigma_min = 0.0
            sigma_max = 1.0
            sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
            sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
            sigmas = math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1))
            one_minus_z = 1.0 - sigmas
            scale_factor = one_minus_z[-1] / (1 - terminal)
            sigmas = 1.0 - (one_minus_z / scale_factor)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, **kwargs):
        self.sigmas, self.timesteps = self.set_timesteps_fn(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            **kwargs,
        )

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
