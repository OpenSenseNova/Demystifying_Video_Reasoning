wan_series = [
    {
        # Example: ModelConfig(model_id="krea/krea-realtime-video", origin_file_pattern="krea-realtime-video-14b.safetensors")
        "model_hash": "5ec04e02b42d2580483ad69f4e76346a",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": "diffsynth.models.wan_video_text_encoder.WanTextEncoder",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth")
        "model_hash": "ccc42284ea13e1ad04693284c7a09be6",
        "model_name": "wan_video_vae",
        "model_class": "diffsynth.models.wan_video_vae.WanVideoVAE",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="meituan-longcat/LongCat-Video", origin_file_pattern="dit/diffusion_pytorch_model*.safetensors")
        "model_hash": "8b27900f680d7251ce44e2dc8ae1ffef",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.longcat_video_dit.LongCatVideoTransformer3DModel",
    },
    {
        # Example: ModelConfig(model_id="ByteDance/Video-As-Prompt-Wan2.1-14B", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors")
        "model_hash": "5f90e66a0672219f12d9a626c8c21f61",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTFromDiffusers"
    },
    {
        # Example: ModelConfig(model_id="ByteDance/Video-As-Prompt-Wan2.1-14B", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors")
        "model_hash": "5f90e66a0672219f12d9a626c8c21f61",
        "model_name": "wan_video_vap",
        "model_class": "diffsynth.models.wan_video_mot.MotWanModel",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_mot.WanVideoMotStateDictConverter"
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
        "model_hash": "5941c53e207d62f20f9025686193c40b",
        "model_name": "wan_video_image_encoder",
        "model_class": "diffsynth.models.wan_video_image_encoder.WanImageEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_image_encoder.WanImageEncoderStateDictConverter"
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1", origin_file_pattern="model.safetensors")
        "model_hash": "dbd5ec76bbf977983f972c151d545389",
        "model_name": "wan_video_motion_controller",
        "model_class": "diffsynth.models.wan_video_motion_controller.WanMotionControllerModel",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "9269f8db9040a9d860eaca435be61814",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-FLF2V-14B-720P", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "3ef3b1f8e1dab83d5b71fd7b617f859f",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'has_image_pos_emb': True}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "349723183fc063b2bfc10bb2835cf677",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 48, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-InP", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "6d6ccde6845b95ad9114ab993d917893",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-14B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "efa44cddf936c70abd0ea28b6cbe946c",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 48, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-14B-InP", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "6bfcfb3b342cb286ce886889d519a77e",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "ac6a5aa74f4a0aab6f64eb9a72f19901",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 32, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06, 'has_ref_conv': False, 'add_control_adapter': True, 'in_dim_control_adapter': 24}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "70ddad9d3a133785da5ea371aae09504",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 48, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06, 'has_ref_conv': True}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "b61c605c2adbd23124d152ed28e049ae",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 32, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'has_ref_conv': False, 'add_control_adapter': True, 'in_dim_control_adapter': 24}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "26bde73488a92e64cc20b0a7485b9e5b",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 48, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'has_ref_conv': True}
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "aafcfd9672c3a2456dc46e1cb6e52c70",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="iic/VACE-Wan2.1-1.3B-Preview", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "a61453409b67cd3246cf0c3bebad47ba",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="iic/VACE-Wan2.1-1.3B-Preview", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "a61453409b67cd3246cf0c3bebad47ba",
        "model_name": "wan_video_vace",
        "model_class": "diffsynth.models.wan_video_vace.VaceWanModel",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vace.VaceWanModelDictConverter"
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "7a513e1f257a861512b1afd387a8ecd9",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "7a513e1f257a861512b1afd387a8ecd9",
        "model_name": "wan_video_vace",
        "model_class": "diffsynth.models.wan_video_vace.VaceWanModel",
        "extra_kwargs": {'vace_layers': (0, 5, 10, 15, 20, 25, 30, 35), 'vace_in_dim': 96, 'patch_size': (1, 2, 2), 'has_image_input': False, 'dim': 5120, 'num_heads': 40, 'ffn_dim': 13824, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vace.VaceWanModelDictConverter"
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "31fa352acb8a1b1d33cd8764273d80a2",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter"
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "31fa352acb8a1b1d33cd8764273d80a2",
        "model_name": "wan_video_animate_adapter",
        "model_class": "diffsynth.models.wan_video_animate_adapter.WanAnimateAdapter",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_animate_adapter.WanAnimateAdapterStateDictConverter"
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control-Camera", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors")
        "model_hash": "47dbeab5e560db3180adf51dc0232fb1",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'has_ref_conv': False, 'add_control_adapter': True, 'in_dim_control_adapter': 24, 'require_clip_embedding': False}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors")
        "model_hash": "2267d489f0ceb9f21836532952852ee5",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 52, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'has_ref_conv': True, 'require_clip_embedding': False},
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors")
        "model_hash": "5b013604280dd715f8457c6ed6d6a626",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'require_clip_embedding': False}
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "966cffdcc52f9c46c391768b27637614",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit_s2v.WanS2VModel",
        "extra_kwargs": {'dim': 5120, 'in_dim': 16, 'ffn_dim': 13824, 'out_dim': 16, 'text_dim': 4096, 'freq_dim': 256, 'eps': 1e-06, 'patch_size': (1, 2, 2), 'num_heads': 40, 'num_layers': 40, 'cond_dim': 16, 'audio_dim': 1024, 'num_audio_token': 4}
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 48, 'dim': 3072, 'ffn_dim': 14336, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 48, 'num_heads': 24, 'num_layers': 30, 'eps': 1e-06, 'seperated_timestep': True, 'require_clip_embedding': False, 'require_vae_embedding': False, 'fuse_vae_embedding_in_latents': True}
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth")
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": "diffsynth.models.wan_video_vae.WanVideoVAE38",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors")
        "model_hash": "06be60f3a4526586d8431cd038a71486",
        "model_name": "wans2v_audio_encoder",
        "model_class": "diffsynth.models.wav2vec.WanS2VAudioEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wans2v_audio_encoder.WanS2VAudioEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/WanToDance-14B", origin_file_pattern="global_model.safetensors")
        "model_hash": "eb18873fc0ba77b541eb7b62dbcd2059",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'wantodance_enable_music_inject': True, 'wantodance_music_inject_layers': [0, 4, 8, 12, 16, 20, 24, 27], 'wantodance_enable_refimage': True, 'has_ref_conv': True, 'wantodance_enable_refface': False, 'wantodance_enable_global': True, 'wantodance_enable_dynamicfps': True, 'wantodance_enable_unimodel': True}
    },
]

ltx2_series = [
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "aca7b0bbf8415e9c98360750268915fc",
        "model_name": "ltx2_dit",
        "model_class": "diffsynth.models.ltx2_dit.LTXModel",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_dit.LTXModelStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="transformer.safetensors")
        "model_hash": "c567aaa37d5ed7454c73aa6024458661",
        "model_name": "ltx2_dit",
        "model_class": "diffsynth.models.ltx2_dit.LTXModel",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_dit.LTXModelStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "aca7b0bbf8415e9c98360750268915fc",
        "model_name": "ltx2_video_vae_encoder",
        "model_class": "diffsynth.models.ltx2_video_vae.LTX2VideoEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_video_vae.LTX2VideoEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="video_vae_encoder.safetensors")
        "model_hash": "7f7e904a53260ec0351b05f32153754b",
        "model_name": "ltx2_video_vae_encoder",
        "model_class": "diffsynth.models.ltx2_video_vae.LTX2VideoEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_video_vae.LTX2VideoEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "aca7b0bbf8415e9c98360750268915fc",
        "model_name": "ltx2_video_vae_decoder",
        "model_class": "diffsynth.models.ltx2_video_vae.LTX2VideoDecoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_video_vae.LTX2VideoDecoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="video_vae_decoder.safetensors")
        "model_hash": "dc6029ca2825147872b45e35a2dc3a97",
        "model_name": "ltx2_video_vae_decoder",
        "model_class": "diffsynth.models.ltx2_video_vae.LTX2VideoDecoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_video_vae.LTX2VideoDecoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "aca7b0bbf8415e9c98360750268915fc",
        "model_name": "ltx2_audio_vae_decoder",
        "model_class": "diffsynth.models.ltx2_audio_vae.LTX2AudioDecoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2AudioDecoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="audio_vae_decoder.safetensors")
        "model_hash": "7d7823dde8f1ea0b50fb07ac329dd4cb",
        "model_name": "ltx2_audio_vae_decoder",
        "model_class": "diffsynth.models.ltx2_audio_vae.LTX2AudioDecoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2AudioDecoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "aca7b0bbf8415e9c98360750268915fc",
        "model_name": "ltx2_audio_vocoder",
        "model_class": "diffsynth.models.ltx2_audio_vae.LTX2Vocoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2VocoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="audio_vocoder.safetensors")
        "model_hash": "f471360f6b24bef702ab73133d9f8bb9",
        "model_name": "ltx2_audio_vocoder",
        "model_class": "diffsynth.models.ltx2_audio_vae.LTX2Vocoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2VocoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "aca7b0bbf8415e9c98360750268915fc",
        "model_name": "ltx2_audio_vae_encoder",
        "model_class": "diffsynth.models.ltx2_audio_vae.LTX2AudioEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2AudioEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="audio_vae_encoder.safetensors")
        "model_hash": "29338f3b95e7e312a3460a482e4f4554",
        "model_name": "ltx2_audio_vae_encoder",
        "model_class": "diffsynth.models.ltx2_audio_vae.LTX2AudioEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2AudioEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "aca7b0bbf8415e9c98360750268915fc",
        "model_name": "ltx2_text_encoder_post_modules",
        "model_class": "diffsynth.models.ltx2_text_encoder.LTX2TextEncoderPostModules",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_text_encoder.LTX2TextEncoderPostModulesStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="text_encoder_post_modules.safetensors")
        "model_hash": "981629689c8be92a712ab3c5eb4fc3f6",
        "model_name": "ltx2_text_encoder_post_modules",
        "model_class": "diffsynth.models.ltx2_text_encoder.LTX2TextEncoderPostModules",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_text_encoder.LTX2TextEncoderPostModulesStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors")
        "model_hash": "33917f31c4a79196171154cca39f165e",
        "model_name": "ltx2_text_encoder",
        "model_class": "diffsynth.models.ltx2_text_encoder.LTX2TextEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_text_encoder.LTX2TextEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "c79c458c6e99e0e14d47e676761732d2",
        "model_name": "ltx2_latent_upsampler",
        "model_class": "diffsynth.models.ltx2_upsampler.LTX2LatentUpsampler",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-22b-dev.safetensors")
        "model_hash": "f3a83ecf3995dcc4fae2d27e08ad5767",
        "model_name": "ltx2_dit",
        "model_class": "diffsynth.models.ltx2_dit.LTXModel",
        "extra_kwargs": {"apply_gated_attention": True, "cross_attention_adaln": True, "caption_channels": None},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_dit.LTXModelStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-22b-dev.safetensors")
        "model_hash": "f3a83ecf3995dcc4fae2d27e08ad5767",
        "model_name": "ltx2_video_vae_encoder",
        "model_class": "diffsynth.models.ltx2_video_vae.LTX2VideoEncoder",
        "extra_kwargs": {"encoder_version": "ltx-2.3"},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_video_vae.LTX2VideoEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-22b-dev.safetensors")
        "model_hash": "f3a83ecf3995dcc4fae2d27e08ad5767",
        "model_name": "ltx2_video_vae_decoder",
        "model_class": "diffsynth.models.ltx2_video_vae.LTX2VideoDecoder",
        "extra_kwargs": {"decoder_version": "ltx-2.3"},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_video_vae.LTX2VideoDecoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-22b-dev.safetensors")
        "model_hash": "f3a83ecf3995dcc4fae2d27e08ad5767",
        "model_name": "ltx2_audio_vae_decoder",
        "model_class": "diffsynth.models.ltx2_audio_vae.LTX2AudioDecoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2AudioDecoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-22b-dev.safetensors")
        "model_hash": "f3a83ecf3995dcc4fae2d27e08ad5767",
        "model_name": "ltx2_audio_vocoder",
        "model_class": "diffsynth.models.ltx2_audio_vae.LTX2VocoderWithBWE",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2VocoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-22b-dev.safetensors")
        "model_hash": "f3a83ecf3995dcc4fae2d27e08ad5767",
        "model_name": "ltx2_audio_vae_encoder",
        "model_class": "diffsynth.models.ltx2_audio_vae.LTX2AudioEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2AudioEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-22b-dev.safetensors")
        "model_hash": "f3a83ecf3995dcc4fae2d27e08ad5767",
        "model_name": "ltx2_text_encoder_post_modules",
        "model_class": "diffsynth.models.ltx2_text_encoder.LTX2TextEncoderPostModules",
        "extra_kwargs": {"separated_audio_video": True, "embedding_dim_gemma": 3840, "num_layers_gemma": 49, "video_attention_heads": 32, "video_attention_head_dim": 128, "audio_attention_heads": 32, "audio_attention_head_dim": 64, "num_connector_layers": 8, "apply_gated_attention": True},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_text_encoder.LTX2TextEncoderPostModulesStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
        "model_hash": "aed408774d694a2452f69936c32febb5",
        "model_name": "ltx2_latent_upsampler",
        "model_class": "diffsynth.models.ltx2_upsampler.LTX2LatentUpsampler",
        "extra_kwargs": {"rational_resampler": False},
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="transformer.safetensors")
        "model_hash": "1c55afad76ed33c112a2978550b524d1",
        "model_name": "ltx2_dit",
        "model_class": "diffsynth.models.ltx2_dit.LTXModel",
        "extra_kwargs": {"apply_gated_attention": True, "cross_attention_adaln": True, "caption_channels": None},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_dit.LTXModelStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="video_vae_encoder.safetensors")
        "model_hash": "eecdc07c2ec30863b8a2b8b2134036cf",
        "model_name": "ltx2_video_vae_encoder",
        "model_class": "diffsynth.models.ltx2_video_vae.LTX2VideoEncoder",
        "extra_kwargs": {"encoder_version": "ltx-2.3"},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_video_vae.LTX2VideoEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="video_vae_decoder.safetensors")
        "model_hash": "deda2f542e17ee25bc8c38fd605316ea",
        "model_name": "ltx2_video_vae_decoder",
        "model_class": "diffsynth.models.ltx2_video_vae.LTX2VideoDecoder",
        "extra_kwargs": {"decoder_version": "ltx-2.3"},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_video_vae.LTX2VideoDecoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="audio_vocoder.safetensors")
        "model_hash": "7d7823dde8f1ea0b50fb07ac329dd4cb",
        "model_name": "ltx2_audio_vae_decoder",
        "model_class": "diffsynth.models.ltx2_audio_vae.LTX2AudioDecoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2AudioDecoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="audio_vae_encoder.safetensors")
        "model_hash": "29338f3b95e7e312a3460a482e4f4554",
        "model_name": "ltx2_audio_vae_encoder",
        "model_class": "diffsynth.models.ltx2_audio_vae.LTX2AudioEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2AudioEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="audio_vocoder.safetensors")
        "model_hash": "cd436c99e69ec5c80f050f0944f02a15",
        "model_name": "ltx2_audio_vocoder",
        "model_class": "diffsynth.models.ltx2_audio_vae.LTX2VocoderWithBWE",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2VocoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="text_encoder_post_modules.safetensors")
        "model_hash": "05da2aab1c4b061f72c426311c165a43",
        "model_name": "ltx2_text_encoder_post_modules",
        "model_class": "diffsynth.models.ltx2_text_encoder.LTX2TextEncoderPostModules",
        "extra_kwargs": {"separated_audio_video": True, "embedding_dim_gemma": 3840, "num_layers_gemma": 49, "video_attention_heads": 32, "video_attention_head_dim": 128, "audio_attention_heads": 32, "audio_attention_head_dim": 64, "num_connector_layers": 8, "apply_gated_attention": True},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_text_encoder.LTX2TextEncoderPostModulesStateDictConverter",
    },
]

MODEL_CONFIGS = wan_series + ltx2_series
