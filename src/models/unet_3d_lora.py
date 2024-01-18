from collections import defaultdict
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Union

import safetensors
import torch
from torch import nn
from huggingface_hub.utils import validate_hf_hub_args

from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.loaders.unet import LORA_WEIGHT_NAME, LORA_WEIGHT_NAME_SAFE
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_model_dict_into_meta

from diffusers.utils import (
    USE_PEFT_BACKEND,
    _get_model_file,
    is_accelerate_available,
    logging,
)

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

logger = logging.get_logger(__name__)

class UNet3DConditionLoadersMixin(UNet2DConditionLoadersMixin):
    """
    Load LoRA layers into a [`UNet2DCondtionModel`].
    """

    @validate_hf_hub_args
    def load_attn_procs(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
        r"""
        Load pretrained attention processor layers into [`UNet2DConditionModel`]. Attention processor layers have to be
        defined in
        [`attention_processor.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
        and be a `torch.nn.Module` class.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the model id (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a directory (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.unet.load_attn_procs(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        ```
        """
        from diffusers.models.attention_processor import CustomDiffusionAttnProcessor
        from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear, LoRAConv2dLayer, LoRALinearLayer

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        network_alphas = kwargs.pop("network_alphas", None)

        _pipeline = kwargs.pop("_pipeline", None)

        is_network_alphas_none = network_alphas is None

        allow_pickle = False

        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # Let's first try to load .safetensors weights
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = safetensors.torch.load_file(model_file, device="cpu")
                except IOError as e:
                    if not allow_pickle:
                        raise e
                    # try loading non-safetensors weights
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # fill attn processors
        lora_layers_list = []

        is_lora = all(("lora" in k or k.endswith(".alpha")) for k in state_dict.keys()) and not USE_PEFT_BACKEND
        is_custom_diffusion = any("custom_diffusion" in k for k in state_dict.keys())

        if is_lora:
            # correct keys
            state_dict, network_alphas = self.convert_state_dict_legacy_attn_format(state_dict, network_alphas)

            if network_alphas is not None:
                network_alphas_keys = list(network_alphas.keys())
                used_network_alphas_keys = set()

            lora_grouped_dict = defaultdict(dict)
            mapped_network_alphas = {}

            all_keys = list(state_dict.keys())
            for key in all_keys:
                value = state_dict.pop(key)
                attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                lora_grouped_dict[attn_processor_key][sub_key] = value

                # Create another `mapped_network_alphas` dictionary so that we can properly map them.
                if network_alphas is not None:
                    for k in network_alphas_keys:
                        if k.replace(".alpha", "") in key:
                            mapped_network_alphas.update({attn_processor_key: network_alphas.get(k)})
                            used_network_alphas_keys.add(k)

            if not is_network_alphas_none:
                if len(set(network_alphas_keys) - used_network_alphas_keys) > 0:
                    raise ValueError(
                        f"The `network_alphas` has to be empty at this point but has the following keys \n\n {', '.join(network_alphas.keys())}"
                    )

            if len(state_dict) > 0:
                raise ValueError(
                    f"The `state_dict` has to be empty at this point but has the following keys \n\n {', '.join(state_dict.keys())}"
                )

            for key, value_dict in lora_grouped_dict.items():
                attn_processor = self
                for sub_key in key.split("."):
                    attn_processor = getattr(attn_processor, sub_key)

                # Process non-attention layers, which don't have to_{k,v,q,out_proj}_lora layers
                # or add_{k,v,q,out_proj}_proj_lora layers.
                rank = value_dict["lora.down.weight"].shape[0]

                if isinstance(attn_processor, LoRACompatibleConv):
                    in_features = attn_processor.in_channels
                    out_features = attn_processor.out_channels
                    kernel_size = attn_processor.kernel_size

                    ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                    with ctx():
                        lora = LoRAConv2dLayer(
                            in_features=in_features,
                            out_features=out_features,
                            rank=rank,
                            kernel_size=kernel_size,
                            stride=attn_processor.stride,
                            padding=attn_processor.padding,
                            network_alpha=mapped_network_alphas.get(key),
                        )
                elif isinstance(attn_processor, LoRACompatibleLinear):
                    ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                    with ctx():
                        lora = LoRALinearLayer(
                            attn_processor.in_features,
                            attn_processor.out_features,
                            rank,
                            mapped_network_alphas.get(key),
                        )
                else:
                    #raise ValueError(f"Module {key} is not a LoRACompatibleConv or LoRACompatibleLinear module.")
                    continue

                value_dict = {k.replace("lora.", ""): v for k, v in value_dict.items()}
                lora_layers_list.append((attn_processor, lora))

                if low_cpu_mem_usage:
                    device = next(iter(value_dict.values())).device
                    dtype = next(iter(value_dict.values())).dtype
                    load_model_dict_into_meta(lora, value_dict, device=device, dtype=dtype)
                else:
                    lora.load_state_dict(value_dict)

        elif is_custom_diffusion:
            attn_processors = {}
            custom_diffusion_grouped_dict = defaultdict(dict)
            for key, value in state_dict.items():
                if len(value) == 0:
                    custom_diffusion_grouped_dict[key] = {}
                else:
                    if "to_out" in key:
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                    else:
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-2]), ".".join(key.split(".")[-2:])
                    custom_diffusion_grouped_dict[attn_processor_key][sub_key] = value

            for key, value_dict in custom_diffusion_grouped_dict.items():
                if len(value_dict) == 0:
                    attn_processors[key] = CustomDiffusionAttnProcessor(
                        train_kv=False, train_q_out=False, hidden_size=None, cross_attention_dim=None
                    )
                else:
                    cross_attention_dim = value_dict["to_k_custom_diffusion.weight"].shape[1]
                    hidden_size = value_dict["to_k_custom_diffusion.weight"].shape[0]
                    train_q_out = True if "to_q_custom_diffusion.weight" in value_dict else False
                    attn_processors[key] = CustomDiffusionAttnProcessor(
                        train_kv=True,
                        train_q_out=train_q_out,
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                    )
                    attn_processors[key].load_state_dict(value_dict)
        elif USE_PEFT_BACKEND:
            # In that case we have nothing to do as loading the adapter weights is already handled above by `set_peft_model_state_dict`
            # on the Unet
            pass
        else:
            raise ValueError(
                f"{model_file} does not seem to be in the correct format expected by LoRA or Custom Diffusion training."
            )

        # <Unsafe code
        # We can be sure that the following works as it just sets attention processors, lora layers and puts all in the same dtype
        # Now we remove any existing hooks to
        is_model_cpu_offload = False
        is_sequential_cpu_offload = False

        # For PEFT backend the Unet is already offloaded at this stage as it is handled inside `lora_lora_weights_into_unet`
        if not USE_PEFT_BACKEND:
            if _pipeline is not None:
                for _, component in _pipeline.components.items():
                    if isinstance(component, nn.Module) and hasattr(component, "_hf_hook"):
                        is_model_cpu_offload = isinstance(getattr(component, "_hf_hook"), CpuOffload)
                        is_sequential_cpu_offload = isinstance(getattr(component, "_hf_hook"), AlignDevicesHook)

                        logger.info(
                            "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
                        )
                        remove_hook_from_module(component, recurse=is_sequential_cpu_offload)

            # only custom diffusion needs to set attn processors
            if is_custom_diffusion:
                self.set_attn_processor(attn_processors)

            # set lora layers
            for target_module, lora_layer in lora_layers_list:
                target_module.set_lora_layer(lora_layer)

            self.to(dtype=self.dtype, device=self.device)

            # Offload back.
            if is_model_cpu_offload:
                _pipeline.enable_model_cpu_offload()
            elif is_sequential_cpu_offload:
                _pipeline.enable_sequential_cpu_offload()
            # Unsafe code />