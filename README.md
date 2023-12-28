---
pipeline_tag: text-to-image
inference: false
license: other
license_name: sai-nc-community
license_link: https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.TXT
tags:
- tensorrt
- sdxl-turbo
- text-to-image
---

# SDXL-Turbo Tensorrt 
## Introduction

<!-- Provide a quick summary of what the model is/does. -->
This repository hosts the TensorRT version of **Stable Diffusion XL Turbo** created in collaboration with [NVIDIA](https://huggingface.co/nvidia). The optimized versions give substantial improvements in speed and efficiency.

SDXL-Turbo is a fast generative text-to-image model that can synthesize photorealistic images from a text prompt in a single network evaluation.
A real-time demo is available here: http://clipdrop.co/stable-diffusion-turbo

## Model Details

### Model Description
SDXL-Turbo is a distilled version of [SDXL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), trained for real-time synthesis. 

- **Developed by:** Stability AI
- **Model type:** Generative text-to-image model
- **Model Description:** This is a conversion of the [SDXL Turbo](https://huggingface.co/stabilityai/sdxl-turbo)


## Performance
#### Timings for 4 steps at 512x512

| Accelerator | CLIP                     | Unet                        | VAE                    |Total                   |
|-------------|--------------------------|-----------------------------|------------------------|------------------------|
| A100        | 1.03 ms                  | 79.31 ms                    | 53.69.34 ms            | 138.57 ms              |
| H100        | 0.78 ms                  | 48.87 ms                    | 30.35 ms               | 83.8 ms                |


## Usage Example
1. Following the [setup instructions](https://github.com/rajeevsrao/TensorRT/blob/release/9.2/demo/Diffusion/README.md) on launching a TensorRT NGC container.
```shell
git clone https://github.com/rajeevsrao/TensorRT.git
cd TensorRT
git checkout release/9.2
docker run --rm -it --gpus all -v $PWD:/workspace nvcr.io/nvidia/pytorch:23.11-py3 /bin/bash
```

2. Download the SDXL LCM TensorRT files from this repo
```shell
git lfs install 
git clone https://huggingface.co/stabilityai/sdxl-turbo-tensorrt
cd sdxl-turbo-tensorrt
git lfs pull
cd ..
```

3. Install libraries and requirements
```shell
cd demo/Diffusion
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
python3 -m pip install --pre --upgrade --extra-index-url https://pypi.nvidia.com tensorrt
```

4. Perform TensorRT optimized inference:

  - **SDXL Turbo**
        
    Works best for 512x512 images and EulerA scheduler. The first invocation produces plan files in --engine-dir specific to the accelerator being run on and are reused for later invocations. 
    ```
    python3 demo_txt2img_xl.py \
      ""Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"" \
      --version=xl-turbo \
      --onnx-dir /workspace/sdxl-turbo-tensorrt/ \
      --engine-dir /workspace/sdxl-turbo-tensorrt/engine \
      --denoising-steps 4 \
      --guidance-scale 0.0 \
      --seed 42 \
      --width 512 \
      --height 512
    ```
