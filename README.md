# AI Image & Video Generation with Stable Diffusion and Zeroscope

This project demonstrates how to generate images using Stable Diffusion v1.5 and videos using the Zeroscope v2 model. The code is designed to run in Google Colab or similar environments with GPU support.

## Features

- **Image Generation**: Create high-quality images from text prompts using Stable Diffusion v1.5
- **Video Generation**: Generate short video clips from text descriptions using Zeroscope v2
- **GPU Optimization**: Automatically detects and utilizes GPU acceleration when available
- **Safety Checker Disabled**: Uses an uncensored version for maximum creative freedom

## Requirements

### Python Libraries
```bash
pip install --quiet diffusers transformers accelerate safetensors gradio einops
```

### System Requirements
- **GPU**: CUDA-compatible GPU recommended for faster generation
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: At least 10GB free space for model downloads

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install --quiet diffusers transformers accelerate safetensors gradio einops
   ```
3. Run the code in a Jupyter notebook or Google Colab environment

## Usage

### Image Generation

```python
import torch
from diffusers import StableDiffusionPipeline

# Setup device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
    safety_checker=None
).to(device)

# Generate image
prompt = "sports trophy"
image = pipe(prompt).images[0]
image.save("output.png")
```

### Video Generation

```python
from diffusers import DiffusionPipeline
import torch

# Load Zeroscope pipeline
pipe = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    torch_dtype=torch.float32
).to("cpu")

# Generate video
prompt = "flying"
video_frames = pipe(prompt, num_inference_steps=25).frames

# Save as MP4
import numpy as np
import imageio

video_frames_np = video_frames[0]
video_frames_np = (video_frames_np * 255).astype(np.uint8)
imageio.mimsave("generated_video.mp4", video_frames_np, fps=8)
```

## Models Used

### Stable Diffusion v1.5
- **Model**: `runwayml/stable-diffusion-v1-5`
- **Purpose**: Text-to-image generation
- **Resolution**: 512x512 pixels
- **Safety Checker**: Disabled for uncensored output

### Zeroscope v2
- **Model**: `cerspense/zeroscope_v2_576w`
- **Purpose**: Text-to-video generation
- **Resolution**: 576x320 pixels
- **Duration**: ~2 seconds (16 frames at 8 FPS)

## Configuration Options

### Image Generation Parameters
- **Prompt**: Text description of desired image
- **Guidance Scale**: Controls adherence to prompt (default: 7.5)
- **Inference Steps**: Number of denoising steps (default: 50)
- **Resolution**: Output image size (default: 512x512)

### Video Generation Parameters
- **Prompt**: Text description of desired video
- **Inference Steps**: Number of denoising steps (recommended: 25)
- **FPS**: Frames per second for output video (default: 8)
- **Duration**: Fixed at 16 frames (~2 seconds)

## Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster generation
2. **Memory Management**: Close unused models to free GPU memory
3. **Batch Processing**: Generate multiple images/videos in batches
4. **Prompt Engineering**: Use detailed, specific prompts for better results

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce batch size or use CPU fallback
- Clear GPU cache: `torch.cuda.empty_cache()`

**Model Download Errors**
- Check internet connection
- Verify model names and availability
- Use alternative mirror if needed

**Video Display Issues**
- Ensure video codec compatibility
- Try different FPS settings
- Check file permissions

### Performance Optimization

```python
# Enable memory efficient attention
pipe.enable_attention_slicing()

# Use CPU offloading for large models
pipe.enable_sequential_cpu_offload()

# Enable xformers for faster attention (if installed)
pipe.enable_xformers_memory_efficient_attention()
```

## Output Formats

- **Images**: PNG format, 512x512 resolution
- **Videos**: MP4 format, 576x320 resolution, 8 FPS

## Legal and Ethical Considerations

- This implementation disables safety checkers
- Users are responsible for ensuring generated content complies with applicable laws
- Respect copyright and intellectual property rights
- Consider the ethical implications of generated content

## Example Prompts

### Image Prompts
- "a futuristic cityscape at sunset"
- "portrait of a person in renaissance style"
- "abstract geometric patterns in blue and gold"
- "a cozy coffee shop interior"

### Video Prompts
- "waves crashing on a beach"
- "a bird flying through clouds"
- "fire burning in a fireplace"
- "raindrops falling on leaves"

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## License

This project is provided as-is for educational and research purposes. Please respect the licenses of the underlying models and libraries.

## Acknowledgments

- Hugging Face Diffusers team for the excellent library
- RunwayML for Stable Diffusion v1.5
- Cerspense for the Zeroscope v2 model
- The open-source AI community for continuous innovation
