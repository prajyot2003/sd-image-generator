import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Load the model (first time, downloads ~4GB weights)
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# define generation function
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# build gradio interface
gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter a text prompt", placeholder="e.g., a cat reading a newspaper"),
    outputs="image",
    title="🎨 Text to Image Generator",
    description="Uses Stable Diffusion to turn text prompts into images"
).launch()