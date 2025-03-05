import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from diffusers import StableDiffusion3Pipeline
import torch
from io import BytesIO
import base64

class InferlessPythonModel:
    def initialize(self):
        model_id = "stabilityai/stable-diffusion-3.5-large"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    def infer(self, inputs):
        prompt = inputs["prompt"]
        negative_prompt = inputs["negative_prompt"]
        inference_steps = inputs["num_inference_steps"]
        guidance_scale = inputs["guidance_scale"]

        image = self.pipe(prompt,negative_prompt=negative_prompt,num_inference_steps=inference_steps,guidance_scale=guidance_scale).images[0]
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()

        return {"generated_image_base64" : img_str }

    def finalize(self):
        self.pipe = None
