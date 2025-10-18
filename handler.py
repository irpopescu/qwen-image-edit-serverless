import base64
import io
import torch
from PIL import Image
from diffusers import DiffusionPipeline
import runpod

# === Încarcă modelul la runtime ===
print("🚀 Loading Qwen Image model from Hugging Face...")
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype=torch.float16
).to("cuda")
print("✅ Model ready!")

# === Funcția principală ===
def generate_image(job):
    """Primește prompt și imagine base64, returnează imagine editată."""
    try:
        inputs = job["input"]
        prompt = inputs.get("prompt", "photo of an object on white background")
        image_b64 = inputs.get("image_b64")

        if not image_b64:
            return {"error": "Missing image_b64 input"}

        # Decodăm imaginea de intrare
        image_bytes = base64.b64decode(image_b64)
        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Generăm imaginea nouă
        result = pipe(prompt=prompt, image=init_image, strength=0.25, guidance_scale=4.5)
        edited_image = result.images[0]

        # Codăm rezultatul în base64
        output_buffer = io.BytesIO()
        edited_image.save(output_buffer, format="PNG")
        output_b64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        return {"image_b64": output_b64}

    except Exception as e:
        return {"error": str(e)}

# === Pornește endpointul RunPod ===
runpod.serverless.start({"handler": generate_image})
