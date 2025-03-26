import torch
import cv2
import numpy as np
import einops
from PIL import Image
import logging
from pathlib import Path
from typing import List
import requests
import time

from ..core.config import settings
from ..models.generation import GenerationParams, JobStatus
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

logger = logging.getLogger(__name__)


class ControlNetService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not settings.FORCE_CPU else "cpu")
        logger.info(f"Using device: {self.device}")

        self.apply_canny = CannyDetector()

        logger.info("Initializing ControlNet model...")
        self.model = self._initialize_model()
        self.ddim_sampler = DDIMSampler(self.model)
        logger.info("Model initialized successfully")

    def _initialize_model(self):
        model = create_model(settings.MODEL_CONFIG).cpu()
        model.load_state_dict(load_state_dict(
            settings.MODEL_PATH,
            location='cuda' if torch.cuda.is_available() and not settings.FORCE_CPU else 'cpu'
        ))
        return model.to(self.device)

    async def process_image(
        self,
        job_id: str,
        params: GenerationParams
    ) -> List[str]:
        """
        Process an image with the given parameters and return paths to generated images.
        """
        try:
            # Setup paths
            job_dir = settings.JOBS_DIR / job_id
            input_path = job_dir / "input.png"

            # Read and preprocess input image
            input_image = self._load_image(str(input_path))

            # Generate images
            results = self._generate(input_image, params)

            # Save results
            result_paths = []
            for idx, result in enumerate(results):
                output_path = job_dir / f"result_{idx}.png"
                Image.fromarray(result).save(output_path)
                result_paths.append(f"result_{idx}.png")

            return result_paths

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess input image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load input image")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _generate(
        self,
        input_image: np.ndarray,
        params: GenerationParams
    ) -> List[np.ndarray]:
        """Generate images using ControlNet."""
        with torch.no_grad():
            # Resize input image
            img = resize_image(HWC3(input_image), params.image_resolution)
            H, W, C = img.shape

            # Apply Canny edge detection
            detected_map = self.apply_canny(img, params.low_threshold, params.high_threshold)
            detected_map = HWC3(detected_map)

            # Prepare control signal
            control = torch.from_numpy(detected_map.copy()).float().to(self.device) / 255.0
            control = torch.stack([control for _ in range(params.num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            # Set random seed
            if params.seed != -1:
                torch.manual_seed(params.seed)
                np.random.seed(params.seed)

            # Prepare conditioning
            cond = {
                "c_concat": [control],
                "c_crossattn": [
                    self.model.get_learned_conditioning(
                        [f"{params.prompt}, {params.a_prompt}"] * params.num_samples
                    )
                ]
            }
            un_cond = {
                "c_concat": [control],
                "c_crossattn": [
                    self.model.get_learned_conditioning(
                        [params.n_prompt] * params.num_samples
                    )
                ]
            }

            shape = (4, H // 8, W // 8)
            samples, _ = self.ddim_sampler.sample(
                params.ddim_steps,
                params.num_samples,
                shape,
                cond,
                verbose=False,
                eta=params.eta,
                unconditional_guidance_scale=params.scale,
                unconditional_conditioning=un_cond
            )

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (
                einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
            ).cpu().numpy().clip(0, 255).astype(np.uint8)

            return [x_samples[i] for i in range(params.num_samples)]


def generate_image(image_path: str, prompt: str):
    """
    Generate an image using the ControlNet API.
    """
    BASE_URL = settings.API_BASE_URL

    try:
        # 1. Upload image
        with open(image_path, "rb") as f:
            response = requests.post(f"{BASE_URL}/upload/", files={"file": f})
            response.raise_for_status()
            job_id = response.json()["job_id"]

        # 2. Start generation
        params = {
            "prompt": prompt,
            "num_samples": 1,
            "image_resolution": 512,
            "strength": 1.0,
            "ddim_steps": 20
        }
        response = requests.post(f"{BASE_URL}/{job_id}/generate/", json=params)
        response.raise_for_status()

        # 3. Poll for results
        MAX_RETRIES = 60
        retries = 0

        while retries < MAX_RETRIES:
            response = requests.get(f"{BASE_URL}/{job_id}/status/")
            data = response.json()

            if data["status"] == "completed":
                for image_name in data["images"]:
                    image_url = f"{BASE_URL}/{job_id}/result/{image_name}"
                    response = requests.get(image_url)
                    response.raise_for_status()

                    output_path = Path(f"output_{image_name}")
                    output_path.write_bytes(response.content)
                    print(f"Saved result to {output_path}")
                break
            elif data["status"] == "failed":
                raise Exception("Generation failed")

            retries += 1
            time.sleep(1)

        if retries == MAX_RETRIES:
            raise TimeoutError("Generation process timed out")

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in generate_image: {e}")
        raise