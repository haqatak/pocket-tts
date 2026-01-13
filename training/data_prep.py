import logging
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional
import scipy.io.wavfile
import numpy as np

# Basic normalization map (very limited demo version)
# In a real scenario, use num2words or a better library.
NUMBER_MAP = {
    "0": "null", "1": "en", "2": "to", "3": "tre", "4": "fire",
    "5": "fem", "6": "seks", "7": "sju", "8": "åtte", "9": "ni",
    "10": "ti"
}

logger = logging.getLogger(__name__)

class SSCProcessor:
    def __init__(self, mimi_model, target_sample_rate: int = 24000, latents_mean: Optional[torch.Tensor] = None, latents_std: Optional[torch.Tensor] = None):
        self.mimi = mimi_model
        self.target_sample_rate = target_sample_rate
        self.latents_mean = latents_mean
        self.latents_std = latents_std

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Normalizes latents to standard Gaussian.
        """
        if self.latents_mean is not None and self.latents_std is not None:
            # Ensure stats are on correct device
            mean = self.latents_mean.to(latents.device)
            std = self.latents_std.to(latents.device)
            # Reshape for broadcasting if needed [C, 1] or [1, C, 1]
            # Latents are [B, C, T]
            if mean.ndim == 1:
                mean = mean.view(1, -1, 1)
                std = std.view(1, -1, 1)
            return (latents - mean) / std
        return latents

    def unnormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Restores latents from standard Gaussian.
        """
        if self.latents_mean is not None and self.latents_std is not None:
            mean = self.latents_mean.to(latents.device)
            std = self.latents_std.to(latents.device)
            if mean.ndim == 1:
                mean = mean.view(1, -1, 1)
                std = std.view(1, -1, 1)
            return latents * std + mean
        return latents

    def normalize_text(self, text: str) -> str:
        """
        Converts numbers, dates, abbreviations to full Norwegian text.
        This is a placeholder implementation.
        """
        words = text.split()
        normalized_words = []
        for word in words:
            if word in NUMBER_MAP:
                normalized_words.append(NUMBER_MAP[word])
            else:
                # Basic digit replacement
                new_word = ""
                for char in word:
                    if char.isdigit() and char in NUMBER_MAP:
                        new_word += NUMBER_MAP[char] + " "
                    else:
                        new_word += char
                normalized_words.append(new_word.strip())

        return " ".join(normalized_words)

    def process_audio(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads audio, resamples, encodes with Mimi to get latents.
        Returns (voice_prompt_latent, target_audio_latent).
        """
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Mix to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)

        # Mimi expects [B, C, T]
        waveform = waveform.unsqueeze(0)

        # Encode to latent
        # mimi.encode_to_latent returns [B, C, T] unquantized latents
        with torch.no_grad():
            latents = self.mimi.encode_to_latent(waveform)

        # Split into prompt (3-5s) and target
        # 24kHz audio. Mimi frame rate depends on model, but let's assume we split based on time.
        # mimi frame rate is sample_rate / hop_length.
        # We can just split the latents tensor.

        # Let's say we take first 3 seconds as prompt.
        # We need to know the frame rate of the latents.
        # mimi.frame_rate gives frames per second.

        frame_rate = self.mimi.frame_rate
        prompt_frames = int(3.0 * frame_rate)

        if latents.shape[-1] <= prompt_frames:
            # Audio too short, use everything as prompt and target (or handle error)
            # For training, we usually want separate prompt and target.
            # But here we might just return same for both if too short, or skip.
            return latents, latents

        prompt_latent = latents[..., :prompt_frames]
        target_latent = latents # Target is usually the whole thing or the rest?
        # The prompt says: (Prompt_Audio_Latent, Target_Text, Target_Audio_Latent)
        # Usually target audio is the full audio, conditioned on the prompt (which is part of it or different).
        # PocketTTS uses the prompt to set speaker style.

        # Apply normalization
        prompt_latent = self.normalize_latents(prompt_latent)
        target_latent = self.normalize_latents(target_latent)

        return prompt_latent, target_latent

    def create_dataset_item(self, audio_path: str, text: str):
        normalized_text = self.normalize_text(text)
        prompt_latent, target_latent = self.process_audio(audio_path)

        return {
            "prompt_latent": prompt_latent,
            "text": normalized_text,
            "target_latent": target_latent
        }

    def compute_stats(self, dataset_paths: list[str], max_samples: int = 1000):
        """
        Computes mean and std of latents from a subset of the dataset.
        """
        latents_list = []
        for i, path in enumerate(dataset_paths):
            if i >= max_samples: break
            # We just need raw latents before norm
            # So temporarily disable norm or access raw process

            # Since process_audio now calls normalize, we should be careful.
            # But process_audio uses self.latents_mean/std.
            # If they are None, it returns raw.
            # So this works if we call compute_stats before setting stats.

            p, t = self.process_audio(path)
            latents_list.append(t) # t is full audio latent

        if not latents_list:
            return None, None

        all_latents = torch.cat(latents_list, dim=-1) # [B, C, Total_T]
        # Calculate mean/std over Time and Batch
        # Shape [B, C, T] -> mean over [0, 2]
        mean = torch.mean(all_latents, dim=[0, 2])
        std = torch.std(all_latents, dim=[0, 2])

        return mean, std

def download_ssc(output_dir: str):
    """
    Dummy function to download SSC dataset.
    In reality, this would use wget or similar to download from specific URLs.
    """
    logger.info(f"Downloading SSC dataset to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Create a dummy file for testing
    dummy_wav = Path(output_dir) / "dummy.wav"
    sample_rate = 24000
    duration = 5 # seconds
    # Generate random noise wav
    data = np.random.uniform(-1, 1, sample_rate * duration)
    scipy.io.wavfile.write(str(dummy_wav), sample_rate, data.astype(np.float32))

    with open(Path(output_dir) / "metadata.txt", "w") as f:
        f.write("dummy.wav|Dette er en test setning med tallet 1945.")

    logger.info("Download complete (simulated).")

def download_npsc(output_dir: str):
    """Dummy function for NPSC."""
    logger.info(f"Downloading NPSC dataset to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
