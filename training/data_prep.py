import logging
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional
import scipy.io.wavfile
import numpy as np

from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio

# Norwegian number and text normalization
# Extended number map for Norwegian
NUMBER_MAP = {
    "0": "null", "1": "en", "2": "to", "3": "tre", "4": "fire",
    "5": "fem", "6": "seks", "7": "sju", "8": "åtte", "9": "ni",
    "10": "ti", "11": "elleve", "12": "tolv", "13": "tretten",
    "14": "fjorten", "15": "femten", "16": "seksten", "17": "sytten",
    "18": "atten", "19": "nitten", "20": "tjue", "30": "tretti",
    "40": "førti", "50": "femti", "60": "seksti", "70": "sytti",
    "80": "åtti", "90": "nitti", "100": "hundre", "1000": "tusen"
}

# Common Norwegian abbreviations
ABBREVIATION_MAP = {
    "f.eks.": "for eksempel",
    "bl.a.": "blant annet",
    "o.l.": "og lignende",
    "m.m.": "med mer",
    "osv.": "og så videre",
    "ca.": "cirka",
    "nr.": "nummer",
    "kr.": "kroner",
    "kl.": "klokken",
    "mrd.": "milliarder",
    "mill.": "millioner",
    "pst.": "prosent",
    "pkt.": "punkt",
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
        This is a basic implementation - consider using num2words for production.
        """
        # First, expand abbreviations
        for abbrev, expansion in ABBREVIATION_MAP.items():
            text = text.replace(abbrev, expansion)

        words = text.split()
        normalized_words = []
        for word in words:
            # Check if whole word is a number in our map
            if word in NUMBER_MAP:
                normalized_words.append(NUMBER_MAP[word])
            else:
                # Handle multi-digit numbers (basic)
                if word.isdigit():
                    # Try to convert simple numbers
                    try:
                        num = int(word)
                        if num in range(21, 100) and num not in NUMBER_MAP:
                            tens = (num // 10) * 10
                            ones = num % 10
                            if tens in [20, 30, 40, 50, 60, 70, 80, 90]:
                                tens_word = NUMBER_MAP.get(str(tens), str(tens))
                                if ones > 0:
                                    ones_word = NUMBER_MAP.get(str(ones), str(ones))
                                    normalized_words.append(f"{tens_word}{ones_word}")
                                else:
                                    normalized_words.append(tens_word)
                            else:
                                normalized_words.append(word)
                        elif str(num) in NUMBER_MAP:
                            normalized_words.append(NUMBER_MAP[str(num)])
                        else:
                            # Read digit by digit for large numbers
                            digit_words = [NUMBER_MAP.get(d, d) for d in word]
                            normalized_words.append(" ".join(digit_words))
                    except ValueError:
                        normalized_words.append(word)
                else:
                    normalized_words.append(word)

        return " ".join(normalized_words)

    def process_audio(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads audio, resamples, encodes with Mimi to get latents.
        Returns (voice_prompt_latent, target_audio_latent).
        """
        # Load audio using project's audio_read (uses wave module, no torchcodec dependency)
        waveform, sample_rate = audio_read(audio_path)

        # Mix to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample using project's convert_audio utility
        waveform = convert_audio(waveform, sample_rate, self.target_sample_rate, 1)

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

        # Squeeze the batch dimension since dataloader will add it back
        # prompt_latent and target_latent are [1, C, T] -> [C, T]
        prompt_latent = prompt_latent.squeeze(0)
        target_latent = target_latent.squeeze(0)

        return {
            "prompt_latent": prompt_latent,  # [C, T_prompt]
            "text": normalized_text,
            "target_latent": target_latent   # [C, T_target]
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

def download_ssc(output_dir: str, create_dummy: bool = True):
    """
    Download NB Tale (Norwegian Speech Database) from Språkbanken.

    The NB Tale dataset contains:
    - 380 speakers from 24 dialect areas
    - Manuscript-read speech and spontaneous speech
    - CC-ZERO license

    Download URLs (manual download required for full dataset):
    - https://www.nb.no/sbfil/talegjenkjenning/lingit/sennheiser_1.tar.gz
    - https://www.nb.no/sbfil/talegjenkjenning/lingit/sennheiser_2.tar.gz
    - https://www.nb.no/sbfil/talegjenkjenning/lingit/sennheiser_3.tar.gz
    - https://www.nb.no/sbfil/talegjenkjenning/lingit/Annotation.zip

    Args:
        output_dir: Directory to save the dataset
        create_dummy: If True, create dummy data for testing
    """
    logger.info(f"Setting up SSC/NB Tale dataset in {output_dir}...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if real data exists
    wav_files = list(output_path.glob("**/*.wav"))
    if wav_files:
        logger.info(f"Found {len(wav_files)} existing WAV files in {output_dir}")
        return

    if create_dummy:
        logger.info("Creating dummy data for testing...")
        # Create dummy files for testing the pipeline
        sample_rate = 24000

        test_sentences = [
            ("test_001.wav", "Hei, jeg heter Ola og kommer fra Oslo."),
            ("test_002.wav", "Det er fint vær i dag."),
            ("test_003.wav", "Jeg liker å lese norske bøker."),
            ("test_004.wav", "Takk for maten, det var veldig godt."),
            ("test_005.wav", "Kan du hjelpe meg med dette?"),
        ]

        for wav_name, text in test_sentences:
            dummy_wav = output_path / wav_name
            duration = 5  # seconds
            # Generate random noise wav (placeholder) - use int16 for PCM format
            data = np.random.uniform(-0.1, 0.1, sample_rate * duration)
            # Convert to int16 for proper WAV PCM format
            data_int16 = (data * 32767).astype(np.int16)
            scipy.io.wavfile.write(str(dummy_wav), sample_rate, data_int16)

        # Create metadata file
        with open(output_path / "metadata.txt", "w", encoding="utf-8") as f:
            for wav_name, text in test_sentences:
                f.write(f"{wav_name}|{text}\n")

        logger.info(f"Created {len(test_sentences)} dummy samples in {output_dir}")
        logger.info("For real training, download NB Tale from: https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-31/")
    else:
        logger.info("To download NB Tale dataset manually:")
        logger.info("1. Visit: https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-31/")
        logger.info("2. Download sennheiser_*.tar.gz files")
        logger.info("3. Extract to: " + str(output_path))
        logger.info("4. Create metadata.txt with format: filename.wav|transcript")


def download_npsc(output_dir: str, create_dummy: bool = True):
    """
    Download NPSC (Norwegian Parliamentary Speech Corpus).

    NPSC contains transcribed speech from the Norwegian Parliament (Stortinget).
    Available from: https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-58/

    Args:
        output_dir: Directory to save the dataset
        create_dummy: If True, create dummy data for testing
    """
    logger.info(f"Setting up NPSC dataset in {output_dir}...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if real data exists
    wav_files = list(output_path.glob("**/*.wav"))
    if wav_files:
        logger.info(f"Found {len(wav_files)} existing WAV files in {output_dir}")
        return

    if create_dummy:
        logger.info("Creating dummy NPSC data for testing...")
        sample_rate = 24000

        test_sentences = [
            ("npsc_001.wav", "Stortinget vedtok i går en ny lov."),
            ("npsc_002.wav", "Statsministeren holdt en tale om økonomien."),
            ("npsc_003.wav", "Opposisjonen fremmet et nytt forslag."),
        ]

        for wav_name, text in test_sentences:
            dummy_wav = output_path / wav_name
            duration = 5
            data = np.random.uniform(-0.1, 0.1, sample_rate * duration)
            # Convert to int16 for proper WAV PCM format
            data_int16 = (data * 32767).astype(np.int16)
            scipy.io.wavfile.write(str(dummy_wav), sample_rate, data_int16)

        with open(output_path / "metadata.txt", "w", encoding="utf-8") as f:
            for wav_name, text in test_sentences:
                f.write(f"{wav_name}|{text}\n")

        logger.info(f"Created {len(test_sentences)} dummy NPSC samples")
        logger.info("For real training, download NPSC from: https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-58/")
    else:
        logger.info("To download NPSC dataset manually:")
        logger.info("1. Visit: https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-58/")
        logger.info("2. Download the audio and transcript files")
        logger.info("3. Extract to: " + str(output_path))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
