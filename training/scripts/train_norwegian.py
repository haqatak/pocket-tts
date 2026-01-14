#!/usr/bin/env python3
"""
Norwegian TTS Training Script

This script trains a Norwegian TTS model by fine-tuning the pretrained pocket-tts model
on Norwegian speech datasets (SSC and NPSC).

Usage:
    uv run python -m training.scripts.train_norwegian --data_dir data/SSC --epochs 10

Steps:
1. Download Norwegian datasets (SSC/NPSC)
2. Optionally patch tokenizer for Norwegian characters
3. Compute latent normalization statistics
4. Train with consistency distillation
"""

import argparse
import logging
import torch
import copy
from pathlib import Path

from pocket_tts.models.tts_model import TTSModel
from training.data_prep import SSCProcessor, download_ssc
from training.train import SSCDataset, train_one_epoch, update_ema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_dataset_from_metadata(metadata_path: Path) -> list[tuple[str, str]]:
    """
    Load dataset from metadata file.
    Expected format: audio_filename|transcript
    """
    data_list = []
    data_dir = metadata_path.parent

    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return data_list

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|", 1)
            if len(parts) == 2:
                audio_file, text = parts
                audio_path = data_dir / audio_file
                if audio_path.exists():
                    data_list.append((str(audio_path), text))
                else:
                    logger.warning(f"Audio file not found: {audio_path}")

    logger.info(f"Loaded {len(data_list)} samples from {metadata_path}")
    return data_list


def main():
    parser = argparse.ArgumentParser(description="Train Norwegian TTS model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/SSC",
        help="Directory containing the dataset (default: data/SSC)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/norwegian",
        help="Output directory for checkpoints (default: checkpoints/norwegian)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (default: 1, only 1 supported currently)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--head_batch_multiplier",
        type=int,
        default=8,
        help="Head batch multiplier for HBM (default: 8)"
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone and only train flow head"
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA model for consistency distillation"
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay rate (default: 0.999)"
    )
    parser.add_argument(
        "--download_data",
        action="store_true",
        help="Download SSC dataset before training"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on (default: cpu)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download data if requested
    if args.download_data:
        logger.info("Downloading SSC dataset...")
        download_ssc(args.data_dir)

    # Load pretrained model using the class method
    logger.info("Loading pretrained pocket-tts model...")
    model = TTSModel.load_model()  # Uses default variant and parameters
    device = torch.device(args.device)
    model.flow_lm.to(device)
    model.mimi.to(device)

    # Create EMA model if requested
    ema_model = None
    if args.use_ema:
        logger.info("Creating EMA model for consistency distillation...")
        ema_model = TTSModel.load_model()
        ema_model.flow_lm.to(device)
        ema_model.mimi.to(device)
        # Copy weights
        ema_model.flow_lm.load_state_dict(model.flow_lm.state_dict())

    # Create data processor
    logger.info("Initializing data processor...")
    processor = SSCProcessor(
        mimi_model=model.mimi,
        target_sample_rate=24000
    )

    # Load dataset
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / "metadata.txt"

    data_list = load_dataset_from_metadata(metadata_path)

    if not data_list:
        logger.error(f"No data found in {args.data_dir}. Make sure metadata.txt exists with format: audio.wav|transcript")
        logger.info("You can download sample data with --download_data flag")
        return

    # Compute normalization statistics
    logger.info("Computing latent normalization statistics...")
    audio_paths = [item[0] for item in data_list[:min(100, len(data_list))]]
    mean, std = processor.compute_stats(audio_paths, max_samples=100)
    if mean is not None:
        processor.latents_mean = mean
        processor.latents_std = std
        logger.info(f"Latent stats - Mean shape: {mean.shape}, Std shape: {std.shape}")

    # Create dataset and dataloader
    dataset = SSCDataset(data_list, processor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False
    )

    # Setup optimizer
    if args.freeze_backbone:
        # Only optimize flow head
        params = list(model.flow_lm.flow_net.parameters())
        logger.info(f"Training only flow head: {sum(p.numel() for p in params)} parameters")
    else:
        params = list(model.flow_lm.parameters())
        logger.info(f"Training full model: {sum(p.numel() for p in params)} parameters")

    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.flow_lm.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        if ema_model and "ema_state_dict" in checkpoint:
            ema_model.flow_lm.load_state_dict(checkpoint["ema_state_dict"])

    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Freeze backbone: {args.freeze_backbone}")
    logger.info(f"Use EMA: {args.use_ema}")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            head_batch_multiplier=args.head_batch_multiplier,
            ema_model=ema_model,
            freeze_backbone=args.freeze_backbone
        )

        logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.flow_lm.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "args": vars(args)
            }
            if ema_model:
                save_dict["ema_state_dict"] = ema_model.flow_lm.state_dict()

            torch.save(save_dict, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_path = output_dir / "norwegian_tts_final.pt"
    torch.save({
        "model_state_dict": model.flow_lm.state_dict(),
        "ema_state_dict": ema_model.flow_lm.state_dict() if ema_model else None,
        "args": vars(args)
    }, final_path)
    logger.info(f"Training complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
