import argparse
import torch
import torch.nn as nn
from data import get_loaders
from models import dit
import os
from engine import train_one_epoch, evaluate
import wandb
from diffusers import AutoencoderKL


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--img_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=64)

    parser.add_argument("--num_epochs", type=int, default=300)

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--T", type=int, default=200)

    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--wandb_run_id", type=str, default=None)

    return parser


def main(args):
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a compatible NVIDIA GPU."
    args.device = torch.device("cuda")

    assert args.batch_size % args.micro_batch_size == 0, "batch_size must be divisible by micro_batch_size"
    args.num_accum_steps = args.batch_size // args.micro_batch_size


    train_loader, val_loader = get_loaders(args)


    # the diffusion is performed in the latent space obtained from the VAE, which is of dimension (4, img_size/8, img_size/8)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(args.device)
    vae.eval()
    vae = torch.compile(vae)

    base_model = dit(args.img_size // 8, 4, args.num_classes).to(args.device)
    model = base_model
    model = torch.compile(base_model)

    print(f"Num params   vae: {(sum(p.numel() for p in vae.parameters())) / 1e6} M    model: {(sum(p.numel() for p in model.parameters())) / 1e6} M")


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)

    criterion = nn.MSELoss()

    scaler = torch.amp.GradScaler("cuda")


    # --- resume optimizer, lr scheduler, and scaler states if resuming from checkpoint ---
    start_epoch = 1

    if args.resume_from_checkpoint:
        assert args.wandb_run_id is not None, "wandb_run_id must be specified when resuming from checkpoint."
        assert os.path.exists(f"{args.output_dir}/{args.wandb_run_id}.pt"), f"Checkpoint path '{args.output_dir}/{args.wandb_run_id}.pt' does not exist."

        checkpoint = torch.load(f"{args.output_dir}/{args.wandb_run_id}.pt", map_location=args.device)
        base_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        del checkpoint # free up memory
        
        print(f"Resumed training from checkpoint: {args.output_dir}/{args.wandb_run_id}.pt")


    # --- initialize wandb ---
    config = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    run = wandb.init(
        project="image-generation",
        dir="./wandb_logs",
        config=config,
        mode="online",
        id=args.wandb_run_id if args.resume_from_checkpoint else None,
        resume="must" if args.resume_from_checkpoint else False
    )
    wandb.define_metric("epoch")
    for metric in ("train_loss", "lr", "val_loss", "sampled_imgs"):
        wandb.define_metric(metric, step_metric="epoch")


    # --- training loop ---
    for epoch in range(start_epoch, args.num_epochs+1):
        train_one_epoch(args, train_loader, vae, model, criterion, optimizer, scaler, epoch)
        evaluate(args, val_loader, vae, model, criterion, scaler, epoch)

        if epoch % 50 == 0 or epoch == args.num_epochs:
            model_to_save = model.module if hasattr(model, "module") else model # Unwrap the model if it's wrapped by DDP
            model_to_save = model_to_save._orig_mod if hasattr(model_to_save, "_orig_mod") else model_to_save # Unwrap the model if it's wrapped by torch.compile
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": config
            }
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(checkpoint, f"{args.output_dir}/{run.id}.pt")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
