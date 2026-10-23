import torch
from diffusion_functions import noise_imgs, sample
import wandb
import time


def train_one_epoch(args, train_loader, vae, model, criterion, optimizer, scaler, epoch, log_interval=10):
    model.train()

    total_loss = torch.tensor(0.0, device=args.device)
    total_samples = torch.tensor(0, device=args.device)

    # torch.cuda.synchronize(args.device)
    # t0 = time.perf_counter()
    for i, (imgs, labels) in enumerate(train_loader):
        # t_data = time.perf_counter()

        imgs, labels = imgs.to(args.device), labels.to(args.device)

        is_accumulation_end = (i+1) % args.num_accum_steps == 0 or (i+1) == len(train_loader)

        # torch.cuda.synchronize(args.device)
        # t_transfer = time.perf_counter()

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            posterior = vae.encode(imgs)
        x1 = posterior.latent_dist.sample() * vae.config.scaling_factor

        # torch.cuda.synchronize(args.device)
        # t_vae = time.perf_counter()

        t = torch.rand(x1.shape[0], device=args.device)
        xt, x0 = noise_imgs(args, x1, t)
        targets = x1 - x0

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            preds = model(xt, t, labels)
            loss = criterion(preds, targets) / args.num_accum_steps
        scaler.scale(loss).backward()

        if is_accumulation_end:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.detach() * x1.size(0) * args.num_accum_steps
        total_samples += x1.size(0)

        # torch.cuda.synchronize(args.device)
        # t_model = time.perf_counter()
        # print(
        #     f"Batch {i+1}/{len(train_loader)} - Loss: {loss.detach().item():.4f}"
        #     f" - Total: {t_model - t0:.3f}s"
        #     f" - Data wait (CPU): {t_data - t0:.3f}s"
        #     f" - Transfer (CPU->GPU): {t_transfer - t_data:.3f}s"
        #     f" - VAE: {t_vae - t_transfer:.3f}s"
        #     f" - Model step (noise+forward+backward+optim): {t_model - t_vae:.3f}s"
        # )

        # send batch loss to W&B occasionally
        if (i+1) % log_interval == 0:
            wandb.log({"batch_train_loss": loss.detach().item() * args.num_accum_steps})

        # t0 = time.perf_counter()

    avg_loss = total_loss / total_samples

    wandb.log({"epoch": epoch, "train_loss": avg_loss.item(), "lr": optimizer.param_groups[0]["lr"]})


@torch.no_grad()
def evaluate(args, val_loader, vae, model, criterion, scaler, epoch):
    model.eval()

    total_loss = torch.tensor(0.0, device=args.device)
    total_samples = torch.tensor(0, device=args.device)

    for imgs, labels in val_loader:
        imgs, labels = imgs.to(args.device), labels.to(args.device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            posterior = vae.encode(imgs)
        x1 = posterior.latent_dist.sample() * vae.config.scaling_factor

        t = torch.rand(x1.shape[0], device=args.device)
        xt, x0 = noise_imgs(args, x1, t)
        targets = x1 - x0

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            preds = model(xt, t, labels)
            loss = criterion(preds, targets)

        total_loss += loss.detach() * x1.size(0)
        total_samples += x1.size(0)

    avg_loss = total_loss / total_samples

    sampled_imgs = sample(args, 3, model, torch.tensor([0, 1, 2], device=args.device)) / vae.config.scaling_factor
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        sampled_imgs = vae.decode(sampled_imgs).sample
    sampled_imgs = sampled_imgs * args.std.view(1, -1, 1, 1).to(args.device) + args.mean.view(1, -1, 1, 1).to(args.device)

    wandb.log({"epoch": epoch, "val_loss": avg_loss.item(), "sampled_imgs": [wandb.Image(img.detach().cpu()*255) for img in sampled_imgs]})
