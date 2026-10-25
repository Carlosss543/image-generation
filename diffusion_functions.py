import torch


def noise_imgs(args, x1, t):
    t = t.view(-1, 1, 1, 1)
    x0 = torch.randn_like(x1, device=args.device)
    xt = (1 - t) * x0 + t * x1
    return xt, x0


@torch.no_grad()
def sample(args, n_imgs, model, labels=None):
    model.eval()
    xt = torch.randn((n_imgs, 4, 32, 32), device=args.device)
    dt = 1 / args.T
    for i in range(args.T):
        t = torch.full((n_imgs,), i * dt, device=args.device)
        xt = xt + dt * model(xt, t, labels)
    return xt
