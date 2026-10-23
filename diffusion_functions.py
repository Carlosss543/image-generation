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
    for t in torch.linspace(0, 1, args.T):
        t = t.expand(xt.shape[0]).view(-1, 1).to(args.device)
        xt = xt + 1/args.T * model(xt, t, labels)
    return xt
