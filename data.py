import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_loaders(args):

    args.mean, args.std = torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5])

    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(args.mean, args.std),
    ])

    train_dataset = datasets.ImageFolder(f"{args.data_dir}/train", transform=transform)
    val_dataset = datasets.ImageFolder(f"{args.data_dir}/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.micro_batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=args.micro_batch_size, shuffle=False, num_workers=1)

    print("Number of training images:", len(train_dataset), "Number of validation images:", len(val_dataset))

    return train_loader, val_loader
