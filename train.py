from argparse import Namespace

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from tqdm import tqdm

from data_utils import get_image_dataloaders
from models import BrainAgeCNN
from utils import AvgMeter, mean_absolute_error, seed_everything, TensorboardLogger


config = Namespace()
config.batch_size = 16
config.img_size = 96
config.num_workers = 0

config.lr = 1e-3
config.betas = (0.9, 0.999)

config.seed = 0
config.log_dir = './logs'
config.num_steps = 1500
config.val_freq = 50
config.log_freq = 10
config.device = 'cuda'

seed_everything(config.seed)


def train(config, model, optimizer, train_loader, val_loader, writer):
    model.train()
    step = 0
    pbar = tqdm(total=config.val_freq, desc='Training')
    avg_loss = AvgMeter()

    while True:
        for x, y in train_loader:
            x = x.to(config.device)
            y = y.to(config.device)
            pbar.update(1)

            # Training step
            optimizer.zero_grad()
            loss = model.train_step(x, y)
            loss.backward()
            optimizer.step()

            avg_loss.add(loss.detach().item())

            # Increment step
            step += 1

            if step % config.log_freq == 0 and not step % config.val_freq == 0:
                train_loss = avg_loss.compute()
                writer.log({'train/loss': train_loss}, step=step)

            # Validate and log at validation frequency
            if step % config.val_freq == 0:
                # Reset avg_loss
                train_loss = avg_loss.compute()
                avg_loss = AvgMeter()

                # Get validation results
                val_results = validate(
                    model,
                    val_loader,
                    config,
                )

                # Print current performance
                print(f"Finished step {step} of {config.num_steps}. "
                      f"Train loss: {train_loss} - "
                      f"val loss: {val_results['val/loss']:.4f} - "
                      f"val MAE: {val_results['val/MAE']:.4f}")

                # Write to tensorboard
                writer.log(val_results, step=step)

                # Reset progress bar
                pbar = tqdm(total=config.val_freq, desc='Training')

            if step >= config.num_steps:
                print(f'\nFinished training after {step} steps\n')
                return model, step


def validate(model, val_loader, config):
    model.eval()
    avg_val_loss = AvgMeter()
    preds = []
    targets = []
    for x, y in val_loader:
        x = x.to(config.device)
        y = y.to(config.device)

        with torch.no_grad():
            loss, pred = model.train_step(x, y, return_prediction=True)
        avg_val_loss.add(loss.item())
        preds.append(pred.cpu())
        targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    mae = mean_absolute_error(preds, targets)
    f = plot_results(preds, targets)
    model.train()
    return {
        'val/loss': avg_val_loss.compute(),
        'val/MAE': mae,
        'val/MAE_plot': f
    }


def plot_results(preds: Tensor, targets: Tensor):
    mae_test = mean_absolute_error(preds, targets)
    # Sort preds and targets to ascending targets
    sort_inds = targets.argsort()
    targets = targets[sort_inds].numpy()
    preds = preds[sort_inds].numpy()

    f = plt.figure()
    plt.plot(targets, targets, 'r.')
    plt.plot(targets, preds, '.')
    plt.plot(targets, targets + mae_test, 'gray')
    plt.plot(targets, targets - mae_test, 'gray')
    plt.suptitle('Mean Average Error')
    plt.xlabel('True Age')
    plt.ylabel('Age predicted')
    return f


if __name__ == '__main__':
    # Init model
    model = BrainAgeCNN().to(config.device)
    # Init optimizers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas
    )
    # Load data
    dataloaders = get_image_dataloaders(
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    # Init tensorboard
    writer = TensorboardLogger(config.log_dir, config)
    # Train
    model, step = train(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        writer=writer
    )
    # Test
    test_results = validate(model, dataloaders['test'], config)
    test_results = {k.replace('val', 'test'): v for k, v in test_results.items()}
    writer.log(test_results, step)
    print(f'Test loss: {test_results["test/loss"]:.4f}')
    print(f'Test MAE: {test_results["test/MAE"]:.4f}')
