#!/usr/bin/env python3
import csv
import os
import time
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from mold.data import create_dataloader
from mold.earlystop import EarlyStopping
from mold.networks.trainer import Trainer
from mold.options.train_options import TrainOptions
from mold.utils import set_seed
from mold.validation import validate

try:
    import wandb
except ImportError:
    print("wandb not installed, skipping wandb logging.")
    wandb = None


"""Currently assumes jpg_prob, blur_prob 0 or 1"""


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.is_train = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = "val"
    val_opt.jpg_method = ["pil"]
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


def main():
    opt = TrainOptions().parse()
    val_opt = get_val_opt()

    set_seed(opt.seed)

    trainer = Trainer(opt)
    if wandb is not None:
        wandb.init(project="fakedet")
        wandb.watch(trainer.model, log="all")

    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)
    print(f"Length of data loader: {len(data_loader)}")

    output_dir = os.path.join(opt.checkpoints_dir, opt.expname)
    train_writer = SummaryWriter(os.path.join(output_dir, "train"))
    val_writer = SummaryWriter(os.path.join(output_dir, "val"))
    early_stopping = EarlyStopping(patience=5, delta=-0.001, verbose=True)

    loss_file = os.path.join(output_dir, "loss.csv")
    with open(loss_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])

    start_time = time.time()
    for epoch in range(opt.niter):
        for i, data in enumerate(data_loader):
            trainer.total_steps += 1

            trainer.set_input(data)
            trainer.optimize_parameters()

            if trainer.total_steps % opt.loss_freq == 0:
                now = datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                iter_time = (time.time() - start_time) / (trainer.total_steps + 1e-8)
                print(
                    f"[{current_time}]",
                    f"Train loss: {trainer.loss:.4f}",
                    f"at step: {trainer.total_steps}",
                    f"iter time: {iter_time:.4f} sec",
                )
                train_writer.add_scalar("loss", trainer.loss, trainer.total_steps)

            if (
                trainer.total_steps in [10, 30, 50, 100, 1000, 5000, 10000] and False
            ):  # save models at these iters
                trainer.save_networks(f"model_iters_{trainer.total_steps}.pth")

        if epoch % opt.save_epoch_freq == 0:
            print(f"saving the model at the end of epoch {epoch}")
            trainer.save_networks("model_epoch_best.pth")
            trainer.save_networks(f"model_epoch_{epoch}.pth")

        with open(loss_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trainer.total_steps, trainer.loss.item()])

        # Validation.
        trainer.eval()
        ap, r_acc, f_acc, acc = validate(trainer.model, val_loader)
        val_writer.add_scalar("accuracy", acc, trainer.total_steps)
        val_writer.add_scalar("ap", ap, trainer.total_steps)
        print(f"(Val @ epoch {epoch}) acc: {acc}; ap: {ap}")

        early_stopping(acc, trainer)
        if early_stopping.early_stop:
            cont_train = trainer.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=5, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        trainer.train()
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
