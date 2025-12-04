import numpy as np

from mold.networks.base_model import BaseModel


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience: int = 1,
        verbose: bool = False,
        delta: float = 0.0,
    ) -> None:
        """
        Args:
            patience: How long to wait after last time validation loss improved.
            verbose: If True, prints a message for each validation loss improvement.
            delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_max = -np.inf
        self.delta = delta

    def __call__(self, score: float, model: BaseModel) -> None:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score: float, model: BaseModel) -> None:
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation accuracy increased ({self.score_max:.6f} --> {score:.6f}).  Saving model ..."
            )
        model.save_networks("earlystop_best.pth")
        self.score_max = score
