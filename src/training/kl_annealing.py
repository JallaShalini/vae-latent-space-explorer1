"""KL annealing schedule for VAE training."""


def get_beta(epoch: int, annealing_epochs: int = 20) -> float:
    if annealing_epochs <= 0:
        return 1.0
    return min(1.0, epoch / annealing_epochs)
