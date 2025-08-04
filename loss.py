import torch
import torch.nn.functional as F


def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    # Compute unreduced CE loss
    loss_1 = F.cross_entropy(y_1, t, reduction="none")
    loss_2 = F.cross_entropy(y_2, t, reduction="none")

    # Sort indices by loss (ascending)
    ind_1_sorted = torch.argsort(loss_1, dim=0)
    ind_2_sorted = torch.argsort(loss_2, dim=0)

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1))

    # Compute pure ratios using numpy indexing
    ind_np = ind  # already a numpy array
    ind_1_np = ind_1_sorted[:num_remember].cpu().numpy()
    ind_2_np = ind_2_sorted[:num_remember].cpu().numpy()

    pure_ratio_1 = noise_or_not[ind_np[ind_1_np]].sum() / float(num_remember)
    pure_ratio_2 = noise_or_not[ind_np[ind_2_np]].sum() / float(num_remember)

    # Select samples to use
    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return (
        loss_1_update,
        loss_2_update,
        pure_ratio_1,
        pure_ratio_2,
    )
