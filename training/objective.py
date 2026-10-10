def combine_training_losses(
    dti_bce,
    auc_loss,
    memory_loss,
    bs_loss,
    auc_weight=0.75,
    memory_weight=0.03,
    bs_weight=0.10,
):
    """Combine the public training-loss terms.

    The implementation of the score-memory module and the private interaction
    head are not included in this release. Their scalar losses can be passed
    into this helper.
    """
    return (
        dti_bce
        + auc_weight * auc_loss
        + memory_weight * memory_loss
        + bs_weight * bs_loss
    )
