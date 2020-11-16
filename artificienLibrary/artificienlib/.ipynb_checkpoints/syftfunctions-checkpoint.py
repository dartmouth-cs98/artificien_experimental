def mse_with_logits(logits, targets, batch_size):
     """ Calculates mse
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    return (logits - targets).sum() / batch_size




