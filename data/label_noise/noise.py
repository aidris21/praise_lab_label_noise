import numpy as np

def matrix_to_array(M) -> np.ndarray:
    """
    see https://stackoverflow.com/questions/3337301/numpy-matrix-to-array
    """
    return np.squeeze(np.asarray(M))

def add_noise_to_labels(labels: np.ndarray, noise_rates) -> np.ndarray:
    """
    Returns a numpy array with noise added to the given label array. If more than two classes are present
    in the labels, the probability that an output label is any class other than its true class given that
    it has been flipped is will be uniform among all classes.

    Parameters:
                    labels (np.ndarray): An array of observed labels
                    noise_rates (dict): A dictionary specifying a mapping from the unique label values
                    to the noise rates for that value. If any values present in `labels` is missing
                    from the dict, a noise rate of 0 will be assumed for those values.
    """

    label_classes, label_encoding = np.unique(labels, return_inverse=True)
    n = labels.size
    m = label_classes.size
    flip_series_list = []
    for label_class in label_classes:
        # Derive indicator for given class that the true label will flip
        noise_rate = noise_rates[label_class] if label_class in noise_rates else 0
        label_indicator = (labels == label_class).astype(int)
        flip_indicator = np.multiply(np.random.binomial(1, noise_rate, n), label_indicator)
        flip_series_list.append(flip_indicator)


    flip_indicator_array: np.ndarray = matrix_to_array(
        np.matmul(np.matrix(flip_series_list).transpose(), np.ones(m))
    )
    flip_amount_array: np.ndarray = np.multiply(np.random.randint(low=1, high=m, size=n), flip_indicator_array)

    noisy_label_encoding = np.mod(np.add(label_encoding, flip_amount_array), m)

    return label_classes[noisy_label_encoding.astype(int)]

    




