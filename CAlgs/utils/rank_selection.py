import numpy as np


def estimate_rank_for_compression_rate(tensor_shape, rate=2):
    cout, cin, kh, kw = tensor_shape

    if cout > cin:
        beta = (1 / rate) * (cout / cin - 1) + 1
    elif cin > cout:
        beta = (1 / rate) * (cin / cout - 1) + 1
        beta = 1 / beta
    else:
        beta = 1.

    a = 1
    b = (cin + beta * cout) / (beta * kh * kw)
    c = - cin * cout / rate / beta

    discr = b ** 2 - 4 * a * c
    max_rank = int((- b + np.sqrt(discr)) / 2 / a)
    # [R4, R3]
    max_rank = (int(np.ceil(beta * max_rank)), max_rank)
    return max_rank
