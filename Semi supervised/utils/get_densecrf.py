import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

def dense_crf(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    # output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 1)
    U = unary_from_softmax(output_probs)
    # U = -np.log(output_probs)
    # U = U.reshape((2, -1))
    # U = np.ascontiguousarray(U)
    # img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)

    feats = create_pairwise_gaussian(sdims=(3, 3), shape=(img.shape[1], img.shape[2] ) )
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=0)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(10)
    Q = np.array(Q)
    Q = np.argmax(Q, axis=0).reshape((h, w))

    return Q