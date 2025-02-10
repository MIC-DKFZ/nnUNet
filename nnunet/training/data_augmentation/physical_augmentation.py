import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform


def generate_uniform_from_range(lb, ub, size=None):
    return np.random.random(size=size) * (ub - lb) + lb


class GRESynthesis(AbstractTransform):
    """Synthesize gradient recalled echo images"""

    def __init__(self, p_gre=1.0, fa_range=(6, 25),
                 tr_range=(20, 100),
                 te_tr_ratio=(0.1, 0.8),
                 data_key="data"):
        self.data_key = data_key
        self.p_gre = p_gre
        self.fa_range = fa_range
        self.tr_range = tr_range
        self.te_tr_ratio = te_tr_ratio

    @staticmethod
    def _gradient_echo(rho, T1, T2, fa, TR, TE):
        T1 = np.clip(T1, 1e-5, None)
        T2 = np.clip(T2, 1e-5, None)

        fa = np.pi * fa / 180.
        cos_fa = np.cos(fa)
        E1 = np.exp(-TR / np.clip(T1, 1e-5, None))
        E2 = np.exp(-TE / np.clip(T2, 1e-5, None))
        signal = rho * (1 - E1) / (1 - cos_fa * E1) * E2
        return signal

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        seg = data_dict.get("seg", None)
        if np.random.random() < self.p_gre:
            rho, T1, T2 = tuple(data[:, [c]] for c in (1, 2, 3))
            fa = generate_uniform_from_range(*self.fa_range, size=(rho.shape[0], 1, 1, 1))
            tr = generate_uniform_from_range(*self.tr_range, size=(rho.shape[0], 1, 1, 1))
            ratio = generate_uniform_from_range(*self.te_tr_ratio, size=(rho.shape[0], 1, 1, 1))
            te = np.clip(tr * ratio, 5.0, None)
            gre = self._gradient_echo(rho, T1, T2, fa, tr, te)
            if seg is not None:
                gre[seg > -1] = np.clip(gre[seg > -1], 0, np.percentile(gre[seg > -1], 99.5))
            else:
                gre = np.clip(gre, 0, np.percentile(gre, 99.5))
            gre = (gre - np.mean(gre, axis=(-2, -1), keepdims=True)) / np.std(gre, axis=(-2, -1), keepdims=True)
            data_dict[self.data_key] = gre
        else:
            data_dict[self.data_key] = data[:, [0]]
        return data_dict


class InversionRecoverySynthesis(AbstractTransform):
    """Synthesize MOLLI images, SSFP readout (True FISP)"""

    def __init__(self, p_inversion=1.0,  data_key="data"):
        self.data_key = data_key
        self.p_inversion = p_inversion

    @staticmethod
    def _inversion_recovery(rho, T1, T2, fa, Tinv):
        fa = np.pi * fa / 180.
        T1 = np.clip(T1, 1e-5, None)
        T2 = np.clip(T2, 1e-5, None)
        cos_fa = np.cos(fa)
        ratio = T1 / T2
        steady_state = rho / (1 + cos_fa + (1 - cos_fa) * ratio)
        inv_factor = 1 + np.sin(fa * 0.5) / np.sin(fa) * (ratio * (1 - np.cos(fa)) + 1 + np.cos(fa))
        t1app_inv = 1 / T1 * np.cos(fa * 0.5) ** 2 + 1 / T2 * np.sin(fa * 0.5) ** 2
        t1app = 1 / t1app_inv
        signal = steady_state * (1 - inv_factor * np.exp(-Tinv / t1app))
        return np.abs(signal)

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        seg = data_dict.get("seg", None)
        if np.random.random() < self.p_inversion:
            rho, T1, T2 = tuple(data_dict[self.data_key][:, [c]] for c in (1, 2, 3))
            Ti = generate_uniform_from_range(180, np.percentile(T1, 99) * 1.2,
                                             size=(rho.shape[0], 1, 1, 1))
            fa = generate_uniform_from_range(8, 15,
                                             size=(rho.shape[0], 1, 1, 1))
            molli = self._inversion_recovery(rho, T1, T2, fa, Ti)
            if seg is not None:
                molli[seg > -1] = np.clip(molli[seg > -1], 0, np.percentile(molli[seg > -1], 99.5))
            else:
                molli = np.clip(molli, 0, np.percentile(molli, 99.5))
            molli = (molli - np.mean(molli, axis=(-2, -1), keepdims=True)) / np.std(molli, axis=(-2, -1), keepdims=True)

            data_dict[self.data_key] = molli
        else:
            data_dict[self.data_key] = data[:, [0]]
        return data_dict


class InvGREAugMixedTransform:
    def __init__(self,
                 inv: InversionRecoverySynthesis, gre: GRESynthesis,
                 p_transform=0.7,
                 p_inv=0.3):
        self.inv = inv
        self.gre = gre
        self.p_transform = p_transform
        self.p_inv = p_inv

    def __call__(self, **data_dict):
        data_orig = data_dict[self.inv.data_key][:, [0]].copy()
        data_inv = self.inv(**data_dict)[self.inv.data_key]
        data_gre = self.gre(**data_dict)[self.gre.data_key]
        inv_mask = generate_uniform_from_range(0, 1, size=(data_orig.shape[0], )) < self.p_inv
        data_gre[inv_mask, ...] = data_inv[inv_mask, ...]

        mask = generate_uniform_from_range(0, 1, size=(data_orig.shape[0], )) < self.p_transform

        # seg = data_dict.get("seg", None)
        # if seg is not None:
        #     blood_map = (seg == 1) | (seg == 3)
        #     myo_map = (seg == 2)
        #
        #     blood = data_gre[blood_map].mean(axis=(1, 2, 3))
        #     myo = data_gre[myo_map].mean(axis=(1, 2, 3))
        data_orig[mask, ...] = data_gre[mask, ...]

        data_dict[self.inv.data_key] = data_orig
        return data_dict

