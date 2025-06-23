import torch
from torch import nn

from .membership_functions import *
from .t_norms import *


class Antecedent(nn.Module):
    def __init__(self, in_dim, out_dim, num_fuzzy_set, mf='Gaussian', frb='CoCo-FRB', tnorm='prod'):
        """
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param num_fuzzy_set: No. of fuzzy sets defined on each feature
        :param mf: membership function, {'Gaussian' (default), 'CEMF' (for AYATSK)}
        :param frb: fuzzy rule base, {'CoCo-FRB' (default), 'FuCo-FRB'}
        :param tnorm: for computing firing strength, {'prod' (default), 'yager', 'yager_simple', 'ale_softmin_yager'}
        """
        super(Antecedent, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_fuzzy_set = num_fuzzy_set
        self.mf = mf
        self.frb = frb
        self.tnorm = tnorm
        self.lambda_yager = 1
        self.k_ecgmf = 1
        self.k_cemf = 10
        self.mf_lower_bound = 0.1

        # CoCo-FRB or FuCo-FRB
        self.FRB = self._init_frb(in_dim, num_fuzzy_set, frb_type=frb)
        self.num_rule = self.FRB.size(0)

        # antecedent initialization
        if mf == 'Gaussian' or mf == 'CEMF':
            partition = torch.arange(self.num_fuzzy_set, dtype=torch.float64) / (self.num_fuzzy_set - 1)
            self.center = nn.Parameter(partition.repeat([self.in_dim, 1]).T)  # [num_fuzzy_set, in_dim]
            self.spread = nn.Parameter(
                torch.ones([num_fuzzy_set, in_dim], dtype=torch.float64))  # [num_fuzzy_set, in_dim]
        else:
            raise ValueError("Invalid value for mf: '{}'".format(mf))

    def _init_frb(self, in_dim, num_fuzzy_set, frb_type):
        """
        generate the index of FRB
        :param in_dim: input dimension
        :param num_fuzzy_set: No. of fuzzy sets defined on each feature
        :param frb_type: fuzzy rule base, {'CoCo-FRB' (default), 'FuCo-FRB'}
        :return: the index of the fuzzy set for computing the firing strength
        """
        if frb_type == 'CoCo-FRB':
            fs_ind = torch.tensor(range(num_fuzzy_set)).unsqueeze(1).repeat_interleave(in_dim, dim=1)
            return fs_ind.long()
        elif frb_type == 'FuCo-FRB':
            fs_ind = torch.zeros([num_fuzzy_set ** in_dim, in_dim])
            for i, ii in enumerate(reversed(range(in_dim))):
                # i: positive order subscript; ii: negative order subscript
                fs_ind[:, ii] = torch.tensor(range(num_fuzzy_set)).repeat_interleave(num_fuzzy_set ** i).repeat(
                    num_fuzzy_set ** ii)
            return fs_ind.long()
        else:
            raise ValueError(
                "Invalid value for frb: '{}', expected 'CoCo-FRB', 'En-FRB', 'Cross-FRB', 'FuCo-FRB'".format(
                    self.frb))

    def forward(self, model_input):
        """
        computing firing strength according to the current input samples
        :param model_input: [num_sam, in_dim]
        :return: firing strengths (without normalization) [num_sam, num_rule]
        """
        model_input = model_input.double()  # [num_sam, in_dim]

        # membership_value, [num_sam, num_rule, in_dim]
        if self.mf == 'Gaussian':
            membership_value = gauss(model_input.unsqueeze(1), self.center, self.spread)
        elif self.mf == 'CEMF':
            membership_value = cemf(model_input.unsqueeze(1), self.center, self.spread, k=self.k_cemf)
        else:
            raise ValueError("Invalid value for mf: '{}'".format(self.mf))

        # firing strength, [num_sam, num_rule]
        in_dim, fs_ind = self.in_dim, self.FRB
        if self.tnorm == 'prod':
            fir_str = membership_value[:, fs_ind, range(in_dim)].prod(dim=2)
        elif self.tnorm == 'yager':
            fir_str = yager(membership_value[:, fs_ind, range(in_dim)], lam=self.lambda_yager, dim=2)
        elif self.tnorm == 'yager_simple':
            fir_str = yager_simple(membership_value[:, fs_ind, range(in_dim)], lam=self.lambda_yager, dim=2)
        elif self.tnorm == 'ale_softmin_yager':
            fir_str = ale_softmin_yager(membership_value[:, fs_ind, range(in_dim)], lam=self.lambda_yager, dim=2)
        else:
            raise ValueError("Invalid value for tnorm: '{}'".format(self.tnorm))

        return fir_str  # [num_sam,num_rule]


class Consequent(nn.Module):
    def __init__(self, in_dim, out_dim, num_rule, order='first'):
        """
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param num_rule: No. of fuzzy rules of the system
        :param order: {'first' (default), 'zero'}
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rule = num_rule
        self.order = order

        # consequent initialization
        if order == 'first':
            # [out_dim, num_rule, in_dim+1]
            self.con_param = nn.Parameter(torch.zeros([out_dim, num_rule, in_dim + 1], dtype=torch.float64))
        elif self.order == 'zero':
            # [out_dim, num_rule]
            self.con_param = nn.Parameter(torch.zeros([out_dim, num_rule], dtype=torch.float64))
        else:
            raise ValueError("Invalid value for order: '{}', expected 'first', 'zero'".format(self.order))

    def forward(self, model_input):
        """
        :param model_input: [num_sam, in_dim]
        :return: rule outputs
        """
        if self.order == 'first':
            # [num_sam, num_rule, out_dim]
            rule_output = (self.con_param[:, :, 1:] @ model_input.T).permute([2, 1, 0]) + self.con_param[:, :, 0].T
        elif self.order == 'zero':
            # [out_dim, num_rule]
            rule_output = self.con_param
        else:
            raise ValueError("Invalid value for tnorm: '{}'".format(self.tnorm))

        return rule_output


class TSK(nn.Module):
    def __init__(self, in_dim, out_dim, num_fuzzy_set, mf='Gaussian', frb='CoCo-FRB', tnorm='prod', order='first'):
        """
        TSK fuzzy system
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param num_fuzzy_set: No. of fuzzy sets defined on each feature
        :param mf: membership function, {'Gaussian' (default), 'CEMF' (for AYATSK)}
        :param frb: fuzzy rule base, {'CoCo-FRB' (default), 'FuCo-FRB'}
        :param tnorm: for computing firing strength, {'prod' (default), 'yager', 'yager_simple', 'ale_softmin_yager'}
        :param order: {'first' (default), 'zero'}
        """
        super(TSK, self).__init__()

        # generate the antecedent and consequent
        self.antecedent = Antecedent(in_dim, out_dim, num_fuzzy_set, mf, frb, tnorm)
        self.num_rule = self.antecedent.num_rule
        self.consequent = Consequent(in_dim, out_dim, self.num_rule, order)

    @property
    def order(self):
        return self.consequent.order

    @property
    def in_dim(self):
        return self.antecedent.in_dim

    @property
    def out_dim(self):
        return self.antecedent.out_dim

    @property
    def num_fuzzy_set(self):
        return self.antecedent.num_fuzzy_set

    def forward(self, model_input):
        """

        :param model_input: [num_sam,in_dim]
        :return: [num_sam, out_dim]
        """
        model_input = model_input.double()

        # compute firing strengths and rule outputs
        fir_str = self.antecedent(model_input)
        rule_output = self.consequent(model_input)

        # de-fuzzy for computing the model outputs
        fir_str_bar = fir_str / fir_str.sum(dim=1).unsqueeze(1)  # [num_sam,num_rule]
        if self.order == 'first':
            model_output = torch.einsum('NRC,NR->NC', rule_output, fir_str_bar)  # [num_sam, out_dim]
        elif self.order == 'zero':
            model_output = fir_str_bar @ rule_output.T  # [num_sam, out_dim]
        else:
            raise ValueError("Invalid value for tnorm: '{}'".format(self.tnorm))

        return model_output
