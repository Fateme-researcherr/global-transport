# Script builds on top of https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/optimal_transport.py

import scipy.optimize
import torch


class OTPlanSampler:
    """Minibatch optimal-transport couplings with respect to the squared Euclidean cost.
    Implements global transport mini-batch exact OT, class-conditional mini-batch exact OT and multi-gpu training.
    """

    def sample_plan_with_scipy(self, x0, x1):
        r"""Exact OT coupling (wrt squared Euclidean cost) between a source and a target
        minibatch, solved as an assignment problem with scipy.
        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        j : numpy array, shape (bs,)
            column indices such that x1[j] is coupled to x0 row-by-row
        """
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0.detach(), x1.detach()) ** 2
        _, j = scipy.optimize.linear_sum_assignment(M.cpu().numpy())
        return j

    def sample_plan_with_scipy_global(self, x0, x1, accelerator):
        r"""Distributed version of :meth:`sample_plan_with_scipy`.

        All-gathers the source (``x0``) and target (``x1``) minibatches from every
        process, solves the exact squared-Euclidean assignment over the *global*
        batch on the main process, broadcasts it, and returns this process's slice
        of ``x1`` reordered to its globally-matched rows. As in the local version,
        ``x0`` is left untouched and only the target is permuted.

        Parameters
        ----------
        x0 : Tensor, shape (local_bs, *dim)
            this process's source minibatch (e.g. data latents)
        x1 : Tensor, shape (local_bs, *dim)
            this process's target minibatch (e.g. noise)
        accelerator : accelerate.Accelerator
            used for the all-gather / broadcast collectives and rank info

        Returns
        -------
        x1_coupled : Tensor, shape (local_bs, *dim)
            the globally OT-matched targets for this process's source rows
        """
        from accelerate.utils import broadcast

        local_bs = x0.shape[0]

        # All-gather along dim 0; every process receives the full global batch in
        # rank order, so global rows [r*local_bs : (r+1)*local_bs] belong to rank r.
        x0_g = accelerator.gather(x0.detach())
        x1_g = accelerator.gather(x1.detach())
        G = x0_g.shape[0]

        # Solve once on the main process so the assignment is identical everywhere,
        # then broadcast the column indices (cheap, avoids cross-rank tie-break drift).
        col = torch.empty(G, dtype=torch.long, device=x1.device)
        if accelerator.is_main_process:
            a = x0_g.reshape(G, -1).float()
            b = x1_g.reshape(G, -1).float()
            M = torch.cdist(a, b) ** 2
            _, j = scipy.optimize.linear_sum_assignment(M.cpu().numpy())
            col = torch.as_tensor(j, dtype=torch.long, device=x1.device)
        col = broadcast(col, from_process=0)

        # Return the matched targets for this process's source rows out of the
        # gathered global targets (the match may come from any rank's batch).
        start = accelerator.process_index * local_bs
        sel = col[start:start + local_bs]
        return x1_g[sel]

    @staticmethod
    def conditional_cost(x0, x1, c, c0=None, beta_mult=10.0):
        r"""Cost of the class-conditional assignment (C2OT, Cheng et al. 2025):

            M = ||x0 - x1||^2 + beta * 1[class(x0) != class(x1)],
            beta = beta_mult * mean(||x0 - x1||^2).

        c labels the rows of x0 (data), c0 the rows of x1 (noise; default: c, i.e. noise k
        inherits the label of datapoint k, so a within-class matching always exists).
        """
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        if c0 is None:
            c0 = c
        K = int(max(int(c.max()), int(c0.max()))) + 1
        oh_data = torch.nn.functional.one_hot(c.long(), K).float().to(x0.device)    # (bs, K) rows of M
        oh_noise = torch.nn.functional.one_hot(c0.long(), K).float().to(x1.device)  # (bs, K) cols of M

        C = torch.cdist(oh_data, oh_noise) ** 2              # 0 if same class, 2 if different
        S = torch.cdist(x0.detach(), x1.detach()) ** 2
        beta = float(beta_mult * S.mean())
        return S + beta * C

    def sample_conditional_plan_with_scipy(self, x0, x1, c, beta_mult=10.0, c0=None):
        r"""Class-conditional variant of :meth:`sample_plan_with_scipy` (C2OT, Cheng et al. 2025).

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            source minibatch (data)
        x1 : Tensor, shape (bs, *dim)
            target minibatch (noise)
        c : LongTensor, shape (bs,)
            class label of every row of x0
        beta_mult : float
            class barrier as a multiple of the mean transport cost
        c0 : LongTensor, shape (bs,), optional
            class label of every row of x1 (default: c)

        Returns
        -------
        j : numpy array, shape (bs,)
            column indices such that x1[j] is coupled to x0 row-by-row
        """
        M = self.conditional_cost(x0, x1, c, c0, beta_mult)
        _, j = scipy.optimize.linear_sum_assignment(M.cpu().numpy())
        return j

    def sample_conditional_plan_with_scipy_global(self, x0, x1, c, accelerator, beta_mult=10.0):
        r"""Distributed version of :meth:`sample_conditional_plan_with_scipy`.

        Like :meth:`sample_plan_with_scipy_global`: the data, noise and labels of every
        process are all-gathered, the class-conditional assignment is solved over the
        *global* batch on the main process and broadcast, and this process's slice of
        ``x1`` is returned reordered to its globally-matched rows.

        Parameters
        ----------
        x0 : Tensor, shape (local_bs, *dim)
            this process's data minibatch
        x1 : Tensor, shape (local_bs, *dim)
            this process's noise minibatch
        c : LongTensor, shape (local_bs,)
            class label of every row of x0
        accelerator : accelerate.Accelerator
            used for the all-gather / broadcast collectives and rank info

        Returns
        -------
        x1_coupled : Tensor, shape (local_bs, *dim)
            the globally CCOT-matched noise for this process's data rows
        """
        from accelerate.utils import broadcast

        local_bs = x0.shape[0]
        x0_g = accelerator.gather(x0.detach())
        x1_g = accelerator.gather(x1.detach())
        c_g = accelerator.gather(c.detach().reshape(local_bs))
        G = x0_g.shape[0]

        col = torch.empty(G, dtype=torch.long, device=x1.device)
        if accelerator.is_main_process:
            M = self.conditional_cost(x0_g.reshape(G, -1).float(), x1_g.reshape(G, -1).float(), c_g, c_g, beta_mult)
            _, j = scipy.optimize.linear_sum_assignment(M.cpu().numpy())
            col = torch.as_tensor(j, dtype=torch.long, device=x1.device)
        col = broadcast(col, from_process=0)

        start = accelerator.process_index * local_bs
        return x1_g[col[start:start + local_bs]]
