import torch

class eig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        # normalize the shape to be batched
        Ashape = A.shape
        if A.ndim == 2:
            A = A.unsqueeze(0)
        elif A.ndim > 3:
            A = A.view(*A.shape[:-2], A.shape[-2], A.shape[-1])

        nbatch = A.shape[0]
        evecs = torch.empty_like(A).to(A.device)
        evals = torch.empty(A.shape[0], A.shape[-1]).to(A.dtype).to(A.device)
        for i in range(nbatch):
            evalue, evec = torch.eig(A[i], eigenvectors=True)

            # check if the eigenvalues contain complex numbers
            if not torch.allclose(evalue[:,1], torch.zeros_like(evalue[:,1])):
                raise ValueError("The eigenvalues contain complex numbers")

            # sort eigenvalues from lowest to highest
            evalue, idxsort = torch.sort(evalue[:,0], dim=-1) # idxsort (na,)
            evec = torch.index_select(evec, dim=-1, index=idxsort)

            evecs[i] = evec
            evals[i] = evalue

        # reshape the results
        evecs = evecs.view(*Ashape)
        evals = evals.view(*Ashape[:-1])

        ctx.evecs = evecs
        ctx.evals = evals
        return evals, evecs

    @staticmethod
    def backward(ctx, grad_evals, grad_evecs):
        # grad_evals: (...,na)
        # grad_evecs: (...,na,na)
        batchshape = grad_evals.shape[:-1]
        na = grad_evals.shape[-1]

        dLde = grad_evals.view(-1,na) # (nbatch, na)
        dLdU = grad_evecs.view(-1,na,na)
        U = ctx.evecs.view(-1,na,na) # (nbatch,na,na)
        Hcurly = ctx.evals.view(-1,na) # (nbatch,na)
        UT = U.transpose(-2,-1) # (nbatch,na,na)

        # calculate the contribution from grad_evals
        UUT = torch.bmm(U, UT)
        UdiagUT = torch.bmm(U, UT*dLde.unsqueeze(-1))
        econtrib, _ = torch.solve(UdiagUT, UUT)

        # calculate the inverse of H-evals
        Hmevals = Hcurly.unsqueeze(-2) - Hcurly.unsqueeze(-1) # (nbatch,na,na)
        Hmevals.diagonal(offset=0, dim1=-2, dim2=-1).fill_(float("inf"))
        Hmevals_inv = 1.0 / Hmevals

        # calculate the contribution from grad_evecs

        # orthogonalizing the dLdU first before applying the other operations
        dLdU_ortho = dLdU - (dLdU * U).sum(dim=-2, keepdim=True) * U
        B = Hmevals_inv * torch.bmm(UT, dLdU_ortho)
        A = torch.bmm(B, UT)
        # A = torch.bmm(B, torch.inverse(U))
        # Ucontrib = torch.bmm(U, A)
        Ucontrib = torch.solve(A, UT)[0]

        # reshape the contributions
        shape = grad_evecs.shape
        econtrib = econtrib.view(*shape)
        Ucontrib = Ucontrib.view(*shape)

        return econtrib + Ucontrib

if __name__ == "__main__":
    from ddft.utils.fd import finite_differences

    A = torch.tensor([
        [[0.7, 0.2, 0.0],
         [0.3, 0.8, 0.0],
         [0.0, 0.0, 2.0]],

        [[0.4, -1.4, -1.1],
         [-1.4, 1.0, 1.3],
         [-1.2, 1.3, 0.1]]
    ]).to(torch.float64).requires_grad_()
    evals, evecs = eig.apply(A)
    print(evals, evecs)

    def getloss(A):
        evals, evecs = eig.apply(A)
        loss1 = (evals**4).sum()
        loss2 = (evecs.abs()**3).sum()
        loss = loss2# + loss2
        return loss

    loss = getloss(A)
    loss.backward()
    Agrad = A.grad.data

    with torch.no_grad():
        Afd = finite_differences(getloss, (A,), 0, eps=1e-4)
    print(Agrad)
    print(Afd)
    print(Agrad / Afd)
