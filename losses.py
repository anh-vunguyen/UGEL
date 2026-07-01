import torch
import numpy as np


def Gaussian_NLL(y, mu, sigma):
    logprob = -torch.log(sigma) - 0.5*torch.log(torch.Tensor([2*torch.pi]).to(sigma.device)) - 0.5*((y-mu)/sigma)**2
    loss = torch.mean(-logprob)
    return loss


def beta_nll_loss(mu, sigma, target, beta=0.75):
    loss = 0.5*((target-mu)/sigma)**2 + torch.log(sigma)
    if beta > 0:
        loss = loss * (sigma ** 2) ** beta
    loss = loss.mean()
    return loss


def Beta_NLL(y, mu, nu):
    eps = 1e-9
    loss = -(torch.lgamma(nu+eps) - torch.lgamma(mu*nu+eps) - torch.lgamma((1-mu)*nu+eps) + (mu*nu-1)*torch.log(y+eps) + ((1-mu)*nu-1)*torch.log(1-y+eps))
    loss = torch.mean(loss)
    return loss


def NIG_NLL(y, gamma, v, alpha, beta):
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v) \
    - alpha*torch.log(twoBlambda) \
    + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda) \
    + torch.lgamma(alpha) \
    - torch.lgamma(alpha+0.5)
    return nll


def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*(a1-1)/b1 * (v2*torch.square(mu2-mu1)) \
        + 0.5*v2/v1 \
        - 0.5*torch.log(torch.abs(v2)/torch.abs(v1)) \
        + 0.5 + a2*torch.log(b1/b2) \
        - (torch.lgamma(a1) - torch.lgamma(a2)) \
        + (a1 - a2)*torch.digamma(a1) \
        - (b1 - b2)*a1/b1
    return KL


def NIG_reg(y, gamma, v, alpha, beta, omega=0.01, kl=False):
    error = torch.abs(y-gamma)
    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha)
        reg = error*evi
    return torch.mean(reg)


def EvidentialRegressionLoss(y_true, evidential_output, coeff=1.0):
    gamma, v, alpha, beta = torch.tensor_split(evidential_output, 4, dim=-1)
    gamma = gamma.squeeze(-1)
    v = v.squeeze(-1)
    alpha = alpha.squeeze(-1)
    beta = beta.squeeze(-1)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_reg(y_true, gamma, v, alpha, beta)
    return loss_nll.mean() + coeff * loss_reg.mean()


def EvidentialRegressionLoss_v2(y_true, evidential_output, coeff=1.0, beta_nll=0.75):
    """
    Using the adaptive weight approach from 'On the Pitfalls of Heteroscedastic Uncertainty Estimation with
    Probabilistic Neural Networks
    """
    gamma, v, alpha, beta = torch.tensor_split(evidential_output, 4, dim=-1)
    gamma = gamma.squeeze(-1)
    v = v.squeeze(-1)
    alpha = alpha.squeeze(-1)
    beta = beta.squeeze(-1)
    
    # Adaptive weight
    variance = beta/((alpha-1)*gamma + 1e-6)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_nll = loss_nll * (variance ** beta_nll)
    
    loss_reg = NIG_reg(y_true, gamma, v, alpha, beta)
    return loss_nll.mean() + coeff * loss_reg.mean()
