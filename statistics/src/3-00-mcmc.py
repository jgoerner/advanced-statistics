import numpy as np
from scipy.stats import norm
from tqdm import tqdm

def mcmc(data, dist, target, init, proposal_width, params_prior, params_const, n_iter):
    """MCMC Metropolis Sampling
    
    Parameter
    ---------
    data: 1-d array,
        observerd data
    dist: scipy.stats distribution,
        distribution that underlies the sampling
    target: str,
        target param to be sampled
    init: float,
        initial value for the target parameter
    proposal_width: float,
        step-width of sampling
    params_prior: dict,
        priors of the distribution
    params_const: dict,
        constant parameter of the distribution,
    n_iter: int,
        number of samplings
        
    Return
    ------
    trace: 1-d array,
        trace of the sampling
    """
    params_current = params_const.copy() # defensive copy (!)
    params_current.update({target: init})
    params_proposal = params_const.copy() # defensive copy (!)
    trace = [params_current[target]]
    for _ in tqdm(range(n_iter)):

        # proposal
        proposal = norm(params_current[target], proposal_width).rvs()
        params_proposal.update({target: proposal})
        
        # likelihoods
        l_current = dist(**params_current).pdf(data).prod()
        l_proposal = dist(**params_proposal).pdf(data).prod()
        
        # priors
        pr_current = dist(**params_prior).pdf(params_current[target])
        pr_proposal = dist(**params_prior).pdf(params_proposal[target])
        
        # probabilities
        p_current = l_current * pr_current
        p_proposal = l_proposal * pr_proposal
        
        # switch (?)
        accept = np.random.rand() < (p_proposal / p_current)
        
        if accept:
            params_current = params_proposal.copy() # defensive copy (!)
        trace.append(params_current[target])
    return trace