import numpy as np
from pyEM.fitting import EMfit, expectation_step
from pyEM.math import compGauss_ms, calc_BICint, calc_LME, calc_fval

class EMModel():
    def __init__(self, all_data=None, objfunc=None, param_names=None, convergence_type='NPL', convergence_method='sum', **kwargs):
        self.all_data = all_data
        self.nsubjects = len(all_data)
        self.fit = objfunc
        self.param_names = param_names
        self.convergence_type = convergence_type
        self.convergence_method = convergence_method
        self.posterior = {
            'mu': 0.1 * np.random.randn(len(param_names), 1) if param_names else None,
            'sigma': np.full((len(param_names),), 100) if param_names else None
        }

        # Placeholder attributes for storing results
        self.norm_params = None
        self.est_params = None
        self.subject_fits = None
        self.outfit = None
        self.covmat = None
        self.BICint = None
        self.LME = None

    def EMfit(self, **kwargs):
        # Call the EMfit function from fitting.py and store results
        self.outfit = EMfit(self.all_data, self.fit, self.param_names, 
                            convergence_type=self.convergence_type, 
                            convergence_method=self.convergence_method, **kwargs)
        self.norm_params = self.outfit['m'].T.copy()
        self.posterior = self.outfit['posterior']

    def calc_BICint(self, nsamples=2000, nll_output='negll'):
        # Call calc_BICint from math.py and store the result
        self.BICint = calc_BICint(self.all_data, self.param_names, self.posterior['mu'], 
                                  self.posterior['sigma'], self.fit, nsamples, nll_output=nll_output)
        
        if self.subject_fits is not None:
            self.subject_fits['BICint'] = self.BICint

    def calc_LME(self):
        # Call calc_LME from math.py and store the result
        Laplace_approx, lme, goodHessian = calc_LME(self.outfit['inv_h'], self.outfit['NPL'])
        self.LME = lme
        self.Laplace_approx = Laplace_approx
        self.goodHessian = goodHessian

        if self.subject_fits is not None:
            self.subject_fits['Laplace_approx'] = Laplace_approx
            self.subject_fits['lme'] = lme
            self.subject_fits['goodHessian'] = goodHessian

    def get_fits(self):
        # Store all relevant outputs in modfit_dict
        modfit_dict = {}
        m = self.outfit['m']
        inv_h = self.outfit['inv_h']
        posterior = self.outfit['posterior']
        NPL = self.outfit['NPL']
        NLPrior = self.outfit['NLPrior']

        # Fill in modfit_dict
        modfit_dict['norm_params'] = m.T.copy()
        modfit_dict['est_params'] = m.T.copy()
        modfit_dict['param_names'] = self.param_names
        modfit_dict['inverse_hess'] = inv_h
        modfit_dict['gauss.mu'] = posterior['mu']
        modfit_dict['gauss.sigma'] = posterior['sigma']
        modfit_dict['NPL'] = NPL
        modfit_dict['NLPrior'] = NLPrior
        modfit_dict['NLL'] = NPL - NLPrior

        # Get covariance matrix
        _, _, _, covmat_out = compGauss_ms(m, inv_h, 2)
        modfit_dict['gauss.cov'] = covmat_out
        try:
            modfit_dict['gauss.corr'] = np.corrcoef(covmat_out)
        except:
            print('Covariance matrix not square, symmetric, or positive semi-definite')
            modfit_dict['gauss.corr'] = np.eye(len(self.param_names))

        # Calculate integrated BIC
        if self.BICint is not None:
            modfit_dict['BICint'] = self.BICint
        else:
            modfit_dict['BICint'] = None

        # Calculate Laplace approximation and LME3
        if self.LME is not None:
            modfit_dict['Laplace_approx'] = self.Laplace_approx
            modfit_dict['lme'] = self.LME
            modfit_dict['goodHessian'] = self.goodHessian
        else:
            modfit_dict['Laplace_approx'] = None
            modfit_dict['lme'] = None
            modfit_dict['goodHessian'] = None

        # Get subject-specific fits
        modfit_dict['subj_fit'] = np.empty((self.nsubjects,), dtype='object')
        for subj_idx in range(self.nsubjects):
            # Fit each subject's data
            subj_fit = self.fit(m[:, subj_idx], *self.all_data[subj_idx], prior=None, output='all')
            modfit_dict['subj_fit'][subj_idx] = subj_fit
            modfit_dict['est_params'][subj_idx, :] = subj_fit['params']

        # Store the complete fit
        self.subject_fits = modfit_dict
        self.est_params = modfit_dict['est_params']

        return modfit_dict

# Example usage:
# subj_dict = rw_models.simulate(params=params, nblocks=6, ntrials=24)
# all_data = []
# for idx, (choices, rewards) in enumerate(zip(subj_dict['choices'], subj_dict['rewards'])):
#     all_data += [[choices, rewards]]
#
# # Fit the model
# RWModel = EMModel(all_data=all_data,
#                   objfunc=rw_models.fit,
#                   param_names=param_names)
# RWModel.EMfit(mstep_maxit=50)
# RWModel.get_fits()
# RWModel.calc_BICint(nll_output='CHOICE_NLL')
# RWModel.calc_LME()