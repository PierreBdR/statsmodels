'''autogenerated and edited by hand

'''

import numpy as np

est = dict(
           N = 202,
           df_m = 2,
           df_r = 199,
           F = 92.94502024547633,
           r2 = .6769775594319385,
           rmse = 10.7037959322668,
           mss = 47782.65712176046,
           rss = 22799.67822456265,
           r2_a = .6737311027428123,
           ll = -763.9752181602238,
           ll_0 = -878.1085999159409,
           rank = 3,
           cmdline = "regress g_realinv g_realgdp L.realint, vce(robust)",
           title = "Linear regression",
           marginsok = "XB default",
           vce = "robust",
           depvar = "g_realinv",
           cmd = "regress",
           properties = "b V",
           predict = "regres_p",
           model = "ols",
           estat_cmd = "regress_estat",
           vcetype = "Robust",
          )

params_table = np.array([
     4.3742216647032,  .32355452428856,  13.519272136038,  5.703151404e-30,
     3.7361862031101,  5.0122571262963,              199,  1.9719565442518,
                   0, -.61399696947899,  .32772840315987, -1.8734933059173,
     .06246625509181, -1.2602631388273,   .0322691998693,              199,
     1.9719565442518,                0, -9.4816727746549,  1.3690593206013,
    -6.9256843965613,  5.860240898e-11, -12.181398261383, -6.7819472879264,
                 199,  1.9719565442518,                0]).reshape(3,9)

params_table_colnames = 'b se t pvalue ll ul df crit eform'.split()

params_table_rownames = 'g_realgdp L.realint _cons'.split()

cov = np.array([
      .1046875301876, -.00084230205782, -.34205013876828, -.00084230205782,
     .10740590623772, -.14114426417778, -.34205013876828, -.14114426417778,
     1.8743234233252]).reshape(3,3)

cov_colnames = 'g_realgdp L.realint _cons'.split()

cov_rownames = 'g_realgdp L.realint _cons'.split()

class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__  = self

        for i,att in enumerate(['params', 'bse', 'tvalues', 'pvalues']):
            self[att] = self.params_table[:,i]


results_hc0 = Bunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

######################

est = dict(
           df_m = 2,
           df_r = 199,
           F = 89.45120275471848,
           N = 202,
           lag = 4,
           rank = 3,
           title = "Regression with Newey-West standard errors",
           cmd = "newey",
           cmdline = "newey g_realinv g_realgdp L.realint, lag(4)",
           estat_cmd = "newey_estat",
           predict = "newey_p",
           vcetype = "Newey-West",
           depvar = "g_realinv",
           properties = "b V",
          )

params_table = np.array([
     4.3742216647032,  .33125644884286,  13.204940401864,  5.282334606e-29,
     3.7209983425819,  5.0274449868245,              199,  1.9719565442518,
                   0, -.61399696947899,  .29582347593197, -2.0755518727668,
     .03922090940364, -1.1973480087863, -.03064593017165,              199,
     1.9719565442518,                0, -9.4816727746549,  1.1859338087713,
    -7.9951112823729,  1.036821797e-13, -11.820282709911, -7.1430628393989,
                 199,  1.9719565442518,                0]).reshape(3,9)

params_table_colnames = 'b se t pvalue ll ul df crit eform'.split()

params_table_rownames = 'g_realgdp L.realint _cons'.split()

cov = np.array([
     .10973083489998,   .0003953117603, -.31803287070833,   .0003953117603,
     .08751152891247, -.06062111121649, -.31803287070833, -.06062111121649,
     1.4064389987868]).reshape(3,3)

cov_colnames = 'g_realgdp L.realint _cons'.split()

cov_rownames = 'g_realgdp L.realint _cons'.split()

results_newey4 = Bunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )



est = dict(
           N = 202,
           inexog_ct = 2,
           exexog_ct = 0,
           endog_ct = 0,
           partial_ct = 0,
           bw = 5,
           df_r = 199,
           df_m = 2,
           sdofminus = 0,
           dofminus = 0,
           r2 = .6769775594319388,
           rmse = 10.7037959322668,
           rss = 22799.67822456265,
           mss = 47782.65712176055,
           r2_a = .6737311027428126,
           F = 89.45120275471867,
           Fp = 1.93466284646e-28,
           Fdf1 = 2,
           Fdf2 = 199,
           yy = 72725.68049533673,
           partialcons = 0,
           cons = 1,
           jdf = 0,
           j = 0,
           ll = -763.9752181602239,
           rankV = 3,
           rankS = 3,
           rankxx = 3,
           rankzz = 3,
           r2c = .6769775594319388,
           r2u = .6864975608440735,
           yyc = 70582.33534632321,
           hacsubtitleV = "Statistics robust to heteroskedasticity and autocorrelation",
           hacsubtitleB = "Estimates efficient for homoskedasticity only",
           title = "OLS estimation",
           predict = "ivreg2_p",
           version = "02.2.08",
           cmdline = "ivreg2 g_realinv g_realgdp L.realint, robust bw(5) small",
           cmd = "ivreg2",
           model = "ols",
           depvar = "g_realinv",
           vcetype = "Robust",
           partialsmall = "small",
           small = "small",
           tvar = "qu",
           kernel = "Bartlett",
           inexog = "g_realgdp L.realint",
           insts = "g_realgdp L.realint",
           properties = "b V",
          )

params_table = np.array([
     4.3742216647032,  .33125644884286,  13.204940401864,  5.282334606e-29,
     3.7209983425819,  5.0274449868245,              199,  1.9719565442518,
                   0, -.61399696947899,  .29582347593197, -2.0755518727668,
     .03922090940364, -1.1973480087863, -.03064593017165,              199,
     1.9719565442518,                0, -9.4816727746549,  1.1859338087713,
    -7.9951112823729,  1.036821797e-13, -11.820282709911, -7.1430628393989,
                 199,  1.9719565442518,                0]).reshape(3,9)

params_table_colnames = 'b se t pvalue ll ul df crit eform'.split()

params_table_rownames = 'g_realgdp L.realint _cons'.split()

cov = np.array([
     .10973083489998,   .0003953117603, -.31803287070833,   .0003953117603,
     .08751152891247, -.06062111121649, -.31803287070833, -.06062111121649,
     1.4064389987868]).reshape(3,3)

cov_colnames = 'g_realgdp L.realint _cons'.split()

cov_rownames = 'g_realgdp L.realint _cons'.split()

results_ivhac4_small = Bunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

est = dict(
           N = 202,
           inexog_ct = 2,
           exexog_ct = 0,
           endog_ct = 0,
           partial_ct = 0,
           bw = 5,
           df_m = 2,
           sdofminus = 0,
           dofminus = 0,
           r2 = .6769775594319388,
           rmse = 10.6240149746225,
           rss = 22799.67822456265,
           mss = 47782.65712176055,
           r2_a = .6737311027428126,
           F = 89.45120275471867,
           Fp = 1.93466284646e-28,
           Fdf1 = 2,
           Fdf2 = 199,
           yy = 72725.68049533673,
           yyc = 70582.33534632321,
           partialcons = 0,
           cons = 1,
           jdf = 0,
           j = 0,
           ll = -763.9752181602239,
           rankV = 3,
           rankS = 3,
           rankxx = 3,
           rankzz = 3,
           r2c = .6769775594319388,
           r2u = .6864975608440735,
           hacsubtitleV = "Statistics robust to heteroskedasticity and autocorrelation",
           hacsubtitleB = "Estimates efficient for homoskedasticity only",
           title = "OLS estimation",
           predict = "ivreg2_p",
           version = "02.2.08",
           cmdline = "ivreg2 g_realinv g_realgdp L.realint, robust bw(5)",
           cmd = "ivreg2",
           model = "ols",
           depvar = "g_realinv",
           vcetype = "Robust",
           partialsmall = "small",
           tvar = "qu",
           kernel = "Bartlett",
           inexog = "g_realgdp L.realint",
           insts = "g_realgdp L.realint",
           properties = "b V",
          )

params_table = np.array([
     4.3742216647032,  .32878742225811,  13.304102798888,  2.191074740e-40,
     3.7298101585076,  5.0186331708989, np.nan,  1.9599639845401,
                   0, -.61399696947899,  .29361854972141, -2.0911382133777,
     .03651567605333, -1.1894787521258, -.03851518683214, np.nan,
     1.9599639845401,                0, -9.4816727746549,  1.1770944273439,
     -8.055150508231,  7.938107001e-16, -11.788735458652, -7.1746100906581,
    np.nan,  1.9599639845401,                0]).reshape(3,9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'g_realgdp L.realint _cons'.split()

cov = np.array([
     .10810116903513,  .00038944079356, -.31330961025227,  .00038944079356,
      .0862118527405, -.05972079768357, -.31330961025227, -.05972079768357,
      1.385551290884]).reshape(3,3)

cov_colnames = 'g_realgdp L.realint _cons'.split()

cov_rownames = 'g_realgdp L.realint _cons'.split()

results_ivhac4_large = Bunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

######################

est = dict(
           N = 202,
           inexog_ct = 2,
           exexog_ct = 0,
           endog_ct = 0,
           partial_ct = 0,
           df_r = 199,
           df_m = 2,
           sdofminus = 0,
           dofminus = 0,
           r2 = .6769775594319388,
           rmse = 10.7037959322668,
           rss = 22799.67822456265,
           mss = 47782.65712176055,
           r2_a = .6737311027428126,
           F = 92.94502024547634,
           Fp = 3.12523087723e-29,
           Fdf1 = 2,
           Fdf2 = 199,
           yy = 72725.68049533673,
           yyc = 70582.33534632321,
           partialcons = 0,
           cons = 1,
           jdf = 0,
           j = 0,
           ll = -763.9752181602239,
           rankV = 3,
           rankS = 3,
           rankxx = 3,
           rankzz = 3,
           r2c = .6769775594319388,
           r2u = .6864975608440735,
           hacsubtitleV = "Statistics robust to heteroskedasticity",
           hacsubtitleB = "Estimates efficient for homoskedasticity only",
           title = "OLS estimation",
           predict = "ivreg2_p",
           version = "02.2.08",
           cmdline = "ivreg2 g_realinv g_realgdp L.realint, robust small",
           cmd = "ivreg2",
           model = "ols",
           depvar = "g_realinv",
           vcetype = "Robust",
           partialsmall = "small",
           small = "small",
           inexog = "g_realgdp L.realint",
           insts = "g_realgdp L.realint",
           properties = "b V",
          )

params_table = np.array([
     4.3742216647032,  .32355452428856,  13.519272136038,  5.703151404e-30,
     3.7361862031101,  5.0122571262963,              199,  1.9719565442518,
                   0, -.61399696947899,  .32772840315987, -1.8734933059173,
     .06246625509181, -1.2602631388273,   .0322691998693,              199,
     1.9719565442518,                0, -9.4816727746549,  1.3690593206013,
    -6.9256843965613,  5.860240898e-11, -12.181398261383, -6.7819472879264,
                 199,  1.9719565442518,                0]).reshape(3,9)

params_table_colnames = 'b se t pvalue ll ul df crit eform'.split()

params_table_rownames = 'g_realgdp L.realint _cons'.split()

cov = np.array([
      .1046875301876, -.00084230205782, -.34205013876828, -.00084230205782,
     .10740590623772, -.14114426417778, -.34205013876828, -.14114426417778,
     1.8743234233252]).reshape(3,3)

cov_colnames = 'g_realgdp L.realint _cons'.split()

cov_rownames = 'g_realgdp L.realint _cons'.split()

results_ivhc0_small = Bunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

###################

est = dict(
           N = 202,
           inexog_ct = 2,
           exexog_ct = 0,
           endog_ct = 0,
           partial_ct = 0,
           df_m = 2,
           sdofminus = 0,
           dofminus = 0,
           r2 = .6769775594319388,
           rmse = 10.6240149746225,
           rss = 22799.67822456265,
           mss = 47782.65712176055,
           r2_a = .6737311027428126,
           F = 92.94502024547633,
           Fp = 3.12523087723e-29,
           Fdf1 = 2,
           Fdf2 = 199,
           yy = 72725.68049533673,
           yyc = 70582.33534632321,
           r2u = .6864975608440735,
           partialcons = 0,
           cons = 1,
           jdf = 0,
           j = 0,
           ll = -763.9752181602239,
           rankV = 3,
           rankS = 3,
           rankxx = 3,
           rankzz = 3,
           r2c = .6769775594319388,
           hacsubtitleV = "Statistics robust to heteroskedasticity",
           hacsubtitleB = "Estimates efficient for homoskedasticity only",
           title = "OLS estimation",
           predict = "ivreg2_p",
           version = "02.2.08",
           cmdline = "ivreg2 g_realinv g_realgdp L.realint, robust",
           cmd = "ivreg2",
           model = "ols",
           depvar = "g_realinv",
           vcetype = "Robust",
           partialsmall = "small",
           inexog = "g_realgdp L.realint",
           insts = "g_realgdp L.realint",
           properties = "b V",
          )

params_table = np.array([
     4.3742216647032,  .32114290415293,  13.620795004769,  3.012701837e-42,
     3.7447931386729,  5.0036501907336, np.nan,  1.9599639845401,
                   0, -.61399696947899,  .32528567293437, -1.8875622892954,
     .05908473670106, -1.2515451731172,  .02355123415926, np.nan,
     1.9599639845401,                0, -9.4816727746549,  1.3588550094989,
    -6.9776927695558,  3.000669464e-12, -12.144979653484, -6.8183658958253,
    np.nan,  1.9599639845401,                0]).reshape(3,9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'g_realgdp L.realint _cons'.split()

cov = np.array([
     .10313276488778, -.00082979262132, -.33697018621231, -.00082979262132,
     .10581076901637, -.13904806223455, -.33697018621231, -.13904806223455,
     1.8464869368401]).reshape(3,3)

cov_colnames = 'g_realgdp L.realint _cons'.split()

cov_rownames = 'g_realgdp L.realint _cons'.split()

results_ivhc0_large = Bunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )
