"""
Hard-coded results for test_regression
"""

### REGRESSION MODEL RESULTS : OLS, GLS, WLS, AR###

import numpy as np

class Longley(object):
    '''
    The results for the Longley dataset were obtained from NIST

    http://www.itl.nist.gov/div898/strd/general/dataarchive.html

    Other results were obtained from Stata
    '''
    def __init__(self):
        self.params = ( 15.0618722713733, -0.358191792925910E-01,
                 -2.02022980381683, -1.03322686717359, -0.511041056535807E-01,
                 1829.15146461355, -3482258.63459582)
        self.bse = (84.9149257747669, 0.334910077722432E-01,
                   0.488399681651699, 0.214274163161675, 0.226073200069370,
                   455.478499142212, 890420.383607373)
        self.scale = 92936.0061673238
        self.rsquared = 0.995479004577296
        self.rsquared_adj = 0.99246501
        self.df_model = 6
        self.df_resid = 9
        self.ess = 184172401.944494
        self.ssr = 836424.055505915
        self.mse_model = 30695400.3240823
        self.mse_resid = 92936.0061673238
        self.fvalue = 330.285339234588
        self.llf = -109.6174
        self.aic = 233.2349
        self.bic = 238.643
        self.pvalues = np.array([ 0.86314083,  0.31268106,  0.00253509,
            0.00094437,  0.8262118 , 0.0030368 ,  0.0035604 ])
        #pvalues from rmodelwrap
        self.resid = np.array((267.34003, -94.01394, 46.28717, -410.11462,
            309.71459, -249.31122, -164.04896, -13.18036, 14.30477, 455.39409,
            -17.26893, -39.05504, -155.54997, -85.67131, 341.93151,
            -206.75783))

    def conf_int(self): # a method to be consistent with sm
        return [(-177.0291,207.1524), (-.111581,.0399428),(-3.125065,
                -.9153928),(-1.517948,-.5485049),(-.5625173,.4603083),
                   (798.7873,2859.515),(-5496529,-1467987)]


    HC0_se=(51.22035, 0.02458, 0.38324, 0.14625, 0.15821,
               428.38438, 832212)
    HC1_se=(68.29380, 0.03277, 0.51099, 0.19499, 0.21094,
                571.17917, 1109615)
    HC2_se=(67.49208, 0.03653, 0.55334, 0.20522, 0.22324,
                617.59295, 1202370)
    HC3_se=(91.11939, 0.05562, 0.82213, 0.29879, 0.32491,
                922.80784, 1799477)

class LongleyGls(object):
    '''
    The following results were obtained from running the test script with R.
    '''
    def __init__(self):
        self.params = (6.73894832e-02, -4.74273904e-01, 9.48988771e+04)
        self.bse = (1.07033903e-02, 1.53385472e-01, 1.39447723e+04)
        self.llf = -121.4294962954981
        self.fittedvalues = [59651.8255, 60860.1385, 60226.5336, 61467.1268,
                63914.0846, 64561.9553, 64935.9028, 64249.1684, 66010.0426,
                66834.7630, 67612.9309, 67018.8998, 68918.7758, 69310.1280,
                69181.4207, 70598.8734]
        self.resid = [671.174465, 261.861502, -55.533603, -280.126803,
                -693.084618, -922.955349, 53.097212, -488.168351, 8.957367,
                1022.236970,  556.069099, -505.899787, -263.775842, 253.871965,
                149.579309, -47.873374]
        self.scale = 542.443043098**2
        self.tvalues = [6.296088, -3.092039, 6.805337]
        self.pvalues = [2.761673e-05, 8.577197e-03, 1.252284e-05]
        self.bic = 253.118790021
        self.aic = 250.858992591

class CCardWLS(object):
    def __init__(self):
        self.params = [-2.6941851611, 158.426977524, -7.24928987289,
                        60.4487736936, -114.10886935]

        self.bse = [3.807306306, 76.39115431, 9.724337321, 58.55088753,
                    139.6874965]
        #NOTE: we compute the scale differently than they do for analytic
        #weights
        self.scale = 189.0025755829012 ** 2
        self.rsquared = .2549143871187359
        self.rsquared_adj = .2104316639616448
        self.df_model = 4
        self.df_resid = 67
        self.ess = 818838.8079468152
        self.ssr = 2393372.229657007
        self.mse_model = 818838.8079468152 / 4
        self.mse_resid = 2393372.229657007 / 67
        self.fvalue = 5.730638077585917
        self.llf = -476.9792946562806
        self.aic = 963.95858931256
        self.bic = 975.34191990764
        # pvalues from R
        self.pvalues = [0.4816259843354, 0.0419360764848, 0.4585895209814,
                        0.3055904431658, 0.4168883565685]
        self.resid = [-286.964904785, -128.071563721, -405.860900879,
                      -20.1363945007, -169.824432373, -82.6842575073,
                      -283.314300537, -52.1719360352, 433.822174072,
                      -190.607543945, -118.839683533, -133.97076416,
                      -85.5728149414, 66.8180847168, -107.571769714,
                      -149.883285522, -140.972610474, 75.9255981445,
                      -135.979736328, -415.701263428, 130.080032349,
                      25.2313785553, 1042.14013672, -75.6622238159,
                      177.336639404, 315.870544434, -8.72801017761,
                      240.823760986, 54.6106033325, 65.6312484741,
                      -40.9218444824, 24.6115856171, -131.971786499,
                      36.1587944031, 92.5052108765, -136.837036133,
                      242.73274231, -65.0315093994, 20.1536407471,
                      -15.8874826431, 27.3513431549, -173.861785889,
                      -113.121154785, -37.1303443909, 1510.31530762,
                      582.916931152, -17.8628063202, -132.77381897,
                      -108.896934509, 12.4665794373, -122.014572144,
                      -158.986968994, -175.798873901, 405.886505127,
                      99.3692703247, 85.3450698853, -179.15007019,
                      -34.1245117188, -33.4909172058, -20.7287139893,
                      -116.217689514, 53.8837738037, -52.1533050537,
                      -100.632293701, 34.9342498779, -96.6685943604,
                      -367.32925415, -40.1300048828, -72.8692245483,
                      -60.8728256226, -35.9937324524, -222.944747925]

    def conf_int(self): # a method to be consistent with sm
        return [( -10.2936,  4.90523), ( 5.949595, 310.9044),
                (-26.65915, 12.16057), (-56.41929, 177.3168),
                (-392.9263, 164.7085)]

class LongleyRTO(object):
    def __init__(self):
        # Regression Through the Origin model
        # from Stata, make sure you force double to replicate
        self.params = [-52.993523, .07107319, -.42346599, -.57256869,
                       -.41420348, 48.417859]
        self.bse = [129.5447812, .0301663805, .4177363573, .2789908665,
                        .3212848136, 17.68947719]
        self.scale = 475.1655079819532**2
        self.rsquared = .9999670130705958
        self.rsquared_adj = .9999472209129532
        self.df_model = 6
        self.df_resid = 10
        self.ess = 68443718827.40025
        self.ssr = 2257822.599757476
        self.mse_model = 68443718827.40025 / 6
        self.mse_resid = 2257822.599757476 / 10
        self.fvalue = 50523.39573737409
        self.llf = -117.5615983965251
        self.aic = 247.123196793
        self.bic = 251.758729126
        self.pvalues = [0.6911082828354, 0.0402241925699, 0.3346175334102,
                        0.0672506018552, 0.2263470345100, 0.0209367642585]
        self.resid = [279.902740479, -130.324661255, 90.7322845459,
                    -401.312530518, -440.467681885, -543.54510498,
                    201.321121216, 215.908889771, 73.0936813354, 913.216918945,
                    424.824859619, -8.56475830078, -361.329742432,
                    27.3456058502, 151.28956604, -492.499359131]

    def conf_int(self):
        return [(-341.6373,   235.6502), ( .0038583,   .1382881),
                (-1.354241,   .5073086), (-1.194199,   .0490617),
                (-1.130071,   .3016637), ( 9.003248,   87.83247)]
