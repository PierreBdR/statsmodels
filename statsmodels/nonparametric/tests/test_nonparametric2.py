import numpy as np
import numpy.testing as npt
import numpy.testing.decorators as dec
import scipy.stats as stats
import statsmodels.nonparametric as nparam
from ..nonparametric2 import SetDefaults, SemiLinear
#import nonparametric2 as nparam
reload(nparam)
import csv

class MyTest(object):
    def setUp(self):
        N = 60
        np.random.seed(123456)
        self.o = np.random.binomial(2, 0.7, size=(N, 1))
        self.o2 = np.random.binomial(3, 0.7, size=(N, 1))
        self.c1 = np.random.normal(size=(N, 1))
        self.c2 = np.random.normal(10, 1, size=(N, 1))
        self.c3 = np.random.normal(10, 2, size=(N, 1))
        self.noise = np.random.normal(size=(N, 1))
        b0 = 0.3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        self.y = b0 + b1 * self.c1 + b2 * self.c2 + self.noise
        self.y2 = b0 + b1 * self.c1 + b2 * self.c2 + self.o + self.noise
        # Italy data from R's np package (the first 50 obs) R>> data (Italy)

        self.Italy_gdp = \
        [8.556, 12.262, 9.587, 8.119, 5.537, 6.796, 8.638,
         6.483, 6.212, 5.111, 6.001, 7.027, 4.616, 3.922,
         4.688, 3.957, 3.159, 3.763, 3.829, 5.242, 6.275,
         8.518, 11.542, 9.348, 8.02, 5.527, 6.865, 8.666,
         6.672, 6.289, 5.286, 6.271, 7.94, 4.72, 4.357,
         4.672, 3.883, 3.065, 3.489, 3.635, 5.443, 6.302,
         9.054, 12.485, 9.896, 8.33, 6.161, 7.055, 8.717,
         6.95]

        self.Italy_year = \
        [1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951,
       1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1952,
       1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952,
       1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1953, 1953,
       1953, 1953, 1953, 1953, 1953, 1953]

        # OECD panel data from NP  R>> data(oecdpanel)
        self.growth = \
        [-0.0017584, 0.00740688, 0.03424461, 0.03848719, 0.02932506,
        0.03769199, 0.0466038,  0.00199456, 0.03679607, 0.01917304,
       -0.00221, 0.00787269, 0.03441118, -0.0109228, 0.02043064,
       -0.0307962, 0.02008947, 0.00580313, 0.00344502, 0.04706358,
        0.03585851, 0.01464953, 0.04525762, 0.04109222, -0.0087903,
        0.04087915, 0.04551403, 0.036916, 0.00369293, 0.0718669,
        0.02577732, -0.0130759, -0.01656641, 0.00676429, 0.08833017,
        0.05092105, 0.02005877,  0.00183858, 0.03903173, 0.05832116,
        0.0494571, 0.02078484,  0.09213897, 0.0070534, 0.08677202,
        0.06830603, -0.00041, 0.0002856, 0.03421225, -0.0036825]

        self.oecd = \
        [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       0, 0, 0, 0]
    def write2file(self, file_name, data):
        data_file = csv.writer(open(file_name, "w"))
        data = np.column_stack(data)
        N = max(np.shape(data))
        K = min(np.shape(data))
        print "shape is for writing in file is: ", (N,K)
        data = np.reshape(data, (N,K))
        for i in range(N):
            data_file.writerow(list(data[i, :]))



class TestUKDE(MyTest):
    @dec.slow
    def test_pdf_mixeddata_CV_LS(self):
        dens_u = nparam.KDE(tdat=[self.c1, self.o, self.o2], var_type='coo',
                             bw='cv_ls')
        npt.assert_allclose(dens_u.bw, [0.709195, 0.087333, 0.092500],
                            atol=1e-6)

        # Matches R to 3 decimals; results seem more stable than with R.
        # Can be checked with following code:
        ## import rpy2.robjects as robjects
        ## from rpy2.robjects.packages import importr
        ## NP = importr('np')
        ## r = robjects.r
        ## D = {"S1": robjects.FloatVector(c1), "S2":robjects.FloatVector(c2),
        ##      "S3":robjects.FloatVector(c3), "S4":robjects.FactorVector(o),
        ##      "S5":robjects.FactorVector(o2)}
        ## df = robjects.DataFrame(D)
        ## formula = r('~S1+ordered(S4)+ordered(S5)')
        ## r_bw = NP.npudensbw(formula, data=df, bwmethod='cv.ls')

    def test_pdf_mixeddata_LS_vs_ML(self):
        dens_ls = nparam.KDE(tdat=[self.c1, self.o, self.o2], var_type='coo',
                             bw='cv_ls')
        dens_ml = nparam.KDE(tdat=[self.c1, self.o, self.o2], var_type='coo',
                             bw='cv_ml')
        npt.assert_allclose(dens_ls.bw, dens_ml.bw, atol=0, rtol=0.5)

    def test_pdf_mixeddata_CV_ML(self):
        # Test ML cross-validation
        dens_ml = nparam.KDE(tdat=[self.c1, self.o, self.c2], var_type='coc',
                             bw='cv_ml')
        R_bw = [1.021563, 2.806409e-14, 0.5142077]
        npt.assert_allclose(dens_ml.bw, R_bw, atol=0.1, rtol=0.1)

    @dec.slow
    def test_pdf_continuous(self):
        # Test for only continuous data
        dens = nparam.KDE(tdat=[self.growth, self.Italy_gdp],
                            var_type='cc', bw='cv_ls')
        # take the first data points from the training set
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [1.6202284, 0.7914245, 1.6084174, 2.4987204, 1.3705258]

        ## CODE TO REPRODUCE THE RESULTS IN R
        ## library(np)
        ## data(oecdpanel)
        ## data (Italy)
        ## bw <-npudensbw(formula = ~oecdpanel$growth[1:50] + Italy$gdp[1:50],
        ## bwmethod ='cv.ls')
        ## fhat <- fitted(npudens(bws=bw))
        ## fhat[1:5]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    def test_pdf_ordered(self):
        # Test for only ordered data
        dens = nparam.KDE(tdat=[self.oecd], var_type='o', bw='cv_ls')
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [0.7236395, 0.7236395, 0.2763605, 0.2763605, 0.7236395]
        # lower tol here. only 2nd decimal
        npt.assert_allclose(sm_result, R_result, atol=1e-1)

    @dec.slow
    def test_unordered_CV_LS(self):
        dens = nparam.KDE(tdat=[self.growth, self.oecd],
                           var_type='cu', bw='cv_ls')
        R_result = [0.0052051, 0.05835941]
        npt.assert_allclose(dens.bw, R_result, atol=1e-2)

    def test_continuous_cdf(self, edat=None):
        dens = nparam.KDE(tdat=[self.Italy_gdp, self.growth],
                            var_type='cc', bw='cv_ml')
        sm_result = dens.cdf()[0:5]
        R_result = [0.192180770, 0.299505196, 0.557303666,
                    0.513387712, 0.210985350]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    def test_mixeddata_cdf(self, edat=None):
        dens = nparam.KDE(tdat=[self.Italy_gdp, self.oecd], var_type='cu',
                           bw='cv_ml')
        sm_result = dens.cdf()[0:5]
        R_result = [0.54700010, 0.65907039, 0.89676865, 0.74132941, 0.25291361]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    @dec.slow
    def test_continuous_cvls_efficient(self):
        N = 1000
        np.random.seed(12345)
        C1 = np.random.normal(size=(N, ))
        C2 = np.random.normal(2, 1, size=(N, ))
        C3 = np.random.beta(0.5,0.2, size=(N,))
        noise = np.random.normal(size=(N, ))
        Y = 0.3 +1.2 * C1 - 0.9 * C2

        dens_efficient = nparam.KDE(tdat=[Y, C1],
                                    var_type='cc', bw='cv_ls',
                                    defaults=SetDefaults(efficient=True,
                                                         n_sub=100))
        dens = nparam.KDE(tdat=[Y, C1], var_type='cc', bw='cv_ls',
                          defaults=SetDefaults(efficient=False))
        print dens.bw
        print dens_efficient.bw

        npt.assert_allclose(dens.bw, dens_efficient.bw, atol=0.1, rtol = 0.2)
        print "test_continuous_cvls_efficient successful"

    @dec.slow
    def test_continuous_cvml_efficient(self):
        N = 1000
        np.random.seed(12345)
        C1 = np.random.normal(size=(N, ))
        C2 = np.random.normal(2, 1, size=(N, ))
        C3 = np.random.beta(0.5,0.2, size=(N,))
        noise = np.random.normal(size=(N, ))
        Y = 0.3 +1.2 * C1 - 0.9 * C2

        dens_efficient = nparam.KDE(tdat=[Y, C1],
                                    var_type='cc', bw='cv_ml',
                                    defaults=SetDefaults(efficient=True,
                                                         n_sub=100))
        dens = nparam.KDE(tdat=[Y, C1], var_type='cc', bw='cv_ml',
                          defaults=SetDefaults(efficient=False))
        print dens.bw
        print dens_efficient.bw

        npt.assert_allclose(dens.bw, dens_efficient.bw, atol=0.1, rtol = 0.2)
        print "test_continuous_cvml_efficient successful"

    @dec.slow
    def test_efficient_notrandom(self):
        N = 1000
        np.random.seed(12345)
        C1 = np.random.normal(size=(N, ))
        C2 = np.random.normal(2, 1, size=(N, ))
        C3 = np.random.beta(0.5,0.2, size=(N,))
        noise = np.random.normal(size=(N, ))
        Y = 0.3 +1.2 * C1 - 0.9 * C2

        dens_efficient = nparam.KDE(tdat=[Y, C1], var_type='cc', bw='cv_ml',
                            defaults=SetDefaults(efficient=True, randomize=False, n_sub=100))
        dens = nparam.KDE(tdat=[Y, C1],
                            var_type='cc', bw='cv_ml')
        print dens.bw
        print dens_efficient.bw

        npt.assert_allclose(dens.bw, dens_efficient.bw, atol=0.1, rtol = 0.2)
        print "test_efficient_notrandom successful"

class TestCKDE(MyTest):
    @dec.slow
    def test_mixeddata_CV_LS(self):
        dens_ls = nparam.ConditionalKDE(tydat=[self.Italy_gdp],
                              txdat=[self.Italy_year],
                              dep_type='c', indep_type='o', bw='cv_ls')
        # Values from the estimation in R with the same data
        npt.assert_allclose(dens_ls.bw, [1.6448, 0.2317373], atol=1e-3)

    def test_continuous_CV_ML(self):
        dens_ml = nparam.ConditionalKDE(tydat=[self.Italy_gdp],
                               txdat=[self.growth], dep_type='c',
                               indep_type='c', bw='cv_ml')
        # Results from R
        npt.assert_allclose(dens_ml.bw, [0.5341164, 0.04510836], atol=1e-3)

    @dec.slow
    def test_unordered_CV_LS(self):
        dens_ls = nparam.ConditionalKDE(tydat=[self.oecd],
                              txdat=[self.growth],
                              dep_type='u', indep_type='c', bw='cv_ls')

    def test_pdf_continuous(self):
        dens = nparam.ConditionalKDE(tydat=[self.growth], txdat=[self.Italy_gdp],
                            dep_type='c', indep_type='c', bw='cv_ml')
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [11.97964, 12.73290, 13.23037, 13.46438, 12.22779]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    @dec.slow
    def test_pdf_mixeddata(self):
        dens = nparam.ConditionalKDE(tydat=[self.Italy_gdp],
                           txdat=[self.Italy_year], dep_type='c',
                           indep_type='o', bw='cv_ls')
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [0.08469226, 0.01737731, 0.05679909, 0.09744726, 0.15086674]

        ## CODE TO REPRODUCE IN R
        ## library(np)
        ## data (Italy)
        ## bw <- npcdensbw(formula =
        ## Italy$gdp[1:50]~ordered(Italy$year[1:50]),bwmethod='cv.ls')
        ## fhat <- fitted(npcdens(bws=bw))
        ## fhat[1:5]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    def test_continuous_normal_ref(self):
        # test for normal reference rule of thumb with continuous data
        dens_nm = nparam.ConditionalKDE(tydat=[self.Italy_gdp],
                              txdat=[self.growth], dep_type='c',
                              indep_type='c', bw='normal_reference')
        sm_result = dens_nm.bw
        R_result = [1.283532, 0.01535401]
        # Here we need a smaller tolerance.check!
        npt.assert_allclose(sm_result, R_result, atol=1e-1)

    def test_continuous_cdf(self):
        dens_nm = nparam.ConditionalKDE(tydat=[self.Italy_gdp],
                              txdat=[self.growth],
                              dep_type='c',
                              indep_type='c', bw='normal_reference')
        sm_result = dens_nm.cdf()[0:5]
        R_result = [0.81304920, 0.95046942, 0.86878727, 0.71961748, 0.38685423]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    @dec.slow
    def test_mixeddata_cdf(self):
        dens = nparam.ConditionalKDE(tydat=[self.Italy_gdp],
                           txdat=[self.Italy_year],
                           dep_type='c', indep_type='o', bw='cv_ls')
        sm_result = dens.cdf()[0:5]
        R_result = [0.8118257, 0.9724863, 0.8843773, 0.7720359, 0.4361867]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    @dec.slow
    def test_continuous_cvml_efficient(self):
        N = 1000
        np.random.seed(12345)
        O = np.random.binomial(2, 0.5, size=(N, ))
        O2 = np.random.binomial(2, 0.5, size=(N, ))
        C1 = np.random.normal(size=(N, ))
        C2 = np.random.normal(2, 1, size=(N, ))
        C3 = np.random.beta(0.5,0.2, size=(N,))
        noise = np.random.normal(size=(N, ))
        b0 = 3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        b3 = 2.3
        Y = b0+ b1 * C1 + b2*O  + noise

        dens_efficient = nparam.ConditionalKDE(tydat=[Y], txdat=[C1],
                    dep_type='c', indep_type='c', bw='cv_ml',
                    defaults=SetDefaults(efficient=True, n_sub=150))

        dens = nparam.ConditionalKDE(tydat=[Y], txdat=[C1],
                           dep_type='c', indep_type='c', bw='cv_ml')

        npt.assert_allclose(dens.bw, dens_efficient.bw, atol=0.1, rtol = 0.15)
        print "test_continuous_cvml_efficient successful"


class TestReg(MyTest):

    def test_ordered_lc_cvls(self):
        model = nparam.Reg(tydat=[self.Italy_gdp], txdat=[self.Italy_year],
                           reg_type='lc', var_type='o', bw='cv_ls')
        sm_bw = model.bw
        R_bw = 0.1390096

        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        sm_mfx = sm_mfx[0:5]
        R_mean = 6.190486

        sm_R2 = model.r_squared()
        R_R2 = 0.1435323

        ## CODE TO REPRODUCE IN R
        ## library(np)
        ## data(Italy)
        ## attach(Italy)
        ## bw <- npregbw(formula=gdp[1:50]~ordered(year[1:50]))
        npt.assert_allclose(sm_bw, R_bw, atol=1e-2)
        npt.assert_allclose(sm_mean, R_mean, atol=1e-2)
        npt.assert_allclose(sm_R2, R_R2, atol=1e-2)

        print "test_ordered_lc_cvls - successful"

    def test_continuousdata_lc_cvls(self):
        model = nparam.Reg(tydat=[self.y], txdat=[self.c1, self.c2],
                           reg_type='lc', var_type='cc', bw='cv_ls')
        # Bandwidth
        sm_bw = model.bw
        R_bw = [0.6163835, 0.1649656]
        # Conditional Mean
        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        sm_mfx = sm_mfx[0:5]
        R_mean = [31.49157, 37.29536, 43.72332, 40.58997, 36.80711]
        # R-Squared
        sm_R2 = model.r_squared()
        R_R2 = 0.956381720885

        npt.assert_allclose(sm_bw, R_bw, atol=1e-2)
        npt.assert_allclose(sm_mean, R_mean, atol=1e-2)
        npt.assert_allclose(sm_R2, R_R2, atol=1e-2)
        print "test_continuousdata_lc_cvls - successful"


    def test_continuousdata_ll_cvls(self):
        model = nparam.Reg(tydat=[self.y], txdat=[self.c1, self.c2],
                           reg_type='ll', var_type='cc', bw='cv_ls')

        sm_bw = model.bw
        R_bw = [1.717891, 2.449415]
        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        sm_mfx = sm_mfx[0:5]
        R_mean = [31.16003, 37.30323, 44.49870, 40.73704, 36.19083]

        sm_R2 = model.r_squared()
        R_R2 = 0.9336019

        npt.assert_allclose(sm_bw, R_bw, atol=1e-2)
        npt.assert_allclose(sm_mean, R_mean, atol=1e-2)
        npt.assert_allclose(sm_R2, R_R2, atol=1e-2)
        print "test_continuousdata_ll_cvls - successful"


    def test_continuous_mfx_ll_cvls(self, file_name='RegData.csv'):
        N = 200
        np.random.seed(1234)
        O = np.random.binomial(2, 0.5, size=(N, ))
        O2 = np.random.binomial(2, 0.5, size=(N, ))
        C1 = np.random.normal(size=(N, ))
        C2 = np.random.normal(2, 1, size=(N, ))
        C3 = np.random.beta(0.5,0.2, size=(N,))
        noise = np.random.normal(size=(N, ))
        b0 = 3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        b3 = 2.3
        Y = b0+ b1 * C1 + b2*C2+ b3 * C3 + noise
        model = nparam.Reg(tydat=[Y], txdat=[C1, C2, C3],
                            reg_type='ll', var_type='ccc', bw='cv_ls')
        sm_bw = model.bw
        print "Bandwidth: ", sm_bw
        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        sm_R2 = model.r_squared()
        print "R2: ", sm_R2
        print "test_continuous_mfx_ll_cvls - successful"
        print model
        npt.assert_allclose(sm_mfx[0,:], [b1,b2,b3], rtol=2e-1)
        self.write2file(file_name, (Y, C1, C2, C3))


    def test_mixed_mfx_ll_cvls(self, file_name='RegData.csv'):
        N = 200
        np.random.seed(1234)
        O = np.random.binomial(2, 0.5, size=(N, ))
        O2 = np.random.binomial(2, 0.5, size=(N, ))
        C1 = np.random.normal(size=(N, ))
        C2 = np.random.normal(2, 1, size=(N, ))
        C3 = np.random.beta(0.5,0.2, size=(N,))
        noise = np.random.normal(size=(N, ))
        b0 = 3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        b3 = 2.3
        Y = b0+ b1 * C1 + b2*C2+ b3 * O + noise
        model = nparam.Reg(tydat=[Y], txdat=[C1, C2, O],
                            reg_type='ll', var_type='cco', bw='cv_ls')
        sm_bw = model.bw
        sm_mean, sm_mfx = model.fit()
        sm_R2 = model.r_squared()
        print "test_continuous_mfx_ll_cvls - successful"
        npt.assert_allclose(sm_mfx[0,:], [b1,b2,b3], rtol=2e-1)
        #self.write2file(file_name, (Y, C1, C2, C3))


    def test_mfx_nonlinear_ll_cvls(self, file_name='RegData.csv'):
        N = 200
        np.random.seed(1234)
        O = np.random.binomial(2, 0.5, size=(N, ))
        O2 = np.random.binomial(2, 0.5, size=(N, ))
        C1 = np.random.normal(size=(N, ))
        C2 = np.random.normal(2, 1, size=(N, ))
        C3 = np.random.beta(0.5,0.2, size=(N,))
        noise = np.random.normal(size=(N, ))
        b0 = 3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        b3 = 2.3
        Y = b0+ b1 * C1 * C2 + b3 * C3 + noise
        model = nparam.Reg(tydat=[Y], txdat=[C1, C2, C3],
                            reg_type='ll', var_type='ccc', bw='cv_ls')
        sm_bw = model.bw
        sm_mean, sm_mfx = model.fit()
        sm_R2 = model.r_squared()
        # Theoretical marginal effects
        mfx1 = b1 * C2
        mfx2 = b1 * C1
        #npt.assert_allclose(sm_mfx[:,0], mfx1, rtol=2e-1)
        #npt.assert_allclose(sm_mfx[0:10,1], mfx2[0:10], rtol=2e-1)
        npt.assert_allclose(sm_mean, Y, rtol = 2e-1)
        #self.write2file(file_name, (Y, C1, C2, C3))
        print "test_continuous_mfx_ll_cvls - successful"

    @dec.slow
    def test_continuous_cvls_efficient(self):
        N = 1000
        np.random.seed(12345)
        O = np.random.binomial(2, 0.5, size=(N, ))
        O2 = np.random.binomial(2, 0.5, size=(N, ))
        C1 = np.random.normal(size=(N, ))
        C2 = np.random.normal(2, 1, size=(N, ))
        C3 = np.random.beta(0.5,0.2, size=(N,))
        noise = np.random.normal(size=(N, ))
        b0 = 3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        b3 = 2.3
        Y = b0+ b1 * C1 + b2*C2

        model_efficient = nparam.Reg(tydat=[Y], txdat=[C1],
                            reg_type='lc', var_type='c', bw='cv_ls',
                    defaults=SetDefaults(efficient=True, n_sub=100))

        print model_efficient.bw
        model = nparam.Reg(tydat=[Y], txdat=[C1],
                            reg_type='ll', var_type='c', bw='cv_ls')
        print model.bw
        #print model_efficient.bw
        print "----"*10

        npt.assert_allclose(model.bw, model_efficient.bw, atol=5e-2, rtol=1e-1)

    @dec.slow
    def test_censored_ll_cvls(self):
        N = 200
        np.random.seed(1234)
        C1 = np.random.normal(size=(N, ))
        C2 = np.random.normal(2, 1, size=(N, ))
        C3 = np.random.beta(0.5,0.2, size=(N,))
        noise = np.random.normal(size=(N, ))
        Y = 0.3 +1.2 * C1 - 0.9 * C2 + noise
        Y[Y>0] = 0  # censor the data
        model = nparam.CensoredReg(tydat=[Y], txdat=[C1, C2], reg_type='ll',
                                   var_type='cc', bw='cv_ls', censor_val=0)
        sm_mean, sm_mfx = model.fit()
        npt.assert_allclose(sm_mfx[0,:], [1.2, -0.9], rtol = 2e-1)

    @dec.slow
    def test_continuous_lc_aic(self):
        N = 200
        np.random.seed(1234)
        C1 = np.random.normal(size=(N, ))
        C2 = np.random.normal(2, 1, size=(N, ))
        C3 = np.random.beta(0.5,0.2, size=(N,))
        noise = np.random.normal(size=(N, ))
        Y = 0.3 +1.2 * C1 - 0.9 * C2 + noise
        #self.write2file('RegData.csv', (Y, C1, C2))

        #CODE TO PRODUCE BANDWIDTH ESTIMATION IN R
        #library(np)
        #data <- read.csv('RegData.csv', header=FALSE)
        #bw <- npregbw(formula=data$V1 ~ data$V2 + data$V3,
        #                bwmethod='cv.aic', regtype='lc')
        model = nparam.Reg(tydat=[Y], txdat=[C1, C2],
                            reg_type='lc', var_type='cc', bw='aic')
        R_bw = [0.4017893, 0.4943397]  # Bandwidth obtained in R
        npt.assert_allclose(model.bw, R_bw, rtol = 1e-3)


    @dec.slow
    def test_significance_continuous(self):
        N = 250
        np.random.seed(12345)
        O = np.random.binomial(2, 0.5, size=(N, ))
        O2 = np.random.binomial(2, 0.5, size=(N, ))
        C1 = np.random.normal(size=(N, ))
        C2 = np.random.normal(2, 1, size=(N, ))
        C3 = np.random.beta(0.5,0.2, size=(N,))
        noise = np.random.normal(size=(N, ))
        b0 = 3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        b3 = 2.3
        Y = b1 * C1 + b2 * C2 + noise

        bw=[11108137.1087194, 1333821.85150218]  # This is the cv_ls bandwidth estimated earlier

        model = nparam.Reg(tydat=[Y], txdat=[C1, C3],
                            reg_type='ll', var_type='cc', bw=bw)
        nboot = 45  # Number of bootstrap samples
        sig_var12 = model.sig_test([0,1], nboot=nboot)  # H0: b1 = 0 and b2 = 0
        npt.assert_equal(sig_var12 == 'Not Significant', False)
        sig_var1 = model.sig_test([0], nboot=nboot)  # H0: b1 = 0
        npt.assert_equal(sig_var1 == 'Not Significant', False)
        sig_var2 = model.sig_test([1], nboot=nboot)  # H0: b2 = 0
        npt.assert_equal(sig_var2 == 'Not Significant', True)
        print "test_significance_continuous ran successfully"

    @dec.slow
    def test_significance_discrete(self):

        N = 200
        np.random.seed(12345)
        O = np.random.binomial(2, 0.5, size=(N, ))
        O2 = np.random.binomial(2, 0.5, size=(N, ))
        C1 = np.random.normal(size=(N, ))
        C2 = np.random.normal(2, 1, size=(N, ))
        C3 = np.random.beta(0.5,0.2, size=(N,))
        noise = np.random.normal(size=(N, ))
        b0 = 3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        b3 = 2.3
        Y = b1 * O + b2 * C2 + noise

        bw= [3.63473198e+00, 1.21404803e+06]
                 # This is the cv_ls bandwidth estimated earlier
        # The cv_ls bandwidth was estimated earlier to save time
        model = nparam.Reg(tydat=[Y], txdat=[O, C3],
                            reg_type='ll', var_type='oc', bw=bw)
        # This was also tested with local constant estimator
        nboot = 45  # Number of bootstrap samples
        sig_var1 = model.sig_test([0], nboot=nboot)  # H0: b1 = 0
        npt.assert_equal(sig_var1 == 'Not Significant', False)
        sig_var2 = model.sig_test([1], nboot=nboot)  # H0: b2 = 0
        npt.assert_equal(sig_var2 == 'Not Significant', True)

    @dec.slow
    def test_semi_linear_model(self):
        N = 800
        np.random.seed(1234)
        C1 = np.random.normal(0,2, size=(N, ))
        C2 = np.random.normal(2, 1, size=(N, ))
        e = np.random.normal(size=(N, ))
        b1 = 1.3
        b2 = -0.7
        Y = b1 * C1 + np.exp(b2 * C2) + e
        model = SemiLinear(tydat=[Y], txdat=[C1], tzdat=[C2],
                           var_type='c', l_K=1)
        b_hat = np.squeeze(model.b)
        # Only tests for the linear part of the regression
        # Currently doesn't work well with the nonparametric part
        # Needs some more work
        npt.assert_allclose(b1, b_hat, rtol=0.1)
