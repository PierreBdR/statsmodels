"""
Test Results for discrete models from Stata
"""
import os
import numpy as np

#### Discrete Model Tests ####
# Note that there is a slight refactor of the classes, so that one dataset
# might be used for more than one model

cur_dir = os.path.abspath(os.path.dirname(__file__))

class Anes():
    def __init__(self):
        """
        Results are from Stata 11 (checked vs R nnet package).
        """
        self.nobs = 944

    def mnlogit_basezero(self):
        params = [-.01153598, .29771435, -.024945, .08249144, .00519655,
                -.37340167, -.08875065, .39166864, -.02289784, .18104276,
                .04787398, -2.2509132, -.1059667, .57345051, -.01485121,
                -.00715242, .05757516, -3.6655835, -.0915567, 1.2787718,
                -.00868135, .19982796, .08449838, -7.6138431, -.0932846,
                1.3469616, -.01790407, .21693885, .08095841, -7.0604782,
                -.14088069, 2.0700801, -.00943265, .3219257, .10889408,
                -12.105751]
        self.params = np.reshape(params, (6,-1), order='F')
        bse = [.0342823657, .093626795, .0065248584, .0735865799,
                .0176336937, .6298376313, .0391615553, .1082386919,
                .0079144618, .0852893563, .0222809297, .7631899491,
                .0570382292, .1585481337, .0113313133, .1262913234,
                .0336142088, 1.156541492, .0437902764, .1288965854,
                .0084187486, .0941250559, .0261963632, .9575809602,
                .0393516553, .1171860107, .0076110152, .0850070091,
                .0229760791, .8443638283, .042138047, .1434089089,
                .0081338625, .0910979921, .025300888, 1.059954821]
        self.bse = np.reshape(bse, (6,-1), order='F')
        self.yhat = np.loadtxt(os.path.join(cur_dir,'yhat_mnlogit.csv'))
        self.phat = np.loadtxt(os.path.join(cur_dir,'phat_mnlogit.csv'))
        self.cov_params = None
        self.llf = -1461.922747312
        self.llnull = -1750.34670999
        self.llr = 576.8479253554
        self.llr_pvalue = 1.8223179e-102
        self.prsquared = .1647810465387
        self.df_model = 30
        self.df_resid = 944 - 36
        self.J = 7
        self.K = 6
        self.aic = 2995.84549462
        self.bic = 3170.45003661
        z =  [-.3364988051, 3.179798597,  -3.823070772, 1.121012042,
            .2946945327, -.5928538661, -2.266269864, 3.618564069,
            -2.893164162, 2.122688754, 2.148652536, -2.949348555,
            -1.857818873, 3.616885888, -1.310634214, -.0566342868,
            1.712822091, -3.169435381, -2.090799808, 9.920912816,
            -1.031191864, 2.123004903, 3.225576554, -7.951122047,
            -2.370538224, 11.49421878, -2.352389066, 2.552011323,
            3.523595639, -8.361890935, -3.34331327, 14.43480847,
            -1.159676452, 3.533839715, 4.303962885, -11.42100649]
        self.z = np.reshape(z, (6,-1), order='F')
        pvalues = [0.7364947525, 0.0014737744, 0.0001317999, 0.2622827367,
            0.7682272401, 0.5532789548, 0.0234348654, 0.0002962422,
            0.0038138191, 0.0337799420, 0.0316619538, 0.0031844460,
            0.0631947400, 0.0002981687, 0.1899813744, 0.9548365214,
            0.0867452747, 0.0015273542, 0.0365460134, 3.37654e-23,
            0.3024508550, 0.0337534410, 0.0012571921, 1.84830e-15,
            0.0177622072, 1.41051e-30, 0.0186532528, 0.0107103038,
            0.0004257334, 6.17209e-17, 0.0008278439, 3.12513e-47,
            0.2461805610, 0.0004095694, 0.0000167770, 3.28408e-30]
        self.pvalues = np.reshape(pvalues, (6,-1), order='F')
        conf_int = [[[-0.0787282, 0.0556562], [0.1142092, 0.4812195],
            [-0.0377335, -0.0121565], [-0.0617356, 0.2267185], [-0.0293649,
            0.0397580], [-1.6078610, 0.8610574]], [[-0.1655059, -0.0119954],
            [0.1795247,	0.6038126], [-0.0384099, -0.0073858], [0.0138787,
            0.3482068], [0.0042042,	0.0915438], [-3.7467380, -0.7550884]],
            [[-0.2177596, 0.0058262], [0.2627019,	0.8841991], [-0.0370602,
            0.0073578], [-0.2546789, 0.2403740], [-0.0083075, 0.1234578],
            [-5.9323630,-1.3988040]],[[-0.1773841, -0.0057293], [1.0261390,
            1.5314040], [-0.0251818, 0.0078191], [0.0153462,	0.3843097],
            [0.0331544,	0.1358423], [-9.4906670, -5.7370190]], [[-0.1704124,
            -0.0161568], [1.1172810,	1.5766420], [-0.0328214, -0.0029868],
            [0.0503282,	0.3835495], [0.0359261,	0.1259907], [-8.7154010,
            -5.4055560]], [[-0.2234697, -0.0582916], [1.7890040, 2.3511560],
            [-0.0253747, 0.0065094], [0.1433769, 0.5004745], [0.0593053,
            0.1584829], [-14.1832200, -10.0282800]]]
        self.conf_int = np.asarray(conf_int)

        # margins, dydx(*) predict(outcome(#))
        self.margeff_dydx_overall = np.array([
            [0.00868085993550, -0.09779854015456, 0.00272556969847,
                -0.01992376579372, -0.00603133322764],
            [0.00699386733148, -0.05022430802614, -0.00211003909752,
                -0.00536980000265, -0.00554366741814],
            [-0.00391040848820, -0.02824717135857, -0.00100551299310,
                0.00664337806861, 0.00097987356999],
            [-0.00182580888015, -0.00573744730031, -0.00004249256428,
                -0.00546669558488, 0.00054101121854],
            [-0.00098558129923, 0.01985550937033, 0.00047972250012,
                0.00172605778905, 0.00211291403209],
            [-0.00153469551647, 0.03755346502013, -0.00068531143399,
                0.00472471794347, 0.00254733486106],
            [-0.00741820702809, 0.12459834487569, 0.00063806819375,
                0.01766610701188, 0.00539385283759]
                ]).T
        self.margeff_dydx_overall_se = np.array([
            [.0038581061, .0080471125, .0007068488, .0082318967, .0020261706],
            [.003904378, .0073600286, .000756431, .0084381578, .0020482238],
            [.003137126, .0056813182, .0006601377, .0068932588, .0018481806],
            [.0019427783, .0031904763, .0003865411, .004361789, .0011523221],
            [.0029863227, .0054076092, .0005886612, .0064426365, .0018886818],
            [.0035806552, .0069497362, .000722511, .0078287717, .0022352393],
            [.0033641608, .008376629, .0006774697, .0073505286, .0021660086]
                ]).T

        self.margeff_dydx_mean = np.array([
            [0.01149887431225, -0.13784207091973, 0.00273313385873,
                -0.02542974260540, -0.00855346837482],
            [0.01114846831102, -0.09864273512889, -0.00222435063712,
                -0.01214617126321, -0.00903581444579],
            [-0.00381702868421, -0.05132297961269, -0.00116763216994,
                0.00624203027060, 0.00021912081810],
            [-0.00233455327258, -0.00928554037343, -0.00000206561214,
                -0.00775415690571, 0.00060004460394],
            [-0.00352579921274, 0.06412187169362, 0.00073938948643,
                0.00747778063206, 0.00459965010365],
            [-0.00574308219449, 0.11126535089794, -0.00057337915464,
                0.01467424346725, 0.00641760846097],
            [-0.00722687818452, 0.12170608820238, 0.00049490419675,
                0.01693601418978, 0.00575285798725]]).T
        self.margeff_dydx_mean_se = np.array([
            [.0043729758, .0110343353, .0008149907, .0092551389, .0023752071],
            [.004875051, .0124746358, .0009613152, .0105665812, .0026524426],
            [.0040718954, .0103613938, .0008554615, .0089931297, .0024374625],
            [.0026430804, .0070845916, .0005364369, .0057654258, .0015988838],
            [.0037798151, .0103849291, .0007393481, .0082021938, .0023489261],
            [.0045654631, .0130329403, .0009128134, .0100053262, .0028048602],
            [.0027682389, .0113292677, .0005325113, .0061289353, .0017330763]
            ]).T

        self.margeff_dydx_dummy_overall = np.array([
            [0.00549149574321, -0.05348235321783, 0.00298963549049,
                -0.01479461677951, -0.00332167981255, -0.26502967041815],
            [0.00345677928276, -0.00950322030929, -0.00189456107189,
                0.00033893662061, -0.00314690167350, -0.21040878091828],
            [-0.00645089013284, 0.00401746940204, -0.00083948249351,
                0.01114202556889, 0.00277069841472, -0.15967397659686],
            [-0.00215436802341, -0.00366545199370, -0.00000002297812,
                -0.00457368049644, 0.00065303026027, -0.00094772782001],
            [0.00058038428936, -0.00369080100124, 0.00035948233235,
                -0.00018863693013, 0.00079351293461, 0.12640653743480],
            [0.00217597030999, -0.01279456622853, -0.00091882392767,
                0.00001651192759, -0.00037998290789, 0.27175070356670],
            [-0.00309932483642, 0.07911868907484, 0.00030378521102,
                0.00805941631677, 0.00263129901425, 0.23790291475181]]).T
        self.margeff_dydx_dummy_overall_se = np.array([
            [.0037314453, .0094102332, .000688838, .0079744554, .0019365971,
                .0243914836],
            [.0038215262, .0095938828, .0007410885, .008259353, .0019984087,
                .0317628806],
            [.0031045718, .00785814, .0006504353, .0067892866, .0018060332,
                0.0262803561],
            [.0019756086, .0051031194, .0003862449, .0043621673, .0011796953,
                .0219999601],
            [.0029714074, .0081732018, .0005715192, .0064742872, .0019130195,
                .0331694192],
            [.0034443743, .0097296187, .0006774867, .0075996454, .0021993881,
                .038600835],
            [.0032003518, .0098741227, .0006335772, .0070902078, .0021003227,
                .0255727127]]).T

        self.margeff_eydx_dummy_overall = np.array([
            [.03939188, -.65758371, .01750922, -.12131806, -.03613241,
                -3.2132513],
            [.02752366, -.383165, -.00830021, -.03652935, -.03286046,
                -1.8741853],
            [-.05006681, -.2719659, -.00626481, .06525323, .01012554,
                -2.0058029],
            [-.05239558, -.22549142, .00025015, -.13104416, .01114517,
                -.27052009],
            [-.00296374, .25627809, .00140513, .03358712, .02296041,
                1.3302701],
            [.00328283, .2800168, -.0083912, .04332782, .01575863,
                1.8441023],
            [-.03257068, .98346111, -.00122118, .10847807, .0406456,
                2.9119099]]).T

        self.margeff_eydx_dummy_overall_se = np.array([
            [.0272085605, .0777760394, .0052427952, .0584011446, .0148618012,
                .5796921383],
            [.0262290023, .0724479385, .005174736, .0567743614, .0144447083,
                .3015738731],
            [.0321415498, .0895589422, .0067480662, .0701460193, .0190451865,
                .3904138447],
            [.0511305319, .1420904068, .0102342163, .1129912244, .0308618233,
                .3693799595],
            [.0340186217, .0991711703, .0065812158, .0737441012, .0212966336,
                .2346982385],
            [.0289250212, .0840662279, .0056743561, .0631772185, .0177278895,
                .2089516714],
            [.0318251305, .1085637405, .0062400589, .0699123044, .0201045606,
                .3727166284]]).T



class DiscreteL1(object):
    def __init__(self):
        """
        Special results for L1 models
        Uses the Spector data and a script to generate the baseline results
        """
        pass

    def logit(self):
        """
        Results generated with:
            data = sm.datasets.spector.load()
            data.exog = sm.add_constant(data.exog, prepend=True)
            alpha = 3 * np.array([0, 1, 1, 1])
            res2 = sm.Logit(data.endog, data.exog).fit_regularized(
                method="l1", alpha=alpha, disp=0, trim_mode='size',
                size_trim_tol=1e-5, acc=1e-10, maxiter=1000)
        """
        nan = np.nan
        self.params = [-4.10271595,  0., 0.15493781, 0.]
        self.conf_int = [[-9.15205122,  0.94661932], [nan, nan],
                [-0.06539482,  0.37527044], [ nan, nan]]
        self.bse = [ 2.5762388 ,         nan,  0.11241668,         nan]
        self.nnz_params = 2
        self.aic = 42.091439368583671
        self.bic =  45.022911174183122
        self.cov_params = [[ 6.63700638, nan, -0.28636261, nan],
                [nan, nan, nan, nan], [-0.28636261, nan,  0.01263751, nan],
                [nan, nan, nan, nan]]

    def sweep(self):
        """
        Results generated with
            params = np.zeros((3, 4))
            alphas = np.array(
                    [[0.1, 0.1, 0.1, 0.1],
                        [0.4, 0.4, 0.5, 0.5], [0.5, 0.5, 1, 1]])
            model = sm.Logit(data.endog, data.exog)
            for i in range(3):
                alpha = alphas[i, :]
                res2 = model.fit_regularized(method="l1", alpha=alpha, disp=0, acc=1e-10,
                        maxiter=1000, trim_mode='off')
                params[i, :] = res2.params
            print params
        """
        self.params = [[-10.37593611,   2.27080968,   0.06670638,   2.05723691],
       [ -5.32670811,   1.18216019,   0.01402395,   1.45178712],
       [ -3.92630318,   0.90126958,  -0.        ,   1.09498178]]

    def probit(self):
        """
        Results generated with
            data = sm.datasets.spector.load()
            data.exog = sm.add_constant(data.exog, prepend=True)
            alpha = np.array([0.1, 0.2, 0.3, 10])
            res2 = sm.Probit(data.endog, data.exog).fit_regularized(
                method="l1", alpha=alpha, disp=0, trim_mode='auto',
                auto_trim_tol=0.02, acc=1e-10, maxiter=1000)
        """
        nan = np.nan
        self.params = [-5.40476992,  1.25018458,  0.04744558,  0.        ]
        self.conf_int = [[-9.44077951, -1.36876033],
       [ 0.03716721,  2.46320194],
       [-0.09727571,  0.19216687],
       [        np.nan,         np.nan]]
        self.bse = [ 2.05922641,  0.61889778,  0.07383875,         np.nan]
        self.nnz_params = 3
        self.aic = 38.399773877542927
        self.bic = 42.796981585942106
        self.cov_params = [[ 4.24041339, -0.83432592, -0.06827915, nan],
       [-0.83432592,  0.38303447, -0.01700249,         nan],
       [-0.06827915, -0.01700249,  0.00545216,         nan],
       [        nan,         nan,         nan,         nan]]

    def mnlogit(self):
        """
        Results generated with
            anes_data = sm.datasets.anes96.load()
            anes_exog = anes_data.exog
            anes_exog = sm.add_constant(anes_exog, prepend=False)
            mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)

            alpha = 10 * np.ones((mlogit_mod.J - 1, mlogit_mod.K))
            alpha[-1,:] = 0
            mlogit_l1_res = mlogit_mod.fit_regularized(
            method='l1', alpha=alpha, trim_mode='auto', auto_trim_tol=0.02,
            acc=1e-10)
        """
        self.params = [[ 0.00100163, -0.05864195, -0.06147822, -0.04769671, -0.05222987,
            -0.09522432],
           [ 0.        ,  0.03186139,  0.12048999,  0.83211915,  0.92330292,
             1.5680646 ],
           [-0.0218185 , -0.01988066, -0.00808564, -0.00487463, -0.01400173,
            -0.00562079],
           [ 0.        ,  0.03306875,  0.        ,  0.02362861,  0.05486435,
             0.14656966],
           [ 0.        ,  0.04448213,  0.03252651,  0.07661761,  0.07265266,
             0.0967758 ],
           [ 0.90993803, -0.50081247, -2.08285102, -5.26132955, -4.86783179,
            -9.31537963]]
        self.conf_int = [[[ -0.0646223 ,   0.06662556],
            [         np.nan,          np.nan],
            [ -0.03405931,  -0.00957768],
            [         np.nan,          np.nan],
            [         np.nan,          np.nan],
            [  0.26697895,   1.55289711]],

           [[ -0.1337913 ,   0.01650741],
            [ -0.14477255,   0.20849532],
            [ -0.03500303,  -0.00475829],
            [ -0.11406121,   0.18019871],
            [  0.00479741,   0.08416684],
            [ -1.84626136,   0.84463642]],

           [[ -0.17237962,   0.04942317],
            [ -0.15146029,   0.39244026],
            [ -0.02947379,   0.01330252],
            [         np.nan,          np.nan],
            [ -0.02501483,   0.09006785],
            [ -3.90379391,  -0.26190812]],

           [[ -0.12938296,   0.03398954],
            [  0.62612955,   1.03810876],
            [ -0.02046322,   0.01071395],
            [ -0.13738534,   0.18464256],
            [  0.03017236,   0.12306286],
            [ -6.91227465,  -3.61038444]],

           [[ -0.12469773,   0.02023799],
            [  0.742564  ,   1.10404183],
            [ -0.02791975,  -0.00008371],
            [ -0.08491561,   0.19464431],
            [  0.0332926 ,   0.11201273],
            [ -6.29331126,  -3.44235233]],

           [[ -0.17165567,  -0.01879296],
            [  1.33994079,   1.79618841],
            [ -0.02027503,   0.00903345],
            [ -0.00267819,   0.29581751],
            [  0.05343135,   0.14012026],
            [-11.10419107,  -7.52656819]]]

        self.bse = [[ 0.03348221,  0.03834221,  0.05658338,  0.04167742,  0.03697408,
                 0.03899631],
               [        np.nan,  0.09012101,  0.13875269,  0.10509867,  0.09221543,
                 0.11639184],
               [ 0.00624543,  0.00771564,  0.01091253,  0.00795351,  0.00710116,
                 0.00747679],
               [        np.nan,  0.07506769,         np.nan,  0.08215148,  0.07131762,
                 0.07614826],
               [        np.nan,  0.02024768,  0.02935837,  0.02369699,  0.02008204,
                 0.02211492],
               [ 0.32804638,  0.68646613,  0.92906957,  0.84233441,  0.72729881,
                 0.91267567]]

        self.nnz_params = 32
        self.aic = 3019.4391360294126
        self.bic = 3174.6431733460686


class Spector():
    """
    Results are from Stata 11
    """
    def __init__(self):
        self.nobs = 32

    def logit(self):
        self.params = [2.82611297201, .0951576702557, 2.37868772835,
                -13.0213483201]
        self.cov_params = [[1.59502033639, -.036920566629, .427615725153,
                -4.57347950298], [-.036920566629, .0200375937069,
                .0149126464275, -.346255757562], [.427615725153 ,
                .0149126464275, 1.13329715236, -2.35916128427],
                [-4.57347950298, -.346255757562, -2.35916128427,
                24.3179625937]]
        self.bse = [1.26294114526, .141554207662, 1.06456430165, 4.93132462871]
        self.phat = np.array([ .02657799236476,
                  .05950126051903,
                  .18725991249084,
                  .02590163610876,
                  .56989300251007,
                  .03485824912786,
                  .02650404907763,
                  .05155897513032,
                  .11112663894892,
                  .69351142644882,
                  .02447037212551,
                  .18999740481377,
                  .32223951816559,
                   .1932111531496,
                  .36098992824554,
                  .03018374741077,
                  .05362640321255,
                  .03858831897378,
                  .58987241983414,
                  .66078591346741,
                  .06137581542134,
                  .90484726428986,
                  .24177247285843,
                  .85209089517593,
                   .8382905125618,
                  .48113295435905,
                  .63542068004608,
                  .30721867084503,
                  .84170418977737,
                  .94534027576447,
                  .52911710739136,
                   .1110308393836])
        self.yhat = np.array([-3.6007342338562,
                 -2.7604126930237,
                 -1.4679137468338,
                 -3.6272060871124,
                  .28141465783119,
                 -3.3209850788116,
                 -3.6035962104797,
                 -2.9120934009552,
                 -2.0792844295502,
                  .81658720970154,
                 -3.6855175495148,
                 -1.4500269889832,
                 -.74349880218506,
                  -1.429278254509,
                 -.57107019424438,
                 -3.4698030948639,
                 -2.8705959320068,
                 -3.2154531478882,
                  .36343798041344,
                  .66679841279984,
                 -2.7273993492126,
                  2.2522828578949,
                 -1.1429864168167,
                  1.7510952949524,
                  1.6455633640289,
                 -.07550399750471,
                  .55554306507111,
                 -.81315463781357,
                  1.6709630489349,
                  2.8504176139832,
                  .11660042405128,
                 -2.0802545547485])
        self.llf = -12.8896334653335
        self.llnull = -20.5917296966173
        self.df_model = 3
        self.df_resid = 32 - 4  #TODO: is this right? not reported in stata
        self.llr = 15.4041924625676
        self.prsquared = .374038332124624
        self.llr_pvalue = .00150187761112892
        self.aic = 33.779266930667
        self.bic = 39.642210541866
        self.z = [2.237723415, 0.6722348408, 2.234423721, -2.640537645]
        self.conf_int = [[.3507938,5.301432],[-.1822835,.3725988],[.29218,
                4.465195],[-22.68657,-3.35613]]
        self.pvalues = [.0252390974, .5014342039, .0254552063, .0082774596]

        # taken from margins command
        self.margeff_nodummy_dydx = [.36258084688424,.01220841099085,
                .30517768382304]
        self.margeff_nodummy_dydx_se = [.1094412, .0177942, .0923796]
        self.margeff_nodummy_dydxmean = [.53385885781692,.01797548988961,
                .44933926079386]
        self.margeff_nodummy_dydxmean_se = [.237038, .0262369, .1967626]
        self.margeff_nodummy_dydxmedian = [.25009492465091,.00842091261329,
                .2105003352955]
        self.margeff_nodummy_dydxmedian_se = [.1546708, .0134314, .0928183]
        self.margeff_nodummy_dydxzero = [6.252993785e-06,2.105437138e-07,
                5.263030788e-06]
        self.margeff_nodummy_dydxzero_se = [.0000288,  9.24e-07, .000025]
        self.margeff_nodummy_dyex = [1.1774000792198,.27896245178384,
                .16960002159996]
        self.margeff_nodummy_dyex_se = [.3616481, .4090679, .0635583]
        self.margeff_nodummy_dyexmean = [1.6641381583512,.39433730945339,
                .19658592659731]
        self.margeff_nodummy_dyexmean_se = [.7388917, .5755722, .0860836]
        #NOTE: PSI at median should be a NaN or 'omitted'
        self.margeff_nodummy_dyexmedian = [.76654095836557,.18947053379898,0]
        self.margeff_nodummy_dyexmedian_se = [ .4740659, .302207, 0]
        #NOTE: all should be NaN
        self.margeff_nodummy_dyexzero = [0,0,0]
        self.margeff_nodummy_dyexzero_se = [0,0,0]

        self.margeff_nodummy_eydx = [1.8546366266779,.06244722072812,
                1.5610138123033]
        self.margeff_nodummy_eydx_se = [.847903, .0930901, .7146715]
        self.margeff_nodummy_eydxmean = [2.1116143062702,.0710998816585,
                1.7773072368626]
        self.margeff_nodummy_eydxmean_se = [ 1.076109, .1081501, .9120842]
        self.margeff_nodummy_eydxmedian = [2.5488082240624,.0858205793373,
                2.1452853812126]
        self.margeff_nodummy_eydxmedian_se = [1.255377, .1283771, 1.106872]
        self.margeff_nodummy_eydxzero = [2.8261067189993,.0951574597115,
                2.3786824653103]
        self.margeff_nodummy_eydxzero_se = [1.262961, .1415544, 1.064574]
        self.margeff_nodummy_eyex = [5.4747106798973,1.3173389907576,
                .44600395466634]
        self.margeff_nodummy_eyex_se = [2.44682, 1.943525, .1567618]
        self.margeff_nodummy_eyexmean = [6.5822977203268,1.5597536538833,
                .77757191612739]
        self.margeff_nodummy_eyexmean_se = [3.354433, 2.372543, .3990368]
        self.margeff_nodummy_eyexmedian = [7.8120973525952,1.9309630350892,0]
        self.margeff_nodummy_eyexmedian_se = [3.847731951, 2.888485089, 0]

        self.margeff_nodummy_eyexzero = [0,0,0]
        self.margeff_nodummy_eyexzero_se = [0,0,0]

        # for below GPA = 2.0, psi = 1
        self.margeff_nodummy_atexog1 = [.1456333017086,.00490359933927,
                .12257689308426]
        self.margeff_nodummy_atexog1_se = [.145633, .0111226, .1777101]
        # for below GPA at mean, tuce = 21, psi = 0
        self.margeff_nodummy_atexog2 = [.25105129214546,.00845311433473,
                .2113052923675]
        self.margeff_nodummy_atexog2_se = [.1735778, .012017, .0971515]

        # must get this from older margeff or i.psi then margins
        self.margeff_dummy_dydx = [.36258084688424,.01220841099085,
                .35751515254729]
        self.margeff_dummy_dydx_se = [.1094412, .0177942, .1420034]
        self.margeff_dummy_dydxmean = [.53385885781692,.01797548988961,
                .4564984096959]
        self.margeff_dummy_dydxmean_se = [.237038, .0262369, .1810537]
        #self.margeff_dummy_dydxmedian
        # from margeff
        self.margeff_dummy_count_dydx_median = [0.250110487483923,
                                0.008426867847905,  0.441897738279663]
        self.margeff_dummy_count_dydx_median_se = [.1546736661, .0134551951,
                                                   .1792363708]

        # estimate with i.psi for the below then use margins
        self.margeff_dummy_eydx = [1.8546366266779,.06244722072812,
                1.5549034398832]
        self.margeff_dummy_eydx_se = [.847903, .0930901, .7283702]
        # ie
        #  margins, eydx(*) at((mean) _all)
        self.margeff_dummy_eydxmean = [2.1116143062702,.0710998816585,
                1.6631775707188]
        self.margeff_dummy_eydxmean_se = [1.076109, .1081501, .801205]

        # Factor variables not allowed in below
        # test raises
        #self.margeff_dummy_dydxzero
        #self.margeff_dummy_eydxmedian
        #self.margeff_dummy_eydxzero
        #self.margeff_dummy_dyex
        #self.margeff_dummy_dyexmean
        #self.margeff_dummy_dyexmedian
        #self.margeff_dummy_dyexzero
        #self.margeff_dummy_eyex
        #self.margeff_count_dummy_dydx_median
        #self.margeff_count_dummy_dydx_median_se

        #NOTE: need old version of margeff for nodisc but at option is broken
        # stata command is margeff, count nodisc
        # this can be replicated with the new results by margeff
        # and then using margins for the last value
        self.margeff_count_dydx = [.3625767598018,  .0122068569914, .3051777]
        self.margeff_count_dydx_se = [.1094379569, .0177869773, .0923796]

        # middle value taken from margeff rest from margins
        self.margeff_count_dydxmean = [.5338588,  0.01797186545386,
                                .4493393 ]
        self.margeff_count_dydxmean_se = [.237038, .0262211, .1967626]

        # with new version of margeff this is just a call to
        # margeff
        # mat list e(margeff_b), nonames format(%17.16g)
        self.margeff_count_dummy_dydxoverall = [.362576759801767,
                                        .012206856991439,  .357515163621704]
        # AFAICT, an easy way to get se is
        # mata
        # V = st_matrix("e(margeff_V)")
        # se = diagonal(cholesky(diag(V)))
        # last SE taken from margins with i.psi, don't know how they
        # don't know why margeff is different, but trust official results
        self.margeff_count_dummy_dydxoverall_se = [.1094379569,   .0177869773,
                                                    .1420034]
                #.1574340751 ]

        # from new margeff
        self.margeff_count_dummy_dydxmean = [0.533849340033768,
                        0.017971865453858,  0.456498405282412]
        self.margeff_count_dummy_dydxmean_se = [.2370202503, .0262210796,
                                            .1810536852 ]

        # for below GPA = 2.0, psi = 1
        self.margeff_dummy_atexog1 = [.1456333017086,.00490359933927,
                .0494715429937]
        self.margeff_dummy_atexog1_se = [.145633, .0111226, .0731368]
        # for below GPA at mean, tuce = 21, psi = 0
        self.margeff_dummy_atexog2 = [.25105129214546,.00845311433473,
                .44265645632553]
        self.margeff_dummy_atexog2_se = [.1735778, .012017, .1811925]
        #The test for the prediction table was taken from Gretl
        #Gretl Output matched the Stata output here for params and SE
        self.pred_table = np.array([[18, 3], [3, 8]])

    def probit(self):
        self.params = [1.62581025407, .051728948442, 1.42633236818,
                -7.45232041607]
        self.cov_params =    [[.481472955383, -.01891350017, .105439226234,
            -1.1696681354], [-.01891350017, .00703757594, .002471864882,
            -.101172838897], [.105439226234, .002471864882, .354070126802,
            -.594791776765], [-1.1696681354, -.101172838897, -.594791776765,
            6.46416639958]]
        self.bse = [.693882522754, .083890261293, .595037920474, 2.54247249731]
        self.llf = -12.8188033249334
        self.llnull = -20.5917296966173
        self.df_model = 3
        self.df_resid = 32 - 4
        self.llr = 15.5458527433678
        self.prsquared = .377478069409622
        self.llr_pvalue = .00140489496775855
        self.aic = 33.637606649867
        self.bic = 39.500550261066
        self.z = [ 2.343062695, .6166263836, 2.397044489, -2.931131182]
        self.conf_int = [[.2658255,2.985795],[-.1126929,.2161508],[.2600795,
            2.592585],[-12.43547,-2.469166]]
        self.pvalues = [.0191261688, .537481188, .0165279168, .0033773013]
        self.phat = [.0181707, .0530805, .1899263, .0185707, .5545748,
                        .0272331, .0185033, .0445714, .1088081, .6631207,
                        .0161024, .1935566, .3233282, .1951826, .3563406,
                        .0219654, .0456943, .0308513, .5934023, .6571863,
                        .0619288, .9045388, .2731908, .8474501, .8341947,
                        .488726, .6424073, .3286732, .8400168, .9522446,
                        .5399595, .123544]
        self.yhat = yhat = np.array([-2.0930860042572,
                  -1.615691781044,
                 -.87816804647446,
                 -2.0842070579529,
                  .13722851872444,
                 -1.9231110811234,
                 -2.0856919288635,
                 -1.6999372243881,
                 -1.2328916788101,
                  .42099541425705,
                 -2.1418602466583,
                 -.86486464738846,
                 -.45841211080551,
                 -.85895526409149,
                 -.36825761198997,
                 -2.0147502422333,
                 -1.6881184577942,
                 -1.8684275150299,
                  .23630557954311,
                  .40479621291161,
                  -1.538782119751,
                  1.3078554868698,
                 -.60319095849991,
                   1.025558590889,
                  .97087496519089,
                 -.02826354466379,
                  .36490100622177,
                 -.44357979297638,
                  .99452745914459,
                  1.6670187711716,
                  .10033150017262,
                 -1.1574513912201])
        self.resid = [-.191509, -.3302762, -.6490455, -.1936247, 1.085867,
                      -.2349926, -.1932698, -.3019776, -.4799906, .9064196,
                      -.1801855, -.6559291, -.8838201, 1.807661, -.9387071,
                      -.2107617, -.3058469, -.2503485, -1.341589, .9162835,
                      -.3575735, .447951, -.7988633, -1.939208, .6021435,
                      1.196623, .9407793, -.8927477, .59048, .3128364,
                      -1.246147, 2.045071]
        self.pred_table = np.array([[18, 3], [3, 8]])


class RandHIE():
    """
    Results obtained from Stata 11
    """
    def __init__(self):
        self.nobs = 20190

    def poisson(self):
        self.params =   [-.052535114675, -.247086797633, .035290201794,
                -.03457750643, .271713973711, .033941474461, -.012635035534,
                .054056326828, .206115121809, .700352877227]
        self.cov_params = None
        self.bse = [.00288398915279, .01061725196728, .00182833684966,
                .00161284852954, .01223913844387, .00056476496963,
                .00925061122826, .01530987068312, .02627928267502,
                .01116266712362]
        predict = np.loadtxt(os.path.join(cur_dir, 'yhat_poisson.csv'),
                   delimiter=",")
        self.phat = predict[:,0]
        self.yhat = predict[:,1]
        self.llf = -62419.588535018
        self.llnull = -66647.181687959
        self.df_model = 9
        self.df_resid = self.nobs - self.df_model - 1
        self.llr = 8455.186305881856
        self.prsquared = .0634324369893758
        self.llr_pvalue = 0
        self.aic = 124859.17707
        self.bic = 124938.306497
        self.z = [-18.21612769, -23.27219872, 19.30180524, -21.43878101,
                22.20041672, 60.09840604, -1.36585953, 3.53081538, 7.84325525,
                62.74063980]
        self.conf_int = [[ -.0581876, -.0468826],[-0.2678962, -0.2262774],
                [0.0317067, 0.0388737],[-0.0377386, -0.0314164],
                [0.2477257, 0.2957022], [0.0328346, 0.0350484],[-0.0307659,
                    0.0054958], [0.0240495, 0.0840631],[0.1546087, 0.2576216],
                [0.6784745, 0.7222313]]
        self.pvalues = [3.84415e-74, 8.4800e-120, 5.18652e-83, 5.8116e-102,
                3.4028e-109, 0, .1719830562, .0004142808, 4.39014e-15, 0]

        # from stata
        # use margins and put i. in front of dummies
        self.margeff_dummy_overall = [-0.15027280560599, -0.66568074771099,
                                 0.10094500919706, -0.09890639687842,
                                 0.77721770295360,  0.09708707452600,
                                -0.03608195237609, 0.15804581481115,
                                0.65104087597053]
        self.margeff_dummy_overall_se = [.008273103,  .0269856266,
                            .0052466639, .0046317555, .0351582169, .0016652181,
                            .0263736472,   .0457480115,  .0913901155]

        # just use margins
        self.margeff_nodummy_overall = [-0.15027280560599, -0.70677348928158,
                                         0.10094500919705, -0.09890639687842,
                                         0.77721770295359, 0.09708707452600,
                                         -0.03614158359367, 0.15462412033340,
                                         0.58957704430148]
        self.margeff_nodummy_overall_se = [.008273103, .0305119343,
                                           .0052466639, .0046317555,
                                           .0351582168, .0016652181,
                                           .0264611158, .0437974779,
                                           .0752099666]

