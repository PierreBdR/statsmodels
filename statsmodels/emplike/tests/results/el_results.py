"""
Results from Matlab and R
"""
import numpy as np


class DescStatRes(object):
    """

    The results were generated from Bruce Hansen's
    MATLAb package:

    Bruce E. Hansen
    Department of Economics
    Social Science Building
    University of Wisconsin
    Madison, WI 53706-1393
    bhansen@ssc.wisc.edu
    http://www.ssc.wisc.edu/~bhansen/

    The R results are from Mai Zhou's emplik package

    """

    def __init__(self):
        self.ci_mean = (13.556709, 14.559394)
        self.test_mean_14 = (.080675, .776385)
        self.test_mean_weights = np.array([[0.01969213],
                                             [0.01911859],
                                             [0.01973982],
                                             [0.01982913],
                                             [0.02004183],
                                             [0.0195765],
                                             [0.01970423],
                                             [0.02015074],
                                             [0.01898431],
                                             [0.02067787],
                                             [0.01878104],
                                             [0.01920531],
                                             [0.01981207],
                                             [0.02031582],
                                             [0.01857329],
                                             [0.01907883],
                                             [0.01943674],
                                             [0.0210042],
                                             [0.0197373],
                                             [0.01997998],
                                             [0.0199233],
                                             [0.01986713],
                                             [0.02017751],
                                             [0.01962176],
                                             [0.0214596],
                                             [0.02118228],
                                             [0.02013767],
                                             [0.01918665],
                                             [0.01908886],
                                             [0.01943081],
                                             [0.01916251],
                                             [0.01868129],
                                             [0.01918334],
                                             [0.01969084],
                                             [0.01984322],
                                             [0.0198977],
                                             [0.02098504],
                                             [0.02132222],
                                             [0.02100581],
                                             [0.01970351],
                                             [0.01942184],
                                             [0.01979781],
                                             [0.02114276],
                                             [0.02096136],
                                             [0.01999804],
                                             [0.02044712],
                                             [0.02174404],
                                             [0.02189933],
                                             [0.02077078],
                                             [0.02082612]]).squeeze()
        self.test_var_3 = (.199385, .655218)
        self.ci_var = (2.290077, 4.423634)
        self.test_var_weights = np.array([[0.020965],
                                             [0.019686],
                                             [0.021011],
                                             [0.021073],
                                             [0.021089],
                                             [0.020813],
                                             [0.020977],
                                             [0.021028],
                                             [0.019213],
                                             [0.02017],
                                             [0.018397],
                                             [0.01996],
                                             [0.021064],
                                             [0.020854],
                                             [0.017463],
                                             [0.019552],
                                             [0.020555],
                                             [0.019283],
                                             [0.021009],
                                             [0.021103],
                                             [0.021102],
                                             [0.021089],
                                             [0.021007],
                                             [0.020879],
                                             [0.017796],
                                             [0.018726],
                                             [0.021038],
                                             [0.019903],
                                             [0.019587],
                                             [0.020543],
                                             [0.019828],
                                             [0.017959],
                                             [0.019893],
                                             [0.020963],
                                             [0.02108],
                                             [0.021098],
                                             [0.01934],
                                             [0.018264],
                                             [0.019278],
                                             [0.020977],
                                             [0.020523],
                                             [0.021055],
                                             [0.018853],
                                             [0.019411],
                                             [0.0211],
                                             [0.02065],
                                             [0.016803],
                                             [0.016259],
                                             [0.019939],
                                             [0.019793]]).squeeze()
        self.mv_test_mean = (.7002663, .7045943)
        self.mv_test_mean_wts = np.array([[0.01877015],
                                             [0.01895746],
                                             [0.01817092],
                                             [0.01904308],
                                             [0.01707106],
                                             [0.01602806],
                                             [0.0194296],
                                             [0.01692204],
                                             [0.01978322],
                                             [0.01881313],
                                             [0.02011785],
                                             [0.0226274],
                                             [0.01953733],
                                             [0.01874346],
                                             [0.01694224],
                                             [0.01611816],
                                             [0.02297437],
                                             [0.01943187],
                                             [0.01899873],
                                             [0.02113375],
                                             [0.02295293],
                                             [0.02043509],
                                             [0.02276583],
                                             [0.02208274],
                                             [0.02466621],
                                             [0.02287983],
                                             [0.0173136],
                                             [0.01905693],
                                             [0.01909335],
                                             [0.01982534],
                                             [0.01924093],
                                             [0.0179352],
                                             [0.01871907],
                                             [0.01916633],
                                             [0.02022359],
                                             [0.02228696],
                                             [0.02080555],
                                             [0.01725214],
                                             [0.02166185],
                                             [0.01798537],
                                             [0.02103582],
                                             [0.02052757],
                                             [0.03096074],
                                             [0.01966538],
                                             [0.02201629],
                                             [0.02094854],
                                             [0.02127771],
                                             [0.01961964],
                                             [0.02023756],
                                             [0.01774807]]).squeeze()
        self.test_skew = (2.498418, .113961)
        self.test_skew_wts = np.array([[0.016698],
                                        [0.01564],
                                        [0.01701],
                                        [0.017675],
                                        [0.019673],
                                        [0.016071],
                                        [0.016774],
                                        [0.020902],
                                        [0.016397],
                                        [0.027359],
                                        [0.019136],
                                        [0.015419],
                                        [0.01754],
                                        [0.022965],
                                        [0.027203],
                                        [0.015805],
                                        [0.015565],
                                        [0.028518],
                                        [0.016992],
                                        [0.019034],
                                        [0.018489],
                                        [0.01799],
                                        [0.021222],
                                        [0.016294],
                                        [0.022725],
                                        [0.027133],
                                        [0.020748],
                                        [0.015452],
                                        [0.015759],
                                        [0.01555],
                                        [0.015506],
                                        [0.021863],
                                        [0.015459],
                                        [0.01669],
                                        [0.017789],
                                        [0.018257],
                                        [0.028578],
                                        [0.025151],
                                        [0.028512],
                                        [0.01677],
                                        [0.015529],
                                        [0.01743],
                                        [0.027563],
                                        [0.028629],
                                        [0.019216],
                                        [0.024677],
                                        [0.017376],
                                        [0.014739],
                                        [0.028112],
                                        [0.02842]]).squeeze()
        self.test_kurt_0 = (1.904269, .167601)
        self.test_corr = (.012025, .912680,)
        self.test_corr_weights = np.array([[0.020037],
                                              [0.020108],
                                              [0.020024],
                                              [0.02001],
                                              [0.019766],
                                              [0.019971],
                                              [0.020013],
                                              [0.019663],
                                              [0.019944],
                                              [0.01982],
                                              [0.01983],
                                              [0.019436],
                                              [0.020005],
                                              [0.019897],
                                              [0.020768],
                                              [0.020468],
                                              [0.019521],
                                              [0.019891],
                                              [0.020024],
                                              [0.01997],
                                              [0.019824],
                                              [0.019976],
                                              [0.019979],
                                              [0.019753],
                                              [0.020814],
                                              [0.020474],
                                              [0.019751],
                                              [0.020085],
                                              [0.020087],
                                              [0.019977],
                                              [0.020057],
                                              [0.020435],
                                              [0.020137],
                                              [0.020025],
                                              [0.019982],
                                              [0.019866],
                                              [0.020151],
                                              [0.019046],
                                              [0.020272],
                                              [0.020034],
                                              [0.019813],
                                              [0.01996],
                                              [0.020657],
                                              [0.019947],
                                              [0.019931],
                                              [0.02008],
                                              [0.02035],
                                              [0.019823],
                                              [0.02005],
                                              [0.019497]]).squeeze()
        self.test_joint_skew_kurt = (8.753952, .012563)



class RegressionResults(object):
    """
    Results for EL Regression
    """
    def __init__(self):
        self.source = 'Matlab'
        self.hy_test_beta0 = (.184861, 1.758104,
                               np.mat([[ 0.04326392,
        0.04736749,  0.03573865,  0.04770535,  0.04721684,
        0.04718301,  0.07088816,  0.05631242,  0.04865098,  0.06572099,
        0.04016013,  0.04438627,  0.04042288,  0.03938043,  0.04006474,
        0.04845233,  0.01991985,  0.03623254,  0.03617999,  0.05607242,
        0.0886806 ]]))

        self.hy_test_beta1 = (.164482, 1.932529, np.mat([[ 0.033328,
        0.051412,  0.03395 ,  0.071695,  0.046433,  0.041303,
        0.033329,  0.036413,  0.03246 ,  0.037776,  0.043872,  0.037507,
        0.04762 ,  0.04881 ,  0.05874 ,  0.049553,  0.048898,  0.04512 ,
        0.041142,  0.048121,  0.11252 ]]))

        self.hy_test_beta2 = (.481866, .494593, np.mat([[ 0.046287,
        0.048632,  0.048772,  0.034769,  0.048416,  0.052447,
        0.053336,  0.050112,  0.056053,  0.049318,  0.053609,  0.055634,
        0.042188,  0.046519,  0.048415,  0.047897,  0.048673,  0.047695,
        0.047503,  0.047447,  0.026279]]))

        self.hy_test_beta3 = (.060005, 3.537250, np.mat([[ 0.036327,
        0.070483,  0.048965,  0.087399,  0.041685,  0.036221,
        0.016752,  0.019585,  0.027467,  0.02957 ,  0.069204,  0.060167,
        0.060189,  0.030007,  0.067371,  0.046862,  0.069814,  0.053041,
        0.053362,  0.041585,  0.033943]]))
