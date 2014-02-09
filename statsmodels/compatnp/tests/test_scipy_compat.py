from numpy.testing import assert_equal
from statsmodels.compatnp.scipy_compat import _next_regular


def test_next_regular():
    hams = {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 8, 8: 8, 14: 15, 15: 15,
        16: 16, 17: 18, 1021: 1024, 1536: 1536, 51200000: 51200000,
        510183360: 510183360, 510183360+1: 512000000, 511000000: 512000000,
        854296875: 854296875, 854296875+1: 859963392,
        196608000000: 196608000000, 196608000000+1: 196830000000,
        8789062500000: 8789062500000, 8789062500000+1: 8796093022208,
        206391214080000: 206391214080000, 206391214080000+1: 206624260800000,
        470184984576000: 470184984576000, 470184984576000+1: 470715894135000,
        7222041363087360: 7222041363087360,
        7222041363087360+1: 7230196133913600,

        # power of 5    5**23
        11920928955078125: 11920928955078125,
        11920928955078125-1: 11920928955078125,

        # power of 3    3**34
        16677181699666569: 16677181699666569,
        16677181699666569-1: 16677181699666569,

        # power of 2   2**54
        18014398509481984: 18014398509481984,
        18014398509481984-1: 18014398509481984,

        # above this, int(ceil(n)) == int(ceil(n+1))
        19200000000000000: 19200000000000000,
        19200000000000000+1: 19221679687500000,

        288230376151711744:   288230376151711744,
        288230376151711744+1: 288325195312500000,
        288325195312500000-1: 288325195312500000,
        288325195312500000:   288325195312500000,
        288325195312500000+1: 288555831593533440,

        # power of 3    3**83
        3990838394187339929534246675572349035227-1:
        3990838394187339929534246675572349035227,
        3990838394187339929534246675572349035227:
        3990838394187339929534246675572349035227,

        # power of 2     2**135
        43556142965880123323311949751266331066368-1:
        43556142965880123323311949751266331066368,
        43556142965880123323311949751266331066368:
        43556142965880123323311949751266331066368,

        # power of 5      5**57
        6938893903907228377647697925567626953125-1:
        6938893903907228377647697925567626953125,
        6938893903907228377647697925567626953125:
        6938893903907228377647697925567626953125,

        # http://www.drdobbs.com/228700538
        # 2**96 * 3**1 * 5**13
        290142196707511001929482240000000000000-1:
        290142196707511001929482240000000000000,
        290142196707511001929482240000000000000:
        290142196707511001929482240000000000000,
        290142196707511001929482240000000000000+1:
        290237644800000000000000000000000000000,

        # 2**36 * 3**69 * 5**7
        4479571262811807241115438439905203543080960000000-1:
        4479571262811807241115438439905203543080960000000,
        4479571262811807241115438439905203543080960000000:
        4479571262811807241115438439905203543080960000000,
        4479571262811807241115438439905203543080960000000+1:
        4480327901140333639941336854183943340032000000000,

        # 2**37 * 3**44 * 5**42
        30774090693237851027531250000000000000000000000000000000000000-1:
        30774090693237851027531250000000000000000000000000000000000000,
        30774090693237851027531250000000000000000000000000000000000000:
        30774090693237851027531250000000000000000000000000000000000000,
        30774090693237851027531250000000000000000000000000000000000000+1:
        30778180617309082445871527002041377406962596539492679680000000,
    }

    for x, y in hams.items():
        assert_equal(_next_regular(x), y)
