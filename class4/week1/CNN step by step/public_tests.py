import numpy as np
from test_utils import single_test, multiple_test


def zero_pad_test(target):
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    pad = 2
    expected_output = expected_output = np.array([[[[0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.]],

                                                   [[0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.]],

                                                   [[0., 0.],
                                                    [0., 0.],
                                                    [1.62434536, -0.61175641],
                                                    [-0.52817175, -1.07296862],
                                                    [0.86540763, -2.3015387],
                                                    [0., 0.],
                                                    [0., 0.]],

                                                   [[0., 0.],
                                                    [0., 0.],
                                                    [1.74481176, -0.7612069],
                                                    [0.3190391, -0.24937038],
                                                    [1.46210794, -2.06014071],
                                                    [0., 0.],
                                                    [0., 0.]],

                                                   [[0., 0.],
                                                    [0., 0.],
                                                    [-0.3224172, -0.38405435],
                                                    [1.13376944, -1.09989127],
                                                    [-0.17242821, -0.87785842],
                                                    [0., 0.],
                                                    [0., 0.]],

                                                   [[0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.]],

                                                   [[0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.],
                                                    [0., 0.]]],


                                                  [[[0., 0.],
                                                    [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [0.04221375, 0.58281521],
                                                      [-1.10061918, 1.14472371],
                                                      [0.90159072, 0.50249434],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [0.90085595, -0.68372786],
                                                      [-0.12289023, -0.93576943],
                                                      [-0.26788808, 0.53035547],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [-0.69166075, -0.39675353],
                                                      [-0.6871727, -0.84520564],
                                                      [-0.67124613, -0.0126646],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.]]],


                                                  [[[0., 0.],
                                                    [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [-1.11731035, 0.2344157],
                                                      [1.65980218, 0.74204416],
                                                      [-0.19183555, -0.88762896],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [-0.74715829, 1.6924546],
                                                      [0.05080775, -0.63699565],
                                                      [0.19091548, 2.10025514],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [0.12015895, 0.61720311],
                                                      [0.30017032, -0.35224985],
                                                      [-1.1425182, -0.34934272],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.]]],


                                                  [[[0., 0.],
                                                    [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [-0.20889423, 0.58662319],
                                                      [0.83898341, 0.93110208],
                                                      [0.28558733, 0.88514116],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [-0.75439794, 1.25286816],
                                                      [0.51292982, -0.29809284],
                                                      [0.48851815, -0.07557171],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [1.13162939, 1.51981682],
                                                      [2.18557541, -1.39649634],
                                                      [-1.44411381, -0.50446586],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.]],

                                                   [[0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.],
                                                      [0., 0.]]]])

    test_cases = [
        {
            "name": "datatype_check",
            "input": [x, pad],
            "expected": expected_output,
            "error":"Datatype mismatch."
        },
        {
            "name": "equation_output_check",
            "input": [x, pad],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    single_test(test_cases, target)


def conv_single_step_test(target):

    np.random.seed(1)
    a_slice_prev = np.random.randn(4, 4, 3)
    W = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)
    expected_output = np.float64(-6.999089450680221)
    test_cases = [
        {
            "name": "datatype_check",
            "input": [a_slice_prev, W, b],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [a_slice_prev, W, b],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [a_slice_prev, W, b],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)


def conv_forward_test(target):
    A_prev = np.random.randn(2, 5, 7, 4)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad" : 1,
                   "stride": 2}
    Z, cache_conv = target(A_prev, W, b, {"pad" : 3, "stride": 1})
    Z_shape = Z.shape
    assert Z_shape[0] == A_prev.shape[0], f"m is wrong. Current: {Z_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert Z_shape[1] == 9, f"n_H is wrong. Current: {Z_shape[1]}.  Expected: 9"
    assert Z_shape[2] == 11, f"n_W is wrong. Current: {Z_shape[2]}.  Expected: 11"
    assert Z_shape[3] == W.shape[3], f"n_C is wrong. Current: {Z_shape[3]}.  Expected: {W.shape[3]}"

    Z, cache_conv = target(A_prev, W, b, {"pad" : 0, "stride": 2})
    assert(Z.shape == (2, 2, 3, 8)), "Wrong shape. Don't hard code the pad and stride values in the function"

    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 7, 4)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 1,
                   "stride": 2}
    expected_Z = np.array([[[[-2.65112363, -0.37849177, -1.97054929, -1.96235299,
                              -1.72259872, 0.4676693, -6.43434016, 1.10764994],
                             [4.67692928, 4.29865415, -1.3608031, 0.80532859,
                              -2.88480108, 8.95280034, 5.32627807, -1.82635258],
                             [-2.05881174, 3.40859795, 0.3502282, 0.68303626,
                              -1.88328065, -1.87480174, 5.8008721, 0.0700918],
                             [-3.50141791, 2.704286, 0.28341346, 4.15637411,
                              -0.46575834, -0.43668824, -5.56866106, 1.72288033]],

                            [[-2.32126108, 0.91040602, 2.31852532, 0.98842271,
                              3.31716611, 4.05638832, -2.48135123, 0.95872443],
                             [6.03978907, -6.96477888, -1.20799344, 2.68913374,
                                -4.35744033, 10.59355329, 3.20856901, 13.98735978],
                             [-3.01280755, -2.90226517, -8.34171936, -5.26220853,
                              5.6630696, 1.08704033, 2.20430705, -10.73218294],
                             [-6.24198266, -0.53158832, -3.29654954, -1.81865997,
                              0.59196322, 2.51134745, -4.24924673, 5.21936641]],

                            [[-2.22187412, -0.95259173, -5.99441273, 0.79147932,
                              1.16919278, -0.17321161, -3.26346299, -3.62407578],
                             [-2.17796037, 8.07171329, -0.5772704, 3.36286738,
                              4.48113645, -2.89198428, 10.99288867, 3.03171932],
                             [-12.49991261, 5.26845833, -1.67648614, -8.65695762,
                                -10.68157258, 6.71492428, 2.83839971, 4.47259772],
                             [0.11421092, -1.90872424, -3.28117601, 0.89922467,
                              0.83985348, -0.25127044, -0.94409718, 5.17244412]]],


                           [[[1.97649814, 2.76743075, -6.39611007, 2.95378171,
                              -0.81235239, -0.53333631, 0.71268871, 4.91385105],
                             [-5.14401869, 6.97041391, -4.53976469, 5.89092653,
                               -5.74606931, 2.74256558, 3.02124802, -10.04187592],
                               [5.53871187, -8.55886701, -4.70962135, 2.55966738,
                                -2.66959504, 5.60010695, -8.37253342, 4.18848278],
                               [0.63364517, -3.71848223, -3.67072772, 4.34226476,
                                -1.21894465, 3.68929452, 5.89166305, 0.94256457]],

                            [[2.36049402, -3.09696204, 8.33521755, 3.04680748,
                              3.7964542, 0.66488788, 1.9935476, 1.54396221],
                               [-7.73457048, 0.287562, 7.97481218, 3.32415996,
                                -4.07121488, 2.69182963, 4.1356109, -5.16178423],
                               [-6.95635186, -0.10924121, -4.12526441, 0.62578199,
                                4.69492086, -3.52748877, 3.63168271, 0.64007629],
                               [7.94980014, 5.71855659, 3.49970333, 12.7718152,
                                8.84959478, 2.37150319, -1.42531648, -0.51126641]],

                            [[-5.29658283, -4.20466999, -6.63067766, -9.87831724,
                              -5.32130395, 7.32417919, 2.96011091, 7.60669481],
                               [11.54630784, -1.93157244, 2.26699242, 7.62184275,
                                5.40584348, -2.88837958, -1.46981877, 7.91314719],
                               [5.94067877, 3.50739649, 0.82512202, 4.80655489,
                                -4.1044945, 4.14358541, 0.13194885, 4.35397285],
                               [4.91298364, -1.44499772, 5.9392078, -3.92690408,
                                2.12840309, 1.27237402, 1.56992581, 0.44270565]]]])
    expected_cache = (A_prev, W, b, hparameters)
    expected_output = (expected_Z, expected_cache)
    test_cases = [
        {
            "name": "datatype_check",
            "input": [A_prev, W, b, hparameters],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [A_prev, W, b, hparameters],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, W, b, hparameters],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)


def pool_forward_test(target):
    
    A_prev = np.random.randn(2, 5, 7, 3)
    A, cache = target(A_prev, {"stride" : 2, "f": 2}, mode = "average")
    A_shape = A.shape
    assert A_shape[0] == A_prev.shape[0], f"m is wrong. Current: {A_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert A_shape[1] == 2, f"n_H is wrong. Current: {A_shape[1]}.  Expected: 2"
    assert A_shape[2] == 3, f"n_W is wrong. Current: {A_shape[2]}.  Expected: 3"
    assert A_shape[3] == A_prev.shape[3], f"n_C is wrong. Current: {A_shape[3]}.  Expected: {A_prev.shape[3]}"
    
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 5, 3)
    hparameters = {"stride": 1, "f": 3}
    expected_cache = (A_prev, hparameters)

    expected_A_max = np.array([[[[1.74481176, 0.90159072, 1.65980218],
                                 [1.74481176, 1.46210794, 1.65980218],
                                 [1.74481176, 1.6924546, 1.65980218]],

                                [[1.14472371, 0.90159072, 2.10025514],
                                 [1.14472371, 0.90159072, 1.65980218],
                                 [1.14472371, 1.6924546, 1.65980218]],

                                [[1.13162939, 1.51981682, 2.18557541],
                                 [1.13162939, 1.51981682, 2.18557541],
                                 [1.13162939, 1.6924546, 2.18557541]]],


                               [[[1.19891788, 0.84616065, 0.82797464],
                                 [0.69803203, 0.84616065, 1.2245077],
                                   [0.69803203, 1.12141771, 1.2245077]],

                                [[1.96710175, 0.84616065, 1.27375593],
                                   [1.96710175, 0.84616065, 1.23616403],
                                   [1.62765075, 1.12141771, 1.2245077]],

                                [[1.96710175, 0.86888616, 1.27375593],
                                   [1.96710175, 0.86888616, 1.23616403],
                                   [1.62765075, 1.12141771, 0.79280687]]]])

    expected_output_max = (expected_A_max, expected_cache)

    expected_A_average = np.array([[[[-3.01046719e-02, -3.24021315e-03, -3.36298859e-01],
                                     [1.43310483e-01, 1.93146751e-01, -
                                      4.44905196e-01],
                                     [1.28934436e-01, 2.22428468e-01, 1.25067597e-01]],

                                    [[-3.81801899e-01, 1.59993515e-02, 1.70562706e-01],
                                     [4.73707165e-02, 2.59244658e-02,
                                        9.20338402e-02],
                                     [3.97048605e-02, 1.57189094e-01, 3.45302489e-01]],

                                    [[-3.82680519e-01, 2.32579951e-01, 6.25997903e-01],
                                     [-2.47157416e-01, -3.48524998e-04,
                                      3.50539717e-01],
                                     [-9.52551510e-02, 2.68511000e-01, 4.66056368e-01]]],


                                   [[[-1.73134159e-01, 3.23771981e-01, -3.43175716e-01],
                                     [3.80634669e-02, 7.26706274e-02, -
                                      2.30268958e-01],
                                       [2.03009393e-02, 1.41414785e-01, -1.23158476e-02]],

                                    [[4.44976963e-01, -2.61694592e-03, -3.10403073e-01],
                                       [5.08114737e-01, -
                                        2.34937338e-01, -2.39611830e-01],
                                       [1.18726772e-01, 1.72552294e-01, -2.21121966e-01]],

                                    [[4.29449255e-01, 8.44699612e-02, -2.72909051e-01],
                                       [6.76351685e-01, -
                                        1.20138225e-01, -2.44076712e-01],
                                       [1.50774518e-01, 2.89111751e-01, 1.23238536e-03]]]])
    expected_output_average = (expected_A_average, expected_cache)
    test_cases = [
        {
            "name": "datatype_check",
            "input": [A_prev, hparameters, "max"],
            "expected": expected_output_max,
            "error":"Datatype mismatch in MAX-Pool"
        },
        {
            "name": "shape_check",
            "input": [A_prev, hparameters, "max"],
            "expected": expected_output_max,
            "error": "Wrong shape in MAX-Pool"
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, hparameters, "max"],
            "expected": expected_output_max,
            "error": "Wrong output in MAX-Pool"
        },
        {
            "name": "datatype_check",
            "input": [A_prev, hparameters, "average"],
            "expected": expected_output_average,
            "error":"Datatype mismatch in AVG-Pool"
        },
        {
            "name": "shape_check",
            "input": [A_prev, hparameters, "average"],
            "expected": expected_output_average,
            "error": "Wrong shape in AVG-Pool"
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, hparameters, "average"],
            "expected": expected_output_average,
            "error": "Wrong output in AVG-Pool"
        }
    ]

    multiple_test(test_cases, target)

######################################
############## UNGRADED ##############
######################################


def conv_backward_test(target):

    test_cases = [
        {
            "name": "datatype_check",
            "input": [parameters, cache, X, Y],
            "expected": expected_output,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [parameters, cache, X, Y],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, cache, X, Y],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)


def create_mask_from_window_test(target):

    test_cases = [
        {
            "name": "datatype_check",
            "input": [parameters, grads],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, grads],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, grads],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)


def distribute_value_test(target):
    test_cases = [
        {
            "name": "datatype_check",
            "input": [X, Y, n_h],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [X, Y, n_h],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [X, Y, n_h],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)


def pool_backward_test(target):

    test_cases = [
        {
            "name": "datatype_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    single_test(test_cases, target)
