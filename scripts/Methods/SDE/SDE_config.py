import numpy as np

MODEL_OPTION = "2state_toy"
MODEL_TYPE = "WINDING"#  "WINDING"/"CYLINDER"
RUN_PL1 = True
RUN_PL2 = True
RUN_TOY_CHECK = True
RUN_MCMC = True
RUN_WILKS = False

if RUN_TOY_CHECK:
    CSV_TRAIN = "toy_generator_2state_train.csv"
    CSV_TEST = "toy_generator_2state_test.csv"
else:
    CSV_TRAIN = "ds1_train.csv"
    CSV_TEST = "ds1_test.csv"

TIME_COL = "Created"
maxiter  = 2000
NIS_THRESHOLD_PERCENTILE = 0.995

match MODEL_OPTION:
    case "1state":
        if MODEL_TYPE == "WINDING":
            if RUN_TOY_CHECK:
                SENSOR_COLS = [
                    "AE PORT GEN.W-WINDING TEMP.",
                    "AE PORT LUB.OIL TEMP.",
                    "POWER_kW",
                ]
                RENAME_MAP = {
                    "AE PORT GEN.W-WINDING TEMP.": "Tw",
                    "AE PORT LUB.OIL TEMP.": "Toil",
                    "POWER_kW": "P",
                }

                STATE_COLS = ["Tw"]
                MEAS_COLS  = ["Tw"]
                INPUT_COLS = ["P", "Toil"]

                PARAMETER_NAMES = ["Cw", "Rg"]

                # Process (model) noise covariance
                Q = np.diag([
                    0.0001,
                ])
                # Measurement noise covariance
                R = np.diag([
                    0.2 ** 2,  # TW
                ])
                C = np.array([
                    [1.0],  # TW

                ])
                # ------------------------------------------------------------
                # Parameter estimation settings
                # ------------------------------------------------------------
                THETA0 = np.array([1e5, 0.1])
                LOWER_BOUND = np.array([1e4, 0.05])
                UPPER_BOUND = np.array([1e7, 1e3])
            else:
                SENSOR_COLS = [
                    "AE PORT GEN.W-WINDING TEMP.",
                    #"AE PORT LUB.OIL TEMP.",
                    "FAN 3 TEMPERATURE",
                    "POWER_kW",
                ]
                RENAME_MAP = {
                    "AE PORT GEN.W-WINDING TEMP.": "T1",
                    #"AE PORT LUB.OIL TEMP.": "Tref",
                    "FAN 3 TEMPERATURE": "Tref",
                    "POWER_kW": "P",
                }

                STATE_COLS = ["T1"]
                MEAS_COLS  = ["T1"]
                INPUT_COLS = ["P", "Tref"]
                PARAMETER_NAMES = ["C1", "R1"]

                W = np.diag([0.01])  # Process-noise covariance W/Q
                V = np.diag([0.04 ** 2])  # 0.01,  ~0.1 °C noise # Measurement noise covariance V/R
                C = np.array([[1.0]])
                # ------------------------------------------------------------
                # Parameter estimation settings
                # ------------------------------------------------------------
                #THETA0 = np.array([1e5, 0.018])
                THETA0 = np.array([1.1e5, 0.05])
                LOWER_BOUND = np.array([1e4, 0.005])
                #UPPER_BOUND = np.array([1e7, 0.05])
                UPPER_BOUND = np.array([1e7, 0.5])


        if MODEL_TYPE == "CYLINDER":
            SENSOR_COLS = [
                "AE PORT CYL.1 EXH.GAS TEMP.",  # the cylinder you're modeling
                "AE PORT HT FW OUTLET TEMP.",  # reference
                "POWER_kW",
            ]
            RENAME_MAP = {
                "AE PORT CYL.1 EXH.GAS TEMP.": "T",
                "AE PORT HT FW OUTLET TEMP.": "Tref",
                "POWER_kW": "P",
            }
            STATE_COLS = ["T"]
            MEAS_COLS = ["T"]
            INPUT_COLS = ["P", "Tref"]
            PARAMETER_NAMES = ["Ccyl", "Rcyl"]
            Q = np.diag([0.01])
            R = np.diag([1.0 ** 2])
            C = np.array([[1.0]])

            THETA0 = np.array([500.0, 0.44])  # Ccyl small (fast exhaust), Rcyl ≈ 0.44
            LOWER_BOUND = np.array([10.0, 0.15])
            UPPER_BOUND = np.array([1e4, 1.3])
    case "1state_2ref":
        SENSOR_COLS = [
            "AE PORT GEN.W-WINDING TEMP.",
            "FAN 3 TEMPERATURE",
            "LT FW TEMP.",
            "POWER_kW",
        ]
        RENAME_MAP = {
            "AE PORT GEN.W-WINDING TEMP.": "T1",
            "FAN 3 TEMPERATURE": "Ta",
            "LT FW TEMP.": "Tc",
            "POWER_kW": "P",
        }

        STATE_COLS = ["T1"]
        MEAS_COLS = ["T1"]
        INPUT_COLS = ["P", "Ta", "Tc"]

        PARAMETER_NAMES = ["C1", "Ra", "Rc"]

        Q = np.diag([0.01])
        R = np.diag([0.04 ** 2])
        C = np.array([[1.0]])

        # Ra, Rc in parallel must give R_par ~ 0.05 (the 1-ref estimate),
        # so each individual R is roughly 2x that.
        THETA0 = np.array([1.1e5, 0.10, 0.10])
        LOWER_BOUND = np.array([1.0e4, 0.005, 0.005])
        UPPER_BOUND = np.array([1.0e7, 5.00, 5.00])
    case "2state_gain":
        SENSOR_COLS = [
            "AE PORT GEN.W-WINDING TEMP.",
            "AE PORT LUB.OIL TEMP.",
            "FAN 3 TEMPERATURE",
            "POWER_kW",
        ]
        RENAME_MAP = {
            "AE PORT GEN.W-WINDING TEMP.": "T1",
            "AE PORT LUB.OIL TEMP.": "T2",
            "FAN 3 TEMPERATURE": "Tref",
            "POWER_kW": "P",
        }
        STATE_COLS = ["T1", "T2"]
        MEAS_COLS = ["T1", "T2"]
        INPUT_COLS = ["P", "Tref"]

        PARAMETER_NAMES = ["C1", "C2", "R1", "R2", "P0"]

        Q = np.diag([0.01, 0.01])
        R = np.diag([0.04 ** 2, 0.04 ** 2])
        C = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        THETA0 = np.array([1e5, 5e5, 0.010, 0.010, 50.0])
        LOWER_BOUND = np.array([1e4, 1e4, 0.002, 0.002, 0.0])
        UPPER_BOUND = np.array([1e7, 1e7, 0.500, 0.500, 1000.0])
    case "2state_toy":

        if RUN_TOY_CHECK:
            SENSOR_COLS = [
                "Tw",
                "Tb",
                "P",
                "Ta",
                "Tc"
            ]
            RENAME_MAP = {
                "Tw": "Tw",
                "Tb": "Tb",
                "P": "P",
                "Ta": "Ta",
                "Tc": "Tc",
            }
            # ------------------------------------------------------------
            # Parameter-estimation settings
            # ------------------------------------------------------------

            # True values used in the simulation:
            #
            # Cw = 1200.0 * 100,    winding thermal capacitance [kJ/°C]
            # Cb = 8000.0 * 100,    bearing/body thermal capacitance [kJ/°C]
            # Rwb = 0.02   thermal  resistance between winding and bearing [°C/kW]
            # Rba = 0.035  thermal  resistance between bearing and ambient [°C/kW]
            # Rbc = 0.05  thermal   resistance between bearing and cooling water [°C/kW]

            THETA0 = np.array([
                1.0e5,  # Cw
                5.0e5,  # Cb
                0.02,  # Rwb
                0.035,  # Rba
                0.05,  # Rbc
            ])
            LOWER_BOUND = np.array([
                1.0e4,  # Cw
                1.0e5,  # Cb
                0.005,  # Rwb
                0.005,  # Rba
                0.005,  # Rbc
            ])
            UPPER_BOUND = np.array([
                1.0e6,  # Cw
                5.0e6,  # Cb
                0.10,  # Rwb
                0.10,  # Rba
                0.10,  # Rbc
            ])
        else:
            SENSOR_COLS = [
                "AE PORT GEN.W-WINDING TEMP.",
                "AE PORT END BRG.TEMP.",
                "POWER_kW",
                "FAN 3 TEMPERATURE",
                "LT FW TEMP.",
            ]
            RENAME_MAP = {
                "AE PORT GEN.W-WINDING TEMP.": "Tw",
                "AE PORT END BRG.TEMP.": "Tb",
                "POWER_kW": "P",
                "FAN 3 TEMPERATURE": "Ta",
                "LT FW TEMP.": "Tc",
            }
            # ------------------------------------------------------------
            # Parameter-estimation settings
            # ------------------------------------------------------------

            # True values used in the simulation:
            #
            # Cw = 1200.0 * 100,    winding thermal capacitance [kJ/°C]
            # Cb = 8000.0 * 100,    bearing/body thermal capacitance [kJ/°C]
            # Rwb = 0.02   thermal  resistance between winding and bearing [°C/kW]
            # Rba = 0.035  thermal  resistance between bearing and ambient [°C/kW]
            # Rbc = 0.05  thermal   resistance between bearing and cooling water [°C/kW]

            THETA0 = np.array([
                1.0e6,  # Cw
                5.0e6,  # Cb
                0.02,  # Rwb
                0.035,  # Rba
                0.05,  # Rbc
            ])
            LOWER_BOUND = np.array([
                1.0e4,  # Cw
                1.0e5,  # Cb
                0.005,  # Rwb
                0.005,  # Rba
                0.005,  # Rbc
            ])
            UPPER_BOUND = np.array([
                1.0e7,  # Cw
                5.0e7,  # Cb
                0.10,  # Rwb
                0.10,  # Rba
                0.10,  # Rbc
            ])

        # State vector:
        #
        # x = [Tw, Tb]
        #
        STATE_COLS = [
            "Tw",
            "Tb",
        ]

        # Both states are measured
        MEAS_COLS = [
            "Tw",
            "Tb",
        ]

        # Input vector:
        #
        # u = [P, Ta, Tc]
        #
        INPUT_COLS = [
            "P",
            "Ta",
            "Tc",
        ]

        # theta = [Cw, Cb, Rwb, Rba, Rbc]
        PARAMETER_NAMES = [
            "Cw",
            "Cb",
            "Rwb",
            "Rba",
            "Rbc"
        ]

        # ------------------------------------------------------------
        # EKF matrices
        # ------------------------------------------------------------

        # Process-noise covariance for [Tw, Tb]
        Q = np.diag([
            0.0001,  # Tw process noise
            0.0001,  # Tb process noise
        ])

        # Measurement noise in the generated dataset is:
        #
        # measurement_noise_std = 0.25 °C
        #
        R = np.diag([
            0.25 ** 2,  # Tw measurement variance
            0.25 ** 2,  # Tb measurement variance
        ])

        # Both states are measured directly:
        #
        # y = Cx
        #
        C = np.array([
            [1.0, 0.0],  # Tw
            [0.0, 1.0],  # Tb
        ])
    case "2state":
        SENSOR_COLS = [
            "AE PORT GEN.W-WINDING TEMP.",
            "AE PORT END BRG.TEMP.",
            "AE PORT LUB.OIL TEMP.",
            "POWER_kW",
        ]
        RENAME_MAP = {
            "AE PORT GEN.W-WINDING TEMP.": "T1",
            "AE PORT END BRG.TEMP.": "T2",
            "AE PORT LUB.OIL TEMP.": "Tref",
            "POWER_kW": "P",
        }

        STATE_COLS = ["T1", "T2"]
        MEAS_COLS = ["T1", "T2"]
        INPUT_COLS = ["P", "Tref"]

        PARAMETER_NAMES = ["C1", "C2", "R1", "R2"]

        Q = np.diag([0.01, 0.01])
        R = np.diag([
            0.04 ** 2,
            0.04 ** 2,
        ])
        C = np.array([
            [1.0, 0.0],  # measure T1
            [0.0, 1.0],  # measure T2
        ])

        THETA0 = np.array([1e5, 5e5, 0.010, 0.010])  # C1, C2, R1, R2
        LOWER_BOUND = np.array([1e4, 1e4, 0.002, 0.002])
        UPPER_BOUND = np.array([1e7, 1e7, 0.05, 0.05])

    case _:
        raise ValueError("Invalid MODEL_OPTION")
