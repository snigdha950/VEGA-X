"""Central configuration for the RAFTS smart-scan prototype."""

NUM_BANDS = 20
NUM_RECEIVERS = 2
NUM_SLOTS = 500

FREQ_MIN_MHZ = 500.0
FREQ_MAX_MHZ = 18000.0

PD = 0.90
PFA = 0.01
SENSITIVITY_DB = -120.0  # simulation parameter, not a calibrated hardware claim

RETUNE_DELAY_SLOTS = 1
MAX_AGE_SLOTS = 80

# Evaluation reward. The online schedulers never receive hidden TP/FP labels;
# these weights are used only after a run to report a transparent net reward.
TRUE_INTERCEPT_REWARD = 1.0
FALSE_ALARM_COST = 1.0

# Frozen parameters of the selected CUSUM-UCB learner.
UCB_EXPLORATION = 2.0
CUSUM_DRIFT = 0.05
CUSUM_THRESHOLD = 4.0

TEMPORAL_MIN_HITS = 4
TEMPORAL_MIN_CONFIDENCE = 0.65
TEMPORAL_MAX_BONUS = 0.20
TEMPORAL_TOLERANCE_SLOTS = 2

RANDOM_SEED = 2026
