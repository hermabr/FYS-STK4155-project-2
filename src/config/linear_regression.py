# Linear regression
T0 = 5
T1 = 50
DEFAULT_INITIAL_ETA = T0 / T1

# SGD ANALYSIS
ANALYSIS_DATA_SIZE = 20
ANALYSIS_DEGREE = 5
ANALYSIS_TEST_SIZE = 0.2

EPOCH_STEP_SIZE = 20
HIGHEST_EPOCH = 120  # TODO: at least 120

N_MINI_BATCH_START = 2
N_MINI_BATCH_END = 120
N_MINI_BATCH_STEP_SIZE = 30  # TODO: maybe 16

NUMBER_OF_ETAS = 10  # TODO: increase for real run
#  SMALLEST_ETA = 0.001
SMALLEST_ETA = 0.1
LARGEST_ETA = 3

NUMBER_OF_LAMBDAS = 5
SMALLEST_LAMBDA = -5
LARGEST_LAMBDA = -1

# TODO: Increase this before the final plotting result
EPOCHS = [1] + list(
    range(EPOCH_STEP_SIZE, HIGHEST_EPOCH + EPOCH_STEP_SIZE, EPOCH_STEP_SIZE)
)
MINIBATCHES = list(range(N_MINI_BATCH_START, N_MINI_BATCH_END, N_MINI_BATCH_STEP_SIZE))
