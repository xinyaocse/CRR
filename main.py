import argparse
from tqdm import trange

from src.load_data import *
from src.perturb_mechanism import *
from src.reconstruction import *
from src.result_analyze import *

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data',
                    default='cancer', type=str,
                    help='the name of the dataset.',
                    choices=['cancer', 'iris', 'auto', 'gas'])
parser.add_argument('-a', '--artificial',
                    default=False, type=bool,
                    help='use synthetic data.')
parser.add_argument('-c', '--correlation',
                    default='0.8', type=float,
                    help='the correlation coefficient of synthetic data.',
                    choices=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, -0.2, -0.4, -0.6, -0.8, -1.0])
parser.add_argument('-e', '--experiment',
                    default=50, type=int,
                    help='repeating times of experiments.')
parser.add_argument('-m', '--mechanism',
                    default='gauss', type=str,
                    help='the perturbation mechanism.',
                    choices=['gauss', 'lap', 'cgm'])
parser.add_argument('-s', '--scale',
                    default=[1.0], type=float, nargs='+',
                    help='the noise scale.')
parser.add_argument('-r', '--rho',
                    default=0.99, type=float,
                    help='the correlation coefficient of noise for 2-D data.')
parser.add_argument('-t', '--threshold',
                    default=0.75, type=float,
                    help='the threshold of inference attack.')

args = parser.parse_args()

DAT = args.data
ART = args.artificial
COR = args.correlation
EXP = args.experiment
MEC = args.mechanism
SCA = args.scale
if len(SCA) == 1:
    SCA = SCA[0]
RHO = args.rho
THR = args.threshold


def main():
    for i in range(EXP):
        # Synthetic dataset
        if ART:
            # 1. Load data
            data_train, data_test = get_synthetic_data(COR)
            # 2. Perturb
            if MEC == 'cgm':
                noisy_data = corr_gaussian_2d(data_train, SCA, RHO)
            else:
                noisy_data = perturb(data_train, MEC, SCA)
        # Real dataset
        else:
            # 1. Load data
            data_train, data_test = get_processed_dataset(DAT)
            # 2. Perturb
            noisy_data = perturb(data_train, MEC, SCA)

        # 3. Reconstruction
        test_size = int(data_test.shape[0] * 1)
        attack_result = reconstruction(noisy_data, data_test[:test_size, :], THR)

        # 4. Result analysis
        mse_ptb, mse_rec, rg = result_analyze(attack_result, noisy_data,data_train)

        # 5. Print Metrics
        print(f'========== Exp {i + 1}/{EXP} ==========')
        print('MSE_ptb = ' + np.array2string(mse_ptb, precision=2))
        print('MSE_rec = ' + np.array2string(mse_rec, precision=2))
        print('RG = ' + np.array2string(rg, precision=2))


if __name__ == "__main__":
    main()
