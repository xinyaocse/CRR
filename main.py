import argparse

from src.load_data import *
from src.perturb_mechanism import *
from src.inference_attack import *
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

        # 3. Inference attack
        test_size = int(data_test.shape[0] * 1)
        attack_result = inference_attack(noisy_data, data_test[:test_size, :], THR)

        # 4. Result analysis
        # (Conducted under the assumption that data are normalized)
        amse_ptb, amse_atk, aes, dp, cdp, ddp, bdp, cadp = result_analyze(attack_result, noisy_data, data_train,
                                                                          MEC, SCA, THR)

        # 5. Print Metrics
        print(f'========== Exp {i + 1}/{EXP} ==========')
        print('AMSE_ptb = ' + np.array2string(amse_ptb, precision=2))
        print('AMSE_atk = ' + np.array2string(amse_atk, precision=2))
        print('AES = ' + np.array2string(aes, precision=2))
        print('DP guarantee = ' + np.array2string(dp, precision=2, max_line_width=9999))
        print('CDP guarantee = ' + np.array2string(cdp, precision=2, max_line_width=9999))
        print('DDP guarantee = ' + np.array2string(ddp, precision=2, max_line_width=9999))
        print('BDP guarantee = ' + np.array2string(bdp, precision=2, max_line_width=9999))
        if MEC == 'gauss' or MEC == 'cgm':
            print('CADP guarantee = ' + np.array2string(cadp, precision=2, max_line_width=9999))


if __name__ == "__main__":
    main()
