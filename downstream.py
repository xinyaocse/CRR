import argparse

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, \
    recall_score, f1_score

from src.load_data import *
from src.perturb_mechanism import *

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data',
                    default='cancer', type=str,
                    help='the name of the dataset.',
                    choices=['cancer', 'iris', 'auto', 'gas'])
parser.add_argument('-e', '--experiment',
                    default=50, type=int,
                    help='repeating times of experiments.')
parser.add_argument('-m', '--mechanism',
                    default='gauss', type=str,
                    help='the perturbation mechanism.',
                    choices=['none', 'gauss', 'lap', 'cgm'])
parser.add_argument('-s', '--scale',
                    default=[1.0], type=float, nargs='+',
                    help='the noise scale.')

args = parser.parse_args()

DAT = args.data
EXP = args.experiment
MEC = args.mechanism
SCA = args.scale
if len(SCA) == 1:
    SCA = SCA[0]


def linear_regression(noisy_X, Y_train, X_test, Y_test):
    # Train
    model = LinearRegression()
    model.fit(noisy_X, Y_train)
    # Test
    pred = model.predict(X_test)
    # Metric
    mse = mean_squared_error(Y_test, pred)
    mae = mean_absolute_error(Y_test, pred)
    r2 = r2_score(Y_test, pred)
    return np.sqrt(mse), mae, r2


def classification(noisy_X, Y_train, X_test, Y_test):
    # Train
    knn = KNeighborsClassifier()
    knn.fit(noisy_X, Y_train)
    # Test
    pred = knn.predict(X_test)
    # Metric
    acc = accuracy_score(pred, Y_test) * 100
    precision = precision_score(pred, Y_test, average='macro') * 100
    recall = recall_score(pred, Y_test, average='macro') * 100
    f1 = f1_score(pred, Y_test, average='macro') * 100
    return acc, precision, recall, f1


def main():
    for i in range(EXP):
        # 1. Load data
        train_x, train_y, test_x, test_y = get_raw_dataset(DAT)
        train_x = normalize(train_x)
        test_x = normalize(test_x)
        # 2. Perturb
        noisy_x = perturb(train_x, MEC, SCA, test_x)
        if DAT == 'auto' or DAT == 'gas':
            # 3. Model learning (regression)
            rmse, mae, r2 = linear_regression(noisy_x, train_y, test_x, test_y)
            # 4. Print result
            print(f'========== Exp {i + 1}/{EXP} ==========')
            print(f'RMSE = {rmse:.2f}')
            print(f'MAE = {mae:.2f}')
            print(f'R2 = {r2:.2f}')
        elif DAT == 'cancer' or DAT == 'iris':
            # 3. Model learning (classification)
            acc, precision, recall, f1 = classification(noisy_x, train_y, test_x, test_y)
            # 4. Print result
            print(f'========== Exp {i + 1}/{EXP} ==========')
            print(f'Accuracy = {acc:.2f}')
            print(f'Precision = {precision:.2f}')
            print(f'Recall = {recall:.2f}')
            print(f'F1 = {f1:.2f}')


if __name__ == "__main__":
    main()
