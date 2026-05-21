# Σ-Guard
Reconstruction-Robust Differentially Private Multi-Attribute Data Publishing

## Environmental Setup

The project primarily relies on the following libraries. Before running the project, ensure that you have them installed in your environment.

```
pip3 install numpy pandas scikit-learn
```

Please ensure that the installed versions of these libraries are compatible with each other. It's recommended to create a virtual environment for the project to avoid conflicts with other projects.

## Structure of the files

The directory and structure of the files are as follows:

```
|-- data ...                    # folder that stores data
    | -- processed              # normalized dataset without missing values
    | -- raw                    # unprocessed dataset
    | -- synthetic              # artificial 2-d data with specific correlations
|-- src ...                     # folder that stores source code
    | -- reconstruction.py    # core for ρ-Rec
    | -- load_data.py           # util for loading data
    | -- perturb_mechanism.py   # util for perturbation, core for CGM
    | -- result_analyze.py      # util for analyzing reconstruction results
README.md                       # project description
downstream.py                   # program entry for downstream evaluation
main.py                         # program entry for reconstruction
```

## Running the Experiments

Our experimental pipeline consists of two main parts : *ρ-Rec reconstruction* and *downstream task*.

### 1. ρ-Rec reconstruction

To run ρ-Rec reconstruction on real dataset, use the following example command:

```
python main.py -d gas -e 50 -m gauss -s 1 -t 0.75
```

For 2-d synthetic data, use the following command:

```
python main.py -a True -e 50 -c 0.8 -m gauss -s 1 -r 0.99 -t 0
```


### 2. downstream task

To run downstream task evaluation, use the following command:

```
python downstream.py -d gas -e 50 -m gauss -s 0.1
```

**Parameters**
- `-d`: the name of the dataset, inactivated when "-a = True"
- `-a`: use artificial data
- `-c`: the correlation coefficient of artificial data, only activated when "-a = True"
- `-e`: repeating times of experiments
- `-m`: the perturbation mechanism
- `-s`: the noise scale
- `-r`: the correlation coefficient of noise for CGM, only activated when "-a = True" and "-m = cgm"
- `-t`: the threshold for reconstruction

## Datasets

You will find the publicly origin datasets here:
- [Gas](https://archive.ics.uci.edu/dataset/322/gas+sensor+array+under+dynamic+gas+mixtures)
- [Iris](https://archive.ics.uci.edu/dataset/53/iris)
- [Auto](https://archive.ics.uci.edu/dataset/9/auto+mpg)
- [Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
