## FedSVD

#### Prepare the environment

Firstly, make sure python3 is installed on your machine. 

Then install the required packages by :

```bash
pip insatll -r requirements.txt
```

#### Reproduce the trials in the paper

The precision evaluation

```bash
# Run the trials
python evaluation_reconstruction_error.py
python evaluation_pca.py
```

The scalability test

```bash
# Run the trials
python evaluation_scalability.py
```

The privacy evaluation

```bash
# Run the trials
python evaluation_privacy.py -d mnist
python evaluation_privacy.py -d cifar10
```

#### Reproduce the plots in the paper

We uploaded the history results to the anonymous GitHub, thus the following plots could be directly executed without running the trials.

```bash
# Plot the reconstruction error
python plot_figure3.py

# DP-FdPCA and plot the figures
# (Need to run "python evaluation_pca.py" first)
python3 DP_FedPCA_plot_figure4.py
python3 DP_FedPCA_plot_figure5.py

# Plot the reconstruction error
python plot_figure6a.py
python plot_figure6b.py
python plot_figure6c.py
```