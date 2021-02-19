import os

# PCA Through FedSVD
os.system('python3 FedSVD.py -m pca -d load_mnist -p 10 -s 1000 -b 5 -o True')
os.system('python3 FedSVD.py -m pca -d load_wine -p 10 -s 650 -b 5 -o True')
os.system('python3 FedSVD.py -m pca -d load_synthetic -p 10 -s 1000 -f 1000 -a 0.01 -b 5 -o True')
os.system('python3 FedSVD.py -m pca -d load_synthetic -p 10 -s 1000 -f 1000 -a 0.1 -b 5 -o True')
os.system('python3 FedSVD.py -m pca -d load_synthetic -p 10 -s 1000 -f 1000 -a 0.5 -b 5 -o True')
os.system('python3 FedSVD.py -m pca -d load_synthetic -p 10 -s 1000 -f 1000 -a 1.0 -b 5 -o True')
