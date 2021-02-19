import os

# 1 Precision Evaluation
# (1) Reconstruction Error & Scalability Test
for block in [1, 2, 4, 8, 16, 32, 64]:
    for sample in [1000, 2000, 3000, 4000, 5000]:
        os.system('python3 FedSVD.py -d load_synthetic -m svd -f 1000 '
                  '-a 1.0 -p 10 -s %s -b %s -t False -l precision' % (sample, block))
