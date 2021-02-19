import os

# Scalability Test (Time)
for _ in range(10):
    for i in range(10, 101, 10):
        sample = i * 1000
        os.system('python3 FedSVD.py -m svd -t True -d load_synthetic -f 1000 '
                  '-a 1.0 -p 10 -s %s -b %s -l efficiency' % (int(sample/10), i))

# Scalability Test (Communication)
for j in [10000, 20000, 30000, 40000, 50000]:
    for i in [2, 4, 6, 8, 10, 12, 14, 16]:
        os.system('python3 FedSVD.py -m svd -d load_synthetic -f 1000 -a 1.0 '
                  '-p %s -s %s -b 100 -l scalability' % (i, int(j / i)))
