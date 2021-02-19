import os

data_dir = 'dataset'

log_dir = 'log'

results_dir = 'results'

images_dir = 'images'

paths = [data_dir, log_dir, results_dir, images_dir]

for path in paths:
    if os.path.isdir(path) is False:
        os.makedirs(path, exist_ok=True)
