import os
import glob
for d in glob.glob('dataset/raw_images/*/'):
    count = len(glob.glob(os.path.join(d, '*')))
    print(f'{d}: {count}')
