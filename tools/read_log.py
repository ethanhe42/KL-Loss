import matplotlib as mpl
mpl.use('Agg')
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def read_log(fname):
    with open(fname) as f:
        data = json.load(f)
    print(data['cfg'])
    log = data['log']
    entry = log[0].keys()
    data = []
    for i in log:
        data.append(i.values())
    df = pd.DataFrame(data, columns=entry)
    df.index = df.iter
    for i in ['time', 'eta', 'iter', 'mb_qsize', 'mem']:
        df = df.drop(i, axis=1)
    df = df.astype(float)
    df.plot(ylim=[0, 2]secondary_y=['accuracy'])
    savepth = os.path.splitext(fname)[0]+'.pdf'
    print('save to ', savepth)
    plt.savefig(savepth)


if __name__ == '__main__':
    read_log(sys.argv[1])
