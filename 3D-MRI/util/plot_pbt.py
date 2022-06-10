import math
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings("ignore")

def make_dataset(dir, file_ext=[]):
    paths = []
    assert os.path.exists(dir) and os.path.isdir(dir), '{} is not a valid directory'.format(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            for ext in file_ext:
                if fname.endswith(ext):
                    path = os.path.join(root, fname)
                    paths.append(path)
    return paths

def plotPBT(path):
    name = path.split('/')[-2]
    paths = sorted(make_dataset(path, ['result.json']))
    scores = []
    for i, path in enumerate(paths):
        scores.append([])
        with open(path, 'r') as f:
            for line in f:
                step_line = json.loads(line.rstrip())
                scores[-1].append(step_line['score'])
    max_iter = max(list(map(len, scores)))
    plt.figure()
    for i in range(len(scores)):
        plt.plot(scores[i])


    x = int(math.ceil(max_iter*1.1/10.0))*10
    plt.plot(list(range(x)), [0.15]*x, 'r--')
    plt.legend([*['_nolegend_']*len(scores), '15% error mark'])
    plt.xlabel("Steps")
    plt.ylabel("Mean Relative Error")
    plt.ylim(bottom=0)
    plt.savefig('%s.png'%name, format='png', bbox_inches='tight')

if __name__ == "__main__":
    plotPBT('/home/kreitnerl/mrs-gan/ray_results/test_feat/')
