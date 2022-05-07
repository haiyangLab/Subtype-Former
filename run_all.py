import os
import numpy as np
import random

seed = 42
np.random.seed(seed)
random.seed(seed)


def main():
    cancer_dict = ['BRCA', 'UCEC', 'HNSC', 'THCA', 'LUAD', 'KIRC', 'PRAD', 'LUSC', 'SKCM', 'STAD', 'ALL']
    for cancer in cancer_dict:
        if cancer == 'ALL':
            cmd = 'python SubtypeFormer.py' + ' -c ' + cancer
            print(cmd)
            os.system(cmd)
            cmd = 'python SubtypeFormer.py -m tsne' + ' -c ' + cancer
            print(cmd)
            os.system(cmd)
        else:
            for method in ["SubtypeFormer", "cc", "nmiari", "tsne"]:
                if method == 'nmiari':
                    cmd = 'python SubtypeFormer.py' + ' -c ' + cancer + ' -d dataset_2'
                    print(cmd)
                    os.system(cmd)
                    cmd = 'python SubtypeFormer.py -m ' + method + ' -c ' + cancer
                    print(cmd)
                    os.system(cmd)
                else:
                    cmd = 'python SubtypeFormer.py -m ' + method + ' -c ' + cancer
                    print(cmd)
                    os.system(cmd)


if __name__ == "__main__":
    # run all of the task, including: subtype, Consensus clustering, nmi and ari, and tsne
    main()
