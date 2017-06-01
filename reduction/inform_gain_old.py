from reduction.reduction_base import Reduction
import numpy as np


class InformationGainOld(Reduction):

    def __init__(self, threshold, count_nan=True, progress_bar=None):
        super().__init__(threshold, count_nan, progress_bar)

    def function_abstract(self, X, y):
        def info_gain():
            entropy_x_set = 0
            entropy_x_not_set = 0

            for c in classCnt:
                probas = classCnt[c] / float(featureTot)
                entropy_x_set = entropy_x_set - probas * np.log(probas)
                probas = (class_tot_cnt[c] - classCnt[c]) / float(tot - featureTot)
                entropy_x_not_set = entropy_x_not_set - probas * np.log(probas)

            for c in class_tot_cnt:
                if c not in classCnt:
                    probas = class_tot_cnt[c] / float(tot - featureTot)
                    entropy_x_not_set = entropy_x_not_set - probas * np.log(probas)

            return entropy_before - ((featureTot / float(tot)) * entropy_x_set
                                     + ((tot - featureTot) / float(tot)) * entropy_x_not_set)

        tot = X.shape[0]
        class_tot_cnt = {}
        entropy_before = 0

        for i in y:
            if i not in class_tot_cnt:
                class_tot_cnt[i] = 1
            else:
                class_tot_cnt[i] = class_tot_cnt[i] + 1

        for c in class_tot_cnt:
            probs = class_tot_cnt[c] / float(tot)
            entropy_before = entropy_before - probs * np.log(probs)

        nz = X.T.nonzero()
        pre = 0
        classCnt = {}
        featureTot = 0
        information_gain = []

        for i in range(0, len(nz[0])):
            if i != 0 and nz[0][i] != pre:
                for notappear in range(pre + 1, nz[0][i]):
                    information_gain.append(0)
                ig = info_gain()
                information_gain.append(ig)
                pre = nz[0][i]
                classCnt = {}
                featureTot = 0
            featureTot = featureTot + 1
            yclass = y[nz[1][i]]
            if yclass not in classCnt:
                classCnt[yclass] = 1
            else:
                classCnt[yclass] = classCnt[yclass] + 1

        ig = info_gain()
        information_gain.append(ig)

        return np.asarray(information_gain)
