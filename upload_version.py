import numpy as np
import scipy
import copy
import glob
import scipy.io as sio

import time
import csv
import os

from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import poisson, norm, gamma, bernoulli

from utility_old import *

if __name__ == '__main__':
    relation_data = ['cite_relation']
    data_feature = ['cite_features']

    pathss = ''
    pathss_save = ''

    for name_i_index in [0]:
        name_i = relation_data[name_i_index]
        IterationTime = 4000

        filess = pathss+name_i+'.mat'

        dataR_matrix, dataNum, test_relation = load_data_fan(filess)
        dataR = np.asarray(np.where(dataR_matrix==1)).T
        dataR_test = np.asarray(np.where(test_relation!=-1)).T
        dataR_test_val = test_relation[test_relation!=-1]

        notdelete_index = (dataR_test[:, 0]!=dataR_test[:, 1])
        dataR_test = dataR_test[notdelete_index]
        dataR_test_val = dataR_test_val[notdelete_index]

        KK = 20
        LL = 3

#         feaMat = scio.loadmat(pathss+data_feature[name_i_index]+'.mat')['features'].astype(float)
        feaMat = np.eye(dataNum)
        FF = feaMat.shape[1]


        dataR_H = (dataR_matrix==1).astype(int)+np.eye(dataNum)

        dataR_H[dataR_H>1] = 1

        # model initialization
        M_val, X_i, Z_ik, Z_k1k2, pis, FT, betas, Lambdas = initialize_model(dataR, dataR_H, dataNum, KK, LL, feaMat)

        pass_beta = np.zeros((LL-1, np.sum(dataR_H==1)))
        for l_beta in range(betas.shape[0]):
            pass_beta[l_beta] = (betas[l_beta][dataR_H==1])
        sDGRM = sDGRM_class(dataNum, LL, KK, M_val, X_i, Z_ik, Z_k1k2, pis, FT, pass_beta, FF, feaMat, Lambdas)


        ids = uniform.rvs()
        collectionTime = int(IterationTime/2)
        test_precision_seq = []
        auc_seq = []
        mean_predict = 0
        mean_pis = 0
        mean_beta = 0
        mean_xi = 0
        mean_m_ik0 = 0
        mean_lambda = 0



        for ite in range(IterationTime):

            # start_time = time.time()
            m_ik, y_ik, z_ik, q_il, z_L = sDGRM.back_propagate_fan(dataR_H)
            sDGRM.sample_pis(m_ik, feaMat, dataR_H)

            sDGRM.hyper_parameter_beta(q_il, dataR, z_ik)
            sDGRM.sample_beta(z_ik, q_il, dataR_H)

            sDGRM.sample_FT((z_L[:(-1)]), q_il, feaMat)
            sDGRM.sample_M()
            sDGRM.sample_X_i(dataR_matrix)
            sDGRM.sample_Z_ik_k1k2(dataR)

            sDGRM.sample_Lambda_k1k2(dataR_matrix)
            sDGRM.sample_alpha(np.sum(z_L[-1]), q_il, feaMat)

            predicted_val = np.sum((sDGRM.X_i[dataR_test[:, 0]][:, :, np.newaxis]*sDGRM.X_i[dataR_test[:, 1]][:, np.newaxis, :])*sDGRM.Lambdas[np.newaxis, :, :], axis=(1,2))

            if (ite >collectionTime):
                mean_predict = (mean_predict*(ite - collectionTime-1)+predicted_val)/(ite-collectionTime)
                mean_pis = (mean_pis*(ite - collectionTime-1)+sDGRM.pis)/(ite-collectionTime)
                mean_beta = (mean_beta*(ite - collectionTime-1)+sDGRM.betas)/(ite-collectionTime)
                mean_xi = (mean_xi*(ite - collectionTime-1)+sDGRM.X_i)/(ite-collectionTime)
                mean_m_ik0 = (mean_m_ik0*(ite - collectionTime-1)+m_ik)/(ite-collectionTime)
                mean_lambda = (mean_lambda*(ite - collectionTime-1)+sDGRM.Lambdas)/(ite-collectionTime)

                current_AUC = roc_auc_score(dataR_test_val, mean_predict)
                current_precision = average_precision_score(dataR_test_val, mean_predict)
                auc_seq.append(current_AUC)
                test_precision_seq.append(current_precision)

                np.savez_compressed(
                    pathss_save + 'Add_0_r1_' + name_i + '_LL_' + str(LL) + '_KK_' + str(KK) + '_' + str(ids), pis = mean_pis,
                    dataR_H=dataR_H, m_ik = mean_m_ik0, lambdas = mean_lambda, beta = mean_beta, xi = mean_xi, test_precision_seq=test_precision_seq, auc_seq=auc_seq)
