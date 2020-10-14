import numpy as np
import scipy
import copy
import scipy.io as scio
from sklearn.metrics import mean_squared_error
from math import sqrt

from scipy.stats import poisson, norm, gamma, dirichlet, uniform, beta
import math



def load_data_cold_start(fileName):
    data_file = fileName

    relation_matrix = scio.loadmat(data_file)['datas'].astype(int)
    relation_matrix[relation_matrix>1] = 1

    relation_matrix[(np.arange(relation_matrix.shape[0]), np.arange(relation_matrix.shape[0]))] = 0

    [data_num, col_num] = (relation_matrix.shape)

    test_ratio = 0.1

    test_index = np.sort(np.random.choice(data_num, int(data_num*test_ratio), replace=False))
    train_index_index = np.ones((data_num), dtype = int)
    train_index_index[test_index] = 0
    train_index = np.arange(data_num)[train_index_index==1]

    train_matrix = copy.copy(relation_matrix[np.ix_(train_index, train_index)])
    test_matrix_1 = copy.copy(relation_matrix[np.ix_(train_index, test_index)])
    test_matrix_2 = copy.copy(relation_matrix[np.ix_(test_index, train_index)])

    data_num = len(train_index)

    return train_matrix, data_num, train_index, test_index, test_matrix_1, test_matrix_2



def load_data_fan(fileName):
    # Initialize the Coordinates of each point to [dataNum**2,2] matrix
    # Initialization relation to [dataNum**2, 1] matrix
    #  print('Please input the data file Name:')
    #  dataFile = raw_input()

    data_file = fileName

    relation_matrix = scio.loadmat(data_file)['datas'].astype(int)
    relation_matrix[relation_matrix>1] = 1
    relation_matrix[(np.arange(relation_matrix.shape[0]), np.arange(relation_matrix.shape[0]))] = 0
    [data_num, col_num] = (relation_matrix.shape)
    test_matrix = np.ones(relation_matrix.shape)*(-1)
    test_ratio = 0.1
    for ii in range(data_num):
        test_index_i = np.random.choice(col_num, int(col_num*test_ratio), replace=False)
        test_matrix[ii, test_index_i] = copy.copy(relation_matrix[ii, test_index_i])
        relation_matrix[ii, test_index_i] = -1

    return relation_matrix, data_num, test_matrix


#
#
# def calcualteAUC(y_true, y_scores):
#     n1 = np.sum(y_true)
#     no = len(y_true) - n1
#     rank_indcs =np.argsort(y_scores)
#     R_sorted = y_true[rank_indcs]
#     #+1 because indices in matlab begins with 1
#     # #however in python, begins with 0
#     So=np.sum(np.where(R_sorted>0)[0]+1)
#     aucValue = float(So - (n1*(n1+1))/2)/(n1*no)
#     return aucValue
#


def initialize_model(dataR, dataR_H, dataNum, KK, LL, feaMat):
    # Input:
    # dataR: positive relational data # positive edges x 2
    # KK: number of communities
    # LL: number of features
    # feaMat: feature matrix N X K

    # Output:
    # M: Poisson distribution parameter in generating X_{ik}
    # X_i: latent counts for node i
    # Z_ik: latent integers summary, calculating as \sum_{j,k_2} Z_{ij,kk_2}
    # Z_k1k2: latent integers summary, calculating as \sum_{k,k_2} Z_{ij,kk_2}
    # pis: LL X N X KK: layer-wise mixed-membership distributions
    # FT: F X K, feature transition coefficients
    # betas: LL X N X N: layer-wise information propagation coefficient
    # Lambdas: community compatibility matrix
    # QQ: scaling parameters for Lambdas
    # scala_val: not use at the momment

    pis = np.zeros((LL, dataNum, KK))

    betas = gamma.rvs(1, 1, size=(LL-1, dataNum, dataNum))
    FT = gamma.rvs(1, 1, size=(feaMat.shape[1], KK))

    pis_ll = np.dot(feaMat, FT)+0.1

    psi_inte = gamma.rvs(a = pis_ll/(1+0.01), scale = 1)
    psi_inte = psi_inte/(np.sum(psi_inte, axis=1)[:, np.newaxis])+1e-6
    pis[-1] = psi_inte/(np.sum(psi_inte, axis=1)[:, np.newaxis])

    for ll in np.arange(LL-2, -1, -1):  #  From LL-2 to 0

        psi_ll = np.dot(betas[ll].T, pis[ll+1])

        psi_ll += 0.01 #
        psi_inte = gamma.rvs(a = psi_ll/(1+0.01), scale = 1)
        psi_inte = psi_inte/(np.sum(psi_inte, axis=1)[:, np.newaxis])+1e-6
        pis[ll] = psi_inte/(np.sum(psi_inte, axis=1)[:, np.newaxis])

    # for ii in range(dataNum):
    #     pis[-1][ii] = dirichlet.rvs(pis_ll[ii])
    #
    # for ll in np.arange(LL-2, -1, -1):  #  From LL-2 to 0
    #     psi_ll = np.dot(betas[ll].T, pis[ll+1])     ########################### update here
    #     psi_ll += 0.1 #
    #     for ii in range(dataNum):
    #         pis[ll, ii] = dirichlet.rvs(psi_ll[ii])

    M = dataNum
    X_i = poisson.rvs(M*pis[0]).astype(int)

################################
################################


    R_KK = np.ones((KK, KK)) / (KK ** 2)
    np.fill_diagonal(R_KK, 1 / KK)
    Lambdas = gamma.rvs(a=R_KK, scale=1)


    # k_Lambda = 1/KK
    # c_val_Lambda = 1
    # r_k = gamma.rvs(a = k_Lambda, scale = 1, size = KK)/c_val_Lambda
    #
    # Lambdas = np.dot(r_k.reshape((-1, 1)), r_k.reshape((1, -1)))
    # epsilon = 1
    # np.fill_diagonal(Lambdas, epsilon*r_k)

################################
################################


    Z_ik = np.zeros((dataNum, KK), dtype=int)
    Z_k1k2 = np.zeros((KK, KK), dtype=int)
    for ii in range(len(dataR)):
        pois_lambda = (X_i[dataR[ii][0]][:, np.newaxis] * X_i[dataR[ii][1]][np.newaxis, :]) * Lambdas
        total_val = positive_poisson_sample(np.sum(pois_lambda))

        new_counts = np.random.multinomial(total_val, pois_lambda.reshape((-1)) / np.sum(pois_lambda)).reshape((KK, KK))
        Z_k1k2 += new_counts
        Z_ik[dataR[ii][0]] += np.sum(new_counts, axis=1)
        Z_ik[dataR[ii][1]] += np.sum(new_counts, axis=0)

    return M, X_i, Z_ik, Z_k1k2, pis, FT, betas, Lambdas


################################
################################
def positive_poisson_sample(z_lambda):
    # return positive truncated poisson random variables Z = 1, 2, 3, 4, ...
    # z_lambda: parameter for Poisson distribution

    candidate = 1000
    can_val = np.arange(1, candidate)
    log_vals = can_val*np.log(z_lambda)-np.cumsum(np.log(can_val))
    vals = np.exp(log_vals - np.max(log_vals))

    select_val = np.random.choice(can_val, p = (vals/np.sum(vals)))
    return select_val

def CRT_sample(n_customer, alpha_val):
    return np.sum(uniform.rvs(size = n_customer) < alpha_val/(alpha_val+np.arange(n_customer)))
################################
################################


class sDGRM_class:
    def __init__(self, dataNum, LL, KK, M, X_i, Z_ik, Z_k1k2, pis, FT, betas, FF, feaMat, Lambdas):
        self.feaMat = feaMat
        self.dataNum = dataNum
        self.LL = LL
        self.KK = KK
        self.Lambdas = Lambdas

        self.M = M

        self.X_i = X_i

        self.Z_ik = Z_ik
        self.Z_k1k2 = Z_k1k2

        self.pis = pis  #LL X N X K
        self.FT = FT
        self.betas = betas
        self.alphas = 0.1

        self.FF = FF



        # hyper-parameters
        self.gamma_1_l = np.ones(LL)
        self.gamma_0_l = np.ones(LL)
        self.c_l = np.ones(LL)


        self.r_k = np.ones(self.KK)/self.KK
        self.epsilon = 1
        self.beta_lambda = 1
        self.c_lambda = 1
        self.gamma_0_lambda = 1
        self.c_0_lambda = 1
        self.f_0_lambda = 1e-2
        self.e_0_lambda = 1e-2


    def hyper_parameter_beta(self, qil, dataR, z_ik):
        e_0 = 1.0
        f_0 = 1.0
        g_0 = 1.0
        h_0 = 1.0

        gamma_0 = 1.0
        c_0 = 1.0


        # sampling J_{i'i}^{(l)}
        # J_ii_L = np.zeros((self.LL-1, self.dataNum, self.dataNum), dtype=int)
        J_L_1 = np.zeros(self.LL-1)
        J_L_0 = np.zeros(self.LL-1)
        for ll in range(self.LL-1):
            for ij in range(dataR.shape[1]):
                con_0 = dataR[0, ij]
                con_1 = dataR[1, ij]
                if z_ik[ll, con_0, con_1]>0:
                    J_L_1[ll] += CRT_sample(int(z_ik[ll, con_0, con_1]), self.gamma_1_l[ll])
            for ii in range(len(qil[0])):
                J_L_0[ll] += CRT_sample(int(z_ik[ll, ii, ii]), self.gamma_0_l[ll])

        # sampling gamma_i_l, gamma_l

        n_1_l = np.zeros(self.LL-1)
        n_0_l = np.zeros(self.LL-1)
        for ll in range(self.LL-1):
            inte_val = np.log((self.c_l[ll] - np.log(qil[ll])))-np.log(self.c_l[ll])

            n_1_l[ll] = np.sum(inte_val[dataR[:, 0]])
            n_0_l[ll] = np.sum(inte_val)

            self.gamma_1_l[ll] = gamma.rvs(a = gamma_0+J_L_1[ll], scale = 1)/(c_0+n_1_l[ll])

            self.gamma_0_l[ll] = gamma.rvs(a= gamma_0+J_L_0[ll], scale = 1)/(c_0+n_0_l[ll])

        # sampling c_l
        for ll in range(self.LL-1):
            self.c_l[ll] = gamma.rvs(a =g_0 + self.dataNum*self.gamma_0_l[ll]+len(dataR)*(self.gamma_1_l[ll]), scale = 1)/(h_0 + (np.sum(self.betas[ll])))


    def sample_hyper_Lambda(self, dataR_matrix):
        idx = (dataR_matrix != (-1))
        np.fill_diagonal(idx, 0)
        Phi_KK = np.dot(np.dot(self.X_i.T, idx), self.X_i)
        np.fill_diagonal(Phi_KK, np.diag(Phi_KK)/2)


        # sample r_k
        L_KK = np.zeros((self.KK, self.KK))
        R_KK = np.dot(self.r_k.reshape((-1, 1)), self.r_k.reshape((1, -1)))
        np.fill_diagonal(R_KK, (self.r_k)*self.epsilon)

        p_kk_prime_one_minus = self.beta_lambda / (self.beta_lambda + Phi_KK)
        for k1 in range(self.KK):
            for k2 in range(self.KK):
                if self.Z_k1k2[k1, k2]>0:
                    L_KK[k1, k2] = CRT_sample(self.Z_k1k2[k1, k2], R_KK[k1, k2])
            add_val = np.sum(R_KK[k1]/self.r_k[k1]*np.log(p_kk_prime_one_minus[k1]))
            self.r_k[k1] = gamma.rvs(a = self.gamma_0_lambda/self.KK+np.sum(L_KK[k1]), scale = 1)/(self.c_0_lambda-add_val)


        # sample epsilon
        add_val = np.sum(self.r_k * np.log(np.diag(p_kk_prime_one_minus)))
        self.epsilon = gamma.rvs(a = self.e_0_lambda+np.sum(np.diag(L_KK)), scale = 1)/(self.f_0_lambda - add_val)

        # sample lambda
        self.Lambdas = gamma.rvs(a = self.Z_k1k2 + R_KK, scale = 1)/(self.beta_lambda+Phi_KK)

        # sample beta
        self.beta_lambda = gamma.rvs(a = 1+np.sum(R_KK), scale = 1)/(1+np.sum(self.Lambdas))

        # sample c_0_lambda
        self.c_0_lambda = gamma.rvs(a=1+self.gamma_0_lambda, scale = 1)/(1+np.sum(self.r_k))



    def back_propagate_fan(self, dataR_H):
        # Back propagate the latent counts from X_i to the feature layer
        # dataR_H: the non-zeros locations of \beta (the information propagation matrix)
        # m_ik: LL X N X KK: layer-wise latent counting statistics matrix
        # y_ik: auxiliary values introduced in back propagation
        # q_il, z_L_sum_i, z_ik_sum_k: auxiliary variables used


        m_ik = np.zeros((self.LL, self.dataNum, self.KK))
        m_ik[0] = self.X_i

        y_ik = np.zeros((self.LL, self.dataNum, self.KK))
        # z_ik = np.zeros((self.LL - 1, self.dataNum, self.dataNum, self.KK))

        z_ik_sum_k = np.zeros((self.LL-1, self.dataNum, self.dataNum))
        q_il = np.zeros((self.LL, self.dataNum))
        for ll in range(self.LL - 1):

            propa_mat = np.zeros((self.dataNum, self.dataNum))
            propa_mat[dataR_H==1] = self.betas[ll]
            psi_ll_kk = self.pis[ll + 1][:, np.newaxis, :] * (propa_mat)[:, :, np.newaxis]

            psi_ll = np.sum(psi_ll_kk, axis=0)

            latent_count_i = np.sum(m_ik[ll], axis=1).astype(float)
            beta_para_1 = np.sum(psi_ll, axis=1)

            # latent_count_i += 1e-6
            # beta_para_1 += 1e-6
            #
            # judge_beta = beta_para_1 + latent_count_i
            #
            # judge_bool = (judge_beta < 0.1)
            # beta_para_1[judge_bool] = 0.1
            # latent_count_i[judge_bool] = 0.1
            inte1 = gamma.rvs(a = beta_para_1 + 1e-16, scale = 1)+ 1e-16
            inte2 = gamma.rvs(a = latent_count_i + 1e-16, scale = 1)+ 1e-16
            qil_val = inte1/(inte1+inte2)
################################
################################

            # qil_val = beta.rvs(beta_para_1, latent_count_i)+1e-16
            q_il[ll] = qil_val
            # q_il[ll] = qil_val/np.sum(qil_val)
################################
################################


            for nn in range(self.dataNum):
                for kk in range(self.KK):

                    if m_ik[ll, nn, kk]>0:
                        y_ik[ll, nn, kk] = np.sum(uniform.rvs(size=int(m_ik[ll, nn, kk])) < psi_ll[nn, kk] / (psi_ll[nn, kk] + np.arange(int(m_ik[ll, nn, kk]))))
                        z_ik_ll_nn_kk = np.random.multinomial(y_ik[ll, nn, kk], psi_ll_kk[:, nn, kk] / psi_ll[nn, kk])

                        z_ik_sum_k[ll, nn] += z_ik_ll_nn_kk

                        m_ik[ll+1, :, kk] += z_ik_ll_nn_kk


        psi_ll_kk = self.feaMat[:, :, np.newaxis]*self.FT[np.newaxis, :, :] # N x F x K
        psi_ll = np.sum(psi_ll_kk, axis=1)+self.alphas # N x K
        z_L_sum_i = np.zeros((self.FF+1, self.KK))

        for nn in range(self.dataNum):
            for kk in range(self.KK):
                if m_ik[-1, nn, kk]>0:
                    y_ik[-1, nn, kk] = np.sum(uniform.rvs(size=int(m_ik[-1, nn, kk])) < psi_ll[nn, kk] / (psi_ll[nn, kk] + np.arange(int(m_ik[-1, nn, kk]))))
                    pp = np.append(psi_ll_kk[nn, :, kk], self.alphas)

                    z_L_sum_i[:, kk] += np.random.multinomial(y_ik[ - 1, nn, kk],pp/np.sum(pp))

        latent_count_i = np.sum(m_ik[-1], axis=1).astype(float)
        beta_para_1 = np.sum(psi_ll, axis=1)

        # latent_count_i += 1e-6
        # beta_para_1 += 1e-6
        # judge_beta = beta_para_1+latent_count_i
        # judge_bool = (judge_beta<0.1)
        # beta_para_1[judge_bool] = 0.1
        # latent_count_i[judge_bool] = 0.1

        inte1 = gamma.rvs(a=beta_para_1+ 1e-16, scale=1) + 1e-16
        inte2 = gamma.rvs(a=latent_count_i+ 1e-16, scale=1)+ 1e-16
        qil_val = inte1 / (inte1 + inte2)

        ################################
################################
        # qil_val = beta.rvs(beta_para_1, latent_count_i)+1e-16
        q_il[-1] = qil_val
        # q_il[-1] = qil_val/np.sum(qil_val)
################################
################################

        return m_ik, y_ik, z_ik_sum_k, q_il, z_L_sum_i


    def sample_pis(self, m_ik, feaMat, dataR_H):
        # layer-wise sample mixed-membership distribution

        prior_para = np.dot(feaMat, self.FT)
        prior_para += self.alphas
        para_nn = prior_para+m_ik[-1]

        # para_nn += 0.01 #
        nn_pis = gamma.rvs(a = para_nn, scale = 1)
        nn_pis = nn_pis/(np.sum(nn_pis, axis=1)[:, np.newaxis])+1e-16

        self.pis[-1] = nn_pis/(np.sum(nn_pis, axis=1)[:, np.newaxis])


        for ll in np.arange(self.LL-2, -1, -1):
            propa_mat = np.zeros((self.dataNum, self.dataNum))
            propa_mat[dataR_H==1] = self.betas[ll]

            psi_ll = np.dot(propa_mat.T, self.pis[ll+1])

            para_nn = psi_ll + m_ik[ll]
            # para_nn += 0.01 #
            nn_pis = gamma.rvs(a = para_nn, scale = 1)
            nn_pis = nn_pis/(np.sum(nn_pis, axis=1)[:, np.newaxis])+1e-16
            self.pis[ll] = nn_pis / (np.sum(nn_pis, axis=1)[:, np.newaxis])



    def sample_X_i(self, dataR_matrix):
        # sample the latent counts X_i

################################
################################

        idx = (dataR_matrix != (-1))
        np.fill_diagonal(idx, False)

        for nn in range(self.dataNum):

            Xik_Lambda = np.sum(np.dot(self.Lambdas, ((idx[nn][:, np.newaxis]*self.X_i).T)), axis=1)+ \
                       np.sum(np.dot(self.Lambdas.T, (idx[:, nn][:, np.newaxis]*self.X_i).T), axis=1)

            log_alpha_X = np.log(self.M)+np.log(self.pis[0][nn])-Xik_Lambda

            for kk in range(self.KK):
                n_X = self.Z_ik[nn, kk]
                if n_X == 0:
                    select_val = poisson.rvs(np.exp(log_alpha_X[kk]))
                else:
                    candidates = np.arange(1, self.dataNum+1) # we did not consider 0 because the ratio is 0 for sure
                    pseudos = candidates*log_alpha_X[kk]+n_X*np.log(candidates)-np.cumsum(np.log(candidates))
                    proportions = np.exp(pseudos-max(pseudos))
                    select_val = np.random.choice(candidates, p=proportions/np.sum(proportions))

                self.X_i[nn, kk] = select_val

################################
################################

    def sample_Lambda_k1k2(self, dataR_matrix):
        # sample Lambda according to the gamma distribution

################################
################################
        idx = (dataR_matrix != (-1))
        np.fill_diagonal(idx, False)
        Phi_KK = np.dot(np.dot(self.X_i.T, idx), self.X_i)

        R_KK = np.ones((self.KK, self.KK))/(self.KK**2)
        np.fill_diagonal(R_KK, 1/self.KK)

        self.Lambdas = gamma.rvs(a = self.Z_k1k2 + R_KK, scale = 1)/(1+Phi_KK)
################################
################################


        # np.fill_diagonal(Phi_KK, np.diag(Phi_KK)/2)
        #
        # Phi_KK_1 = np.zeros((self.KK, self.KK))
        # for i1 in range(self.dataNum):
        #     for i2 in range(self.dataNum):
        #         if idx[i1, i2]:
        #             Phi_KK_1 += np.dot(self.X_i[i1][:, np.newaxis], self.X_i[i2][np.newaxis, :])
        # print(np.sum(abs(Phi_KK-Phi_KK_1)))


        # k_Lambda = 1
        # theta_Lambda_inverse = self.dataNum**2
        #
        # X_counts = np.dot(self.X_i.T, self.X_i)
        # new_k_Lambda = k_Lambda + self.Z_k1k2
        # new_theta_Lambda = 1/(X_counts+theta_Lambda_inverse)
        #
        # self.Lambdas = gamma.rvs(a = new_k_Lambda, scale = new_theta_Lambda)

    def sample_Z_ik_k1k2(self, dataR):
        # sampling the latent integers

        Z_ik = np.zeros((self.dataNum, self.KK), dtype=int)
        Z_k1k2 = np.zeros((self.KK, self.KK), dtype=int)
        for ii in range(len(dataR)):
            pois_lambda = (self.X_i[dataR[ii][0]][:, np.newaxis]*self.X_i[dataR[ii][1]][np.newaxis, :])*self.Lambdas
            total_val = positive_poisson_sample(np.sum(pois_lambda))
            new_counts = np.random.multinomial(total_val, pois_lambda.reshape((-1))/np.sum(pois_lambda)).reshape((self.KK, self.KK))
            Z_k1k2 += new_counts
            Z_ik[dataR[ii][0]] += np.sum(new_counts, axis=1)
            Z_ik[dataR[ii][1]] += np.sum(new_counts, axis=0)

        self.Z_k1k2 = Z_k1k2
        self.Z_ik = Z_ik


    def sample_beta(self, z_ik, q_il, dataR_H):
        # Sampling the information propagation coefficients

        for ll in range(self.betas.shape[0]):
            # self.betas[ll] = gamma.rvs(hyper_alpha -np.log(q_il[ll][np.newaxis, :]), hyper_beta+np.sum(z_ik[ll], axis=2))
            # self.betas[ll] = gamma.rvs(a = hyper_beta+np.sum(z_ik[ll], axis=2), scale = hyper_alpha -np.log(q_il[ll][np.newaxis, :]))
            # temp_gamma = np.dot(self.gamma_i_l[ll].reshape((-1, 1)), np.ones((1,self.dataNum)))
            temp_gamma = np.ones((self.dataNum, self.dataNum))*self.gamma_1_l[ll]
            np.fill_diagonal(temp_gamma, self.gamma_0_l[ll])

            temp_qil = np.ones((self.dataNum, 1)).dot(q_il[ll].reshape((1, -1)))

            posterior_a = (temp_gamma + z_ik[ll])[dataR_H == 1]
            posterior_inverse_scale = (self.c_l[ll] - np.log(temp_qil))[dataR_H == 1]

            self.betas[ll] = gamma.rvs(a=posterior_a, scale=1) / posterior_inverse_scale


#     def sample_beta(self, z_ik, q_il, dataR_H):
#         # Sampling the information propagation coefficients
#         hyper_alpha = 1
#         hyper_beta = 1
#         for ll in range(self.betas.shape[0]):
#             # self.betas[ll] = gamma.rvs(hyper_alpha -np.log(q_il[ll][np.newaxis, :]), hyper_beta+np.sum(z_ik[ll], axis=2))
#             # self.betas[ll] = gamma.rvs(a = hyper_beta+np.sum(z_ik[ll], axis=2), scale = hyper_alpha -np.log(q_il[ll][np.newaxis, :]))
#
# ################################
# ################################
#             self.betas[ll] = gamma.rvs(a = hyper_alpha+z_ik[ll], scale = 1)/(hyper_beta -np.log(q_il[ll][:, np.newaxis]))

################################
################################


    def sample_M(self):
        # updating the hyper-parameter M

        # k_M = M_val
        # theta_M_inverse = 1
        self.M = gamma.rvs(a = self.M+np.sum(self.X_i), scale = 1)/(1+self.dataNum)

    def sample_FT(self,z_L, q_il, feaMat):
        # Updating the feature information transition coefficients

        self.FT = gamma.rvs(a = 1+z_L, scale = 1)/(1-np.dot(np.log(q_il[-1]), feaMat)[:, np.newaxis])


    def sample_alpha(self, z_L_alpha, q_il, feaMat):
        # Updating the hyper-parameter alpha

        self.alphas = gamma.rvs(a = 0.1+z_L_alpha, scale = 1)/(1-np.sum(np.log(q_il[-1]))) #############################




