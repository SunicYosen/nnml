#! /usr/bin/python3

import numpy as np 
import random

def random_j_not_i_from_m(i, m): # Select a j for (0:m) not i randomly
    j = i
    while(j==i):
        j = int(random.uniform(0,m))
    
    return j

def clip_alpha(aj, H, L): # make sure (L <= aj <= H)
    if H < L:
        raise NameError("H Must bigger than L")

    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    else:
        aj = aj
    return aj

def calculate_ek(svm_s, k):
    f_xk = float(np.multiply(svm_s.alphas, svm_s.label_mat).T * svm_s.Kernel[:,k] + svm_s.b)

    ek   = f_xk - float(svm_s.label_mat[k])
    return ek

def select_j_ej(i, ei, svm_s):
    max_k = -1
    max_delta_e = 0
    ej    = 0
    svm_s.eCache[i] = [1, ei]
    valid_ecache_list = np.nonzero(svm_s.eCache[:, 0].A)[0]

    if(len(valid_ecache_list)) > 1:
        for k in valid_ecache_list:
            if k == i:
                continue
            ek = calculate_ek(svm_s, k)
            delta_e = abs(ei - ek)
            
            if(delta_e > max_delta_e):
                max_k = k
                max_delta_e = delta_e
                ej = ek

        return max_k, ej
            
    else:
        j = random_j_not_i_from_m(i, svm_s.m)
        ej = calculate_ek(svm_s, j)

    return j, ej

def updata_ek(svm_s, k):
    ek = calculate_ek(svm_s, k)
    svm_s.eCache[k] = [1, ek]

def inner_l(i, svm_s): # Check KKT
    ei = calculate_ek(svm_s, i)
    if((svm_s.label_mat[i] * ei < -svm_s.toler) & (svm_s.alphas[i] < svm_s.C)) | ((svm_s.label_mat[i] * ei > svm_s.toler) & (svm_s.alphas[i] > 0)):
        j, ej = select_j_ej(i, ei, svm_s)
        alpha_i_old = svm_s.alphas[i].copy()
        alpha_j_old = svm_s.alphas[j].copy()
        
        if(svm_s.label_mat[i] != svm_s.label_mat[j]):
            L = max(0, svm_s.alphas[j]-svm_s.alphas[i])
            H = min(svm_s.C, svm_s.C + svm_s.alphas[j] - svm_s.alphas[i])
        else:
            L = max(0, svm_s.alphas[j] + svm_s.alphas[i] - svm_s.C)
            H = min(svm_s.C, svm_s.alphas[j] + svm_s.alphas[i])

        if L == H:
            # print("L == H")
            return 0

        eta = 2.0 * svm_s.Kernel[i,j] - svm_s.Kernel[i,i] - svm_s.Kernel[j,j]

        if eta >= 0:
            print("eta >= 0")
            return 0

        svm_s.alphas[j] -= svm_s.label_mat[j] * (ei - ej) / eta
        svm_s.alphas[j] = clip_alpha(svm_s.alphas[j], H, L)
        updata_ek(svm_s, j)

        if(abs(svm_s.alphas[j] - alpha_j_old) < svm_s.toler):
            # print("j Not moving enough")
            return 0

        svm_s.alphas[i] += svm_s.label_mat[j] * svm_s.label_mat[i] * (alpha_j_old - svm_s.alphas[j])

        updata_ek(svm_s, i)

        b1 = svm_s.b - ei - svm_s.label_mat[i] * (svm_s.alphas[i] - alpha_i_old) * svm_s.Kernel[i,i] - svm_s.label_mat[j] * (svm_s.alphas[j] - alpha_j_old) * svm_s.Kernel[i,j]

        b2 = svm_s.b - ej - svm_s.label_mat[i] * (svm_s.alphas[i] - alpha_i_old) * svm_s.Kernel[i,j] - svm_s.label_mat[j] * (svm_s.alphas[j] - alpha_j_old) * svm_s.Kernel[j,j]

        if (0 < svm_s.alphas[i] < svm_s.C):
            svm_s.b = b1
        elif (0 < svm_s.alphas[j] < svm_s.C):
            svm_s.b = b2
        else:
            svm_s.b = (b1 + b2) / 2.0

        return 1
    else:
        return 0

            