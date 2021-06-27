import numpy as np
import os
import cv2
from functools import partial

class SVD(object):
    def __init__(self, num_img, load_path):
        self.num_img = num_img
        self.load_path = load_path


    def load_data(self):
        for i in range(self.num_img):
            path = self.load_path + str(i) + '.jpg'
            if i == 0:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                train_data, self.unflatten = self.flatten(img)
            else:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()
                train_data = np.vstack((train_data, img))
        train_data = train_data.T
        return train_data
    
    def svd_decomposition(self):
        train_data = self.load_data()
        self.U, self.S, self.V_H = np.linalg.svd(train_data, full_matrices=0, compute_uv=1)
        self.d = len(self.S)
        return self.U, self.S, self.V_H
    
    def reconstruction(self, save_path):
        for i in range(self.d):
            S_i = np.copy(self.S)
            for k in range(len(S_i)):
                if k > i:
                    S_i[k] = 0
            sigma = np.diag(S_i)
            Ap = np.linalg.multi_dot([self.U, sigma, self.V_H])
            dirpath = save_path + str(i)
            isExists = os.path.exists(dirpath)
            if not isExists:
                os.mkdir(dirpath)
            for j in range(Ap.shape[1]):
                img_ap = self.unflatten(Ap[:,j])
                imgpath = dirpath + '/' + str(j) + '.jpg'
                cv2.imwrite(imgpath ,img_ap)
    
    def flatten(self, x):
        original_shape = x.shape
        return x.flatten(), partial(np.reshape, newshape=original_shape)

if __name__ == "__main__":
    num_img = 30
    load_path = './Train/'
    mission = SVD(num_img, load_path)
    U, S, V_H = mission.svd_decomposition()
    save_path = './appro_result/'
    mission.reconstruction(save_path)
    print('Reconstruction Done')

