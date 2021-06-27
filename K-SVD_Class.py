from SVD_Class import SVD
import numpy as np
import os
import cv2
from functools import partial
from sklearn import preprocessing
from sklearn import linear_model
import pdb

class K_SVD(SVD):
    def __init__(self, num_img, load_path, patch_size):
        super(K_SVD, self).__init__(num_img, load_path)
        self.patch_size = patch_size

    def load_new_data(self):
        for i in range(self.num_img):
            path = self.load_path + str(i) + '.jpg'
            if i == 0:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                self.img_shape = img.shape
                train_data_stack, _ = self.patch_partition(img)
            else:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img_patch, _ = self.patch_partition(img)
                train_data_stack = np.hstack((train_data_stack, img_patch))
        train_data_stack = train_data_stack.astype(np.float)
        train_data = train_data_stack
        train_data = self.data_norm(train_data)
        return train_data
    
    def train(self, train_data, K=628, iter = 80):
        # Initialization Dictionary
        n = self.patch_size * self.patch_size
        dic = np.random.random((n, K))
        dic = preprocessing.normalize(dic, norm='l2', axis=0)
        # Train
        try:
            for i in range(iter):
                # first_element = dic[0][0]
                first_element = dic[:,0]
                x = linear_model.orthogonal_mp(dic, train_data)
                # x = decomposition.sparse_encode(train_data.T, dic)
                error = np.linalg.norm(train_data - np.dot(dic, x))
                print('step: %d error: %f'% (i, error))
                dic = self.dic_upgrade(train_data, dic, x, K)
                dic[:,0] = first_element
        except KeyboardInterrupt:
            pass
        np.save('Dic', dic)
        print('Dictionary saved')
        return dic
    
    def test(self, dic, test_data_list, result_file, loss):
        repaired_img_list = []
        for i in range(len(test_data_list)):
            repaired = np.zeros_like(test_data)
            for j in range(repaired.shape[1]):
                index = np.nonzero(test_data_list[i][:,j])[0]
                if not index.shape[0]:
                    continue
                normalize = np.linalg.norm(test_data_list[i][:,j][index])
                mean = np.mean(test_data_list[i][:,j][index])
                test_data_norm_col = (test_data_list[i][:,j][index]-mean)/normalize
                x = linear_model.orthogonal_mp(dic[index,:], test_data_norm_col)
                repaired[:,j] = np.dot(dic, x)*normalize + mean
            repaired_img_list.append(repaired)
        
        for i in range(len(test_data_list)):
            repaired_img_stack =  repaired_img_list[i]
            result_file_i = result_file + str(loss[i])
            isExists = os.path.exists(result_file_i)
            if not isExists:
                os.mkdir(result_file_i)
            repaired_img_stack = np.array_split(repaired_img_stack, num_test_imgs_per_file, axis=1)
            for j in range(num_test_imgs_per_file):
                repaired_img = repaired_img_stack[j]
                repaired_img = self.patch_fuse(repaired_img, self.test_unreshape_patch)
                result_img_path = result_file_i + '/' + str(j) + '.png'
                cv2.imwrite(result_img_path, repaired_img)
        print('Result Saved')
    
    def load_test_data(self, test_path, num_test_imgs_per_file):
        for j in range(num_test_imgs_per_file):
            img_path = test_path + str(j) + '.png'
            if j ==0:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                test_data_stack, self.test_unreshape_patch= self.patch_partition(img)
                test_data = test_data_stack
            else:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_patch, _ = self.patch_partition(img)
                test_data = np.hstack((test_data, img_patch))
            test_data = test_data.astype(np.float)
        return test_data
            
    def reshape(self, x, aim_shape):
        original_shape = x.shape
        return np.reshape(x, aim_shape), partial(np.reshape, newshape=original_shape)
    
    def data_norm(self, X):
        #for data
        for i in range(X.shape[1]):
            norm = np.linalg.norm(X[:,i])
            mean = np.mean(X[:,i])
            X[:, i] = np.copy((X[:, i].T-mean) / norm)
        return X
    
    def dic_upgrade(self, y, D, x, K):
        for i in range(K):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue
            D[:, i] = 0
            E = (y - np.dot(D, x))[:, index]
            U_, S_, V_ = np.linalg.svd(E, full_matrices=0, compute_uv=1)
            D[:, i] = U_[:, 0].T
            x[i, index] = S_[0] * V_[0, :]
        return D
    
    def patch_partition(self, img):
        # horizontal cutting
        img_patch = np.array_split(img, img.shape[0] // self.patch_size)
        # vertical cutting
        for i in range(len(img_patch)):
            img_patch[i] = np.array_split(img_patch[i], img.shape[1] // self.patch_size, axis=1)
        # flattening and concatenate
        new_shape = (self.patch_size*self.patch_size, 1)
        for i in range(img.shape[0] // self.patch_size):
            for j in range(img.shape[1] // self.patch_size):
                if (i==0 and j==0):
                    data_patch, unreshape_patch = self.reshape(img_patch[i][j], new_shape)
                    data_img_stack = data_patch
                else:
                    data_patch = np.reshape(img_patch[i][j], new_shape)
                    data_img_stack = np.hstack((data_img_stack, data_patch))
        return data_img_stack, unreshape_patch
    
    def patch_fuse(self, data_patch, unreshape_patch):
        counter = 0
        for i in range(self.img_shape[0]//self.patch_size):
            # concatenate patches col by col
            for j in range(self.img_shape[1]//self.patch_size):
                if j==0:
                    img_patch = unreshape_patch(data_patch[:, counter])
                    img_patch_col = img_patch
                else:
                    img_patch = unreshape_patch(data_patch[:, counter])
                    img_patch_col = np.hstack((img_patch_col, img_patch))
                counter += 1
            # concatenate the genreated cols row by row
            if i==0: 
                img_row = img_patch_col
            else:
                img_row = np.vstack((img_row, img_patch_col))
        img = img_row
        return img

if __name__ == "__main__":
    num_img = 30
    patch_size = 12
    K = 628
    iter = 20
    load_path = './Train/'
    mission = K_SVD(num_img, load_path, patch_size)
    train_data = mission.load_new_data()
    final_dic = mission.train(train_data, K, iter)

    loss = [0.3, 0.5, 0.7]
    num_test_imgs_per_file = 7
    test_data_list = []
    for i in range(len(loss)):
        path = './TestData_new/Test_' + str(loss[i]) + 'loss/'
        test_data = mission.load_test_data(path, num_test_imgs_per_file)
        test_data_list.append(test_data)
    result_file = './patch_size=12/loss_'
    mission.test(final_dic, test_data_list, result_file, loss)
    


