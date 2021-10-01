#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
plt.style.use('homework')

import os
from os import error, name, path
from pathlib import Path
import pickle

from numpy.lib.npyio import load


def load_training_data(data_dir):
    """loads all M images size `im_shape` in directory named `data_dir` into a numpy array shape (M, N)

    Args:
        data_dir (Path): Directory to load images from

    Returns:
        X: images array in shape (M, N)  
        ids: labels for each image X[i] for i=0:M  
        im_shape: image shape used for image plotting
    """   
    im_addr = os.listdir(data_dir)

    M = len(im_addr)
    X = []
    ids = []
    for i in range(M):
        im = Image.open(path.join(data_dir, im_addr[i]))
        if i == 0:
            im_shape = im.size
            im_shape = (im_shape[1], im_shape[0])
        x = np.array(im.getdata())
        X.append(x)
        fid = im_addr[i].replace('.pgm', '').split('_')
        ids.append(fid[0])
    X = np.array(X)
    N = X.shape[1]

    assert X.shape == (M, N)
    return X, np.array(ids), im_shape


def load_params(load_file):
    """Load Model parameters from a pickle file

    Args:
        load_file (Path): path to pickle

    Returns:
        ndarray: projected images from training set
        list: list of training labels
        PCA: pca object with model parameters 
    """
    with load_file.open('rb') as f:
        Omega, labels, face_pca = pickle.load(f)
    return Omega, labels, face_pca

class PCA:
    def __init__(self, X, original_shape):
        """Creates svd decomposition and stores eigenvalues and U eigenvectors

        Args:
            X ([M, N] ndarray): training data
            original_shape (tuple): image shape for plotting
        """
        self.im_shape = original_shape
        self.train_pca(X)

    def __repr__(self) -> str:
        f"PCA size {self.N}"

    def train_pca(self, X):
        """
        Input X: training data. numpy array shape (M, N)
            M = sample count
            N = feature count
        Effects: sets the following attributes
            eigenvalues: N PCA eigenvalues in decreasing order
            U: PCA eigenvectors where U[i] corresponds to Lambda[i]
            x_mean: average feature value of all samples
            information_k: the cumulative amount of "information" each eigenvector adds
        """
        M, N = X.shape
        self.M = M
        self.N = N

        self.x_mean = X.mean(axis=0)     # step 1
        assert self.x_mean.shape[0] == N
        var = X.std(axis=0)
        A = (X - self.x_mean)/var        # Center X
        assert A.shape == (M, N)
        # Rows of vh[i] are unit length eigenvectors corresponding to singular value s[i]
        u, s, vh = np.linalg.svd(A)     
        self.eigenvalues = np.square(s)
        self.U = vh

        self.K = N
        self.information_k = np.cumsum(self.eigenvalues) / self.eigenvalues.sum()
        # return Lam, U, x_mean

    def plot_mean(self):
        """
        Plots the training sample mean as an image 
        """
        plt.imshow(self.x_mean.reshape(self.im_shape), cmap="gray")
        plt.axis('off')
        # plt.title(r"Mean face: $\bar x$")
        plt.show()

    def plot_eigenvectors(self):
        """
        Plots the first 10 and last eigenvectors as images
        """
        aspect = self.im_shape[0] / self.im_shape[1]
        fig, axs = plt.subplots(2, 10, figsize=(10, 2))

        for i in range(10):
            eigenvector = self.U[i].reshape(self.im_shape)
            axs[0, i].imshow(eigenvector, cmap='gray')
            axs[0, i].axis('off')

            eigenvector = self.U[-10+i].reshape(self.im_shape)
            axs[1, i].imshow(eigenvector, cmap='gray')
            axs[1, i].axis('off')
        
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        plt.show();

    def reconstruct(self, omega, K=None):
        """
        Reconstruct an image using first K eigenvalues
        """
        if K is None:
            K = self.K
        yu = omega.T @ self.U[:K]
        xhat = yu + self.x_mean
        return xhat


    def reconstruction_error(self, my_im, thresholds, K=None, plot=True):
        """
        Projects `my_im` onto the `K` PCA subspace and computes the reconstruction error for a given set of thresholds.
        Input:
            my_im: numpy array shape (N,)
            thresholds: list of information thresholds to check
            K: number of eigenvectors to use
        Returns:
            xhat: the transformed version of `my_im`
            error: the reconstruction error || xhat - my_im ||
        """
        if K is None:
            K = self.K
        
        if plot:
            fig, axs = plt.subplots(1, 1 + len(thresholds))
            axs[0].imshow(my_im.reshape(self.im_shape), cmap='gray')
            axs[0].axis('off')
            axs[0].set_title("Original")

        
        for i, thresh in enumerate(thresholds):
            self.set_threshold(thresh)
            omega = self.transform(my_im)
            xhat = self.reconstruct(omega)
            error = np.linalg.norm(my_im - xhat)
            print(f"Reconstruction error: {error:.2f}")
            
            if plot:
                axs[i+1].imshow(xhat.reshape(self.im_shape), cmap='gray')
                axs[i+1].axis('off')
                axs[i+1].set_title(f"{thresh:.1%}pct")
                # axs[i+1].set_title(f"Err: {error:.0f}")
            
        plt.show()
        return xhat, error


    def set_threshold(self, thres):
        """
        How many dimensions K does is take to account for thres percentage of information?
        Sets self.K which is used in the `transform` method
        """
        threshold_K = np.argmax(self.information_k > thres)
        print(f"{thres:.0%} of the variance is within {threshold_K} largest eigenvalues.")
        self.K = threshold_K
        return threshold_K


    def transform(self, X, K=None):
        if K is None:
            K = self.K
        # U.shape = (K, N)
        omega = self.U[:K] @ (X - self.x_mean).T 
        return omega.T


    def classify(self, X, Omega, test_labels, training_labels):
        """
        Given a test image set `X`, project onto PCA and find a match in `Omega`.
        Choose the top N matches having the lowest error.
        Input:
            X: test image shape(M2, N)
            Omega: projected training images, shape(M, N)
            test_labels: labels corresponding to test images
            training_labels: labels corresponding to training images
        Returns:
            curve: the cumulative matching curve for correct classifications given top N images
            matches: 3 matches [test_idx, train_idx] for N=1
            mismatches: 3 mismatches [test_idx, train_idx] for N=1
        """
        test_Omega = self.transform(X)
        cmcN = 50
        cmcCurve = np.zeros((test_Omega.shape[0], cmcN))
        matches = []
        mismatches = []

        for i, t_omega in enumerate(test_Omega):
            # if (i / test_Omega.shape[0]) % 10:
            #     print(i / test_Omega.shape[0] * 100, sep=", ")

            # compute Mahalanobis distance to training data
            errors = np.sum(np.square(t_omega - Omega[:, :self.K]) / self.eigenvalues[:self.K], axis=1)
            errors_norm = np.linalg.norm(t_omega - Omega[:, :self.K], axis=1)  # Euclidean distance
            min_errors = np.argsort(errors)[:cmcN]  # get top 50 matches
            min_errors_eu = np.argsort(errors_norm)[:cmcN]

            # try:
            #     if min_errors[0] != min_errors_eu[0]:
            #         inMah = test_labels[i] in [training_labels[j] for j in min_errors]
            #         inEu = test_labels[i] in [training_labels[j] for j in min_errors_eu]
            #         # print(f"Test in first 50 Mahalanobis: {inMah}. In euclidean: {inEu}")
            #         if inMah is False and inEu is True:
            #             raise ValueError
            # except ValueError:
            #     pass


            for n in range(1, cmcN+1):
                n_train_idx = min_errors[:n]
                n_train_labels = [training_labels[j] for j in n_train_idx]

                if test_labels[i] in n_train_labels:
                    cmcCurve[i, n-1] = 1
                    if n == 1 and len(matches) < 3:
                        matches.append([i, n_train_idx[0]])
                elif n == 1 and len(mismatches) < 3:
                    mismatches.append([i, n_train_idx[0]])
                
        return cmcCurve.mean(axis=0), matches, mismatches
                    


def train_pca(data_dir, save_path):
    """
    Train PCA on `data_dir` dataset
    Save projections, and pca object in pickle file `save_name`
    """

    with save_path.open('wb') as f:
        print(f"Training PCA on {data_dir}")
        X, ids, im_shape = load_training_data(data_dir)
        pca_tool = PCA(X, im_shape)
        print("Projecting images onto PCA.")
        Omega = pca_tool.transform(X)
        print("Saving training parameters to", save_path)
        pickle.dump([Omega, ids, pca_tool], f, protocol=pickle.HIGHEST_PROTOCOL)

    return Omega, ids, pca_tool


def test_pca(training_param_file, test_dir):
    """
    Read projections, and PCA object from `training_param_file`.
    Load test images from `test_dir`.
    Compare test images with 
    """
    Omega, labels, face_pca = load_params(training_param_file)

    assert np.all(np.diff(face_pca.eigenvalues) <= 0), "Eigenvalues are not in decreasing order"

    im_test, test_labels, im_shape = load_training_data(test_dir)

    # Face recognition steps
    fig_curve, ax_curve = plt.subplots()
    for thres in (.8, .9, .95):
        face_pca.set_threshold(thres)  # set information threshold at 80%
        cmcCurve, matches, mismatches = face_pca.classify(im_test, Omega, test_labels, labels)

        # Show three matching and mismatched queries
        if thres == .8:
            fig_matches, m_axs = plt.subplots(3, 4)
            m_axs[0,0].set_title("Matched\nQuery")
            m_axs[0,1].set_title("Matched\nSample")
            m_axs[0,2].set_title("Mismatched\nQuery")
            m_axs[0,3].set_title("Mismatched\nSample")

            for i, match in enumerate(matches):
                test_im = im_test[match[0]].reshape(im_shape)
                train_im = face_pca.reconstruct(Omega[match[1]], K=face_pca.N).reshape(im_shape)
                m_axs[i,0].imshow(test_im)
                m_axs[i,0].axis('off')
                m_axs[i,1].imshow(train_im)
                m_axs[i,1].axis('off')
            
            for i, match in enumerate(mismatches):
                test_im = im_test[match[0]].reshape(im_shape)
                train_im = face_pca.reconstruct(Omega[match[1]], K=face_pca.N).reshape(im_shape)
                m_axs[i,2].imshow(test_im)
                m_axs[i,2].axis('off')
                m_axs[i,3].imshow(train_im)
                m_axs[i,3].axis('off')
        # Plot the CMC curve for this information threshold
        ax_curve.plot(range(1, 51), cmcCurve, label=thres)

    ax_curve.set_xlabel("Top N Matches")
    ax_curve.set_ylabel("Chance of correct classification")
    ax_curve.set_title("Cumulative Matching Curve")
    ax_curve.set_ylim(0, 1.1)
    ax_curve.set_xlim(1, 50)
    ax_curve.set_xticks([1, 10, 20, 30, 40, 50])
    ax_curve.legend()
    plt.show()


def test_reconstruction(im_dir, param_pickle):
    X, ids, im_shape = load_training_data(im_dir)
    Omega, labels, face_pca = load_params(param_pickle)
    # face_pca.plot_eigenvectors()
    i = 100
    thresholds = (.1, .5, .8, .9, .95, .99, .999)
    face_pca.reconstruction_error(X[i], thresholds, plot=True)


def plot_information(train_H_params, train_L_params):
    _, _, high_pca = load_params(train_H_params)
    _, _, low_pca = load_params(train_L_params)

    for thresh in (.8, .9, .95, .99):
        low_pca.set_threshold(thresh)
        print(f"Low Model pct: {low_pca.K / low_pca.N:.1%}")
        high_pca.set_threshold(thresh)
        print(f"High Model pct: {high_pca.K / high_pca.N:.1%}")

    l_eigenvalues = low_pca.eigenvalues
    h_eigenvalues = high_pca.eigenvalues

    l_information = np.cumsum(l_eigenvalues) / np.sum(l_eigenvalues)
    h_information = np.cumsum(h_eigenvalues) / np.sum(h_eigenvalues)
    l_range = np.array(range(1, len(l_eigenvalues)+1)) / len(l_eigenvalues)
    h_range = np.array(range(1, len(h_eigenvalues)+1)) / len(h_eigenvalues)

    fig, ax = plt.subplots()
    ax.plot(l_range, l_information, label="Low Resolution")
    ax.plot(h_range, h_information, label="High Resolution")
    ax.set_xlabel("Percentage of Model Size")
    ax.set_ylabel("Information")
    ax.legend()

    fig, ax = plt.subplots()
    x_range = 200
    ax.plot(range(1, x_range+1), l_information[:x_range], label="Low Resolution")
    ax.plot(range(1, x_range+1), h_information[:x_range], label="High Resolution")
    ax.set_xlabel("K")
    ax.set_ylabel("Information")
    ax.legend()

    plt.show()


def main():
    root = Path(__file__).parent
    train_H = root / "Faces/fa_H"
    train_L = root / "Faces/fa_L"

    # Params = [Omega, labels, face_pca]
    train_H_params = root / "results/fa_H_trained.pkl"
    train_L_params = root / "results/fa_L_trained.pkl"

    test_H = root / "Faces/fb_H"
    test_L = root / "Faces/fb_L"

    # --- MAIN FUNCTIONS ---
    # Uncomment below to run

    # train_pca(train_L, train_L_params)
    # test_pca(train_H_params, test_H)
    # test_reconstruction(train_H, train_H_params)
    plot_information(train_H_params, train_L_params)

    
if __name__ == '__main__':
    main()