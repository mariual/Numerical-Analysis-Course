# Maryam Alipour | 9612037

# pip install opencv-python, numpy, matplotlib, sklearn
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class FeatureExtraction:

    def choose(self,function,resizeImage):
        if function == "SVD":
            return self.svd(resizeImage)
        elif function == "PCA":
            return self.pca(resizeImage)
        else:
            return self.svd(resizeImage)


    def svd(self,resizeImage):
        union =TruncatedSVD(n_components=2)
        fit = union.fit(resizeImage)
        fit = union.transform(resizeImage)
        im_bw=[]
        for field, possible_values in fit:
            im_bw.append(field)

        return im_bw
    

    def pca(self, resizeImage):
        pca = PCA(n_components=5)
        pca.fit(resizeImage)
        X = pca.transform(resizeImage)
        flatten_feature = list(X.flatten())

        return flatten_feature



class Classifer:

    def choose(self, algorithm, x_train, x_test, y_train, y_test, example):
        if algorithm == "KNN":
            self.knn(x_train, x_test, y_train, y_test, example)
        elif algorithm == "Bayes":
            self.bayes(x_train, x_test, y_train, y_test, example)
        elif algorithm == "LinearSVC":
            self.LinearSVC(x_train, x_test, y_train, y_test, example)
        else:
            self.knn(x_train, x_test, y_train, y_test, example)


    def knn(self, x_train, x_test, y_train, y_test, example):
        error = []
        best_k = dict()

        # Calculating error for K values between 1 and 20
        for i in range(1, 20):
            knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
            knn.fit(x_train, y_train)
            pred_i = knn.predict(x_test)
            error.append(np.mean(pred_i != y_test))
            best_k[i] = np.mean(pred_i != y_test)

        best_k = sorted(best_k.items(), key=lambda k: k[1])[0][0]
        classifier = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        print("--------- Accuracy: ", accuracy_score(y_test, y_pred))
        print("--------- Your Input Prediction: ", classifier.predict(np.array(example).reshape(1, -1))[0][-1])


        plt.figure(figsize=(12, 6))
        plt.plot(range(1, 20), error, color='red', linestyle='dashdot', marker='o',
                 markerfacecolor='green', markersize=10)
        plt.title('Error Rate K Value')
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')
        plt.show()


    def bayes(self, x_train, x_test, y_train, y_test, example):
        # Create a Gaussian Classifier
        gnb = GaussianNB()
        # Train the model using the training sets
        gnb.fit(x_train, y_train)
        # Predict the response for test dataset
        y_pred = gnb.predict(x_test)
        print("--------- Accuracy: ", accuracy_score(y_test, y_pred))
        print("--------- Your Input Prediction: ", gnb.predict(np.array(example).reshape(1, -1))[0][-1])



    def LinearSVC  (self, X_train, X_test, y_train, y_test, example):
        clf = SVC(gamma='auto')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("--------- Accuracy: ", accuracy_score(y_test, y_pred))
        print("--------- Your Input Prediction: ", clf.predict(np.array(example).reshape(1, -1))[0][-1])





class pre:

    def SelectClassfierAndFeatureExtractor(self):
        print("Please select number of classifier:")  # Quadratic Discriminant Analysis
        print("1.KNN    2.Bayes     3.LinearSVC\n")
        classifier = int(input())
        if classifier == 1:
            classifier = "KNN"
        elif classifier == 2:
            classifier = "Bayes"
        elif classifier == 3:
            classifier = "LinearSVC"
        else:
            classifier = "KNN"

        print("\nPlease select number of feature extractor:")
        print("1.SVD     2.PCA \n")
        featureExtractor = int(input())
        if featureExtractor == 1:
            featureExtractor = "SVD"
        elif featureExtractor == 2:
            featureExtractor = "PCA"
        else:
            featureExtractor = "SVD"

        print("--------- Classifier: ", classifier)
        print("--------- Feature Extractor: ", featureExtractor)

        return classifier, featureExtractor

    def preprocess_example(self, example_path, feature_selector, size):
        img = cv2.imread(example_path,cv2.IMREAD_GRAYSCALE)
        # Resize image for get better feature
        resizeImg = cv2.resize(img,(size,size), interpolation=cv2.INTER_AREA)
        # Thredshold for 128 is 0 and 255 is
        _, IMG = cv2.threshold(resizeImg, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        data = FeatureExtraction().choose(feature_selector,IMG)
        return data

        



def ExtractFeatureAndBuildDataset(extract_feature="SVD",size=50):
    DATA = []
    Labels = []
    for path, subdirs, files in os.walk("/Users/maryamalipour/Downloads/NumericalAnalysisProject/persian_digit/"):
        for name in files:
            imagePath = path + '/' + name
            # Read image as gray scale
            img = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
            # Resize image for get better feature
            resizeImg = cv2.resize(img,(size,size), interpolation=cv2.INTER_AREA)
            # Thredshold for 128 is 0 and 255 is
            _, IMG = cv2.threshold(resizeImg, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # Append extract featured DATA
            DATA.append(FeatureExtraction().choose(extract_feature,IMG))
            # Label of each image
            Labels.append(path[14:])

    return DATA,Labels


def main():
    print("-------------- Maryam Alipour | 9612037 ----------------")
    print("---------- OCR(Optical character recognition) ----------")
    print("-------------  Persian Digit Recognition  --------------")

    print("\n")
    # Size of resizing image
    size = 50

    # Select classfier and feature extractor
    classifier , feature_selector = pre().SelectClassfierAndFeatureExtractor()
    # Preprocess example input
    example_path = "/Users/maryamalipour/Downloads/NumericalAnalysisProject/example_image.jpg"  
    example = pre().preprocess_example(example_path, feature_selector, size)
    # Extract feature of image
    DATA,Labels = ExtractFeatureAndBuildDataset(feature_selector,size)
    # Set train and test data
    X_train, X_test, y_train, y_test = train_test_split(DATA, Labels, test_size=0.10)
    cls = Classifer().choose(classifier, X_train, X_test, y_train, y_test, example)
    print("\n----------------------- FINISHED -----------------------")


if __name__=="__main__":
    main()
    