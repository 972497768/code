import os
import numpy as np
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable

######################################
#              Metirc                #
######################################
def np_divide(a, b):
    """
    Define the division function: a and b need to be the same size.
    When b is not 0, value=a/b. When b is 0, value=0.
    """
    return np.divide(a, b, out=np.zeros_like(b, dtype=np.float64), where=b != 0)


class SegmentationMetric(object):
    """
    genConfusionMatrix: The confusion matrix is a two-dimensional array of n*n.

    """
    def __init__(self, nClass):
        self.numClass = nClass
        # Define obfuscation matrix: create all-zero int64 type array of numClass*numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2, dtype=np.int64)

    def pixAccuracy(self):
        """Calculation accuracy"""
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc =(self.confusionMatrix.sum() - np.sum(self.confusionMatrix, axis=1) - np.sum(self.confusionMatrix, axis=0) + \
            np.diag(self.confusionMatrix) + np.diag(self.confusionMatrix) + np.diag(self.confusionMatrix)) / self.confusionMatrix.sum()
        return acc

    def precision(self):
        # precision = TP / (TP+FP
        precision = np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=1)
        return precision

    def recall(self):
        # recall = TP / (TP+FN)
        recall = np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=0)
        return recall


    def classPixelPrecision(self):
        """Calculation accuracy"""
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP precision
        classPre = np_divide(np.diag(self.confusionMatrix), self.confusionMatrix.sum(axis=1))
        # Returns a list value, e.g., [0.90, 0.80, 0.96], indicating the prediction accuracy for each of the categories 1 2 3
        return classPre 

    def classPixelRecall(self):
        """Calculating Recall"""
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FN recall rate
        classRec = np_divide(np.diag(self.confusionMatrix), self.confusionMatrix.sum(axis=0))
        return classRec

    def f1_score(self):
        """Calculating f1 scores"""
        # f1 = (2*precision*recall)/(precision+recall)
        precision = self.classPixelPrecision()
        recall = self.classPixelRecall()
        f1 = np_divide(2 * precision * recall, (precision + recall))
        # return sum(f1) / len(f1)
        return f1

    def meanPixelAccuracy(self):
        """Returns Mpa: average of several categories of accuracy, Cpa: list values: [Accuracy1, Accuracy2, Accuracy3]."""
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = TP / (TP + FP + TN + FN)
        # Cpa = np.diag(self.confusionMatrix) / self.confusionMatrix.sum()  #  np.diag is diagonal, find each diagonal element (TP)
        Cpa = (self.confusionMatrix.sum() - np.sum(self.confusionMatrix, axis=1) - np.sum(self.confusionMatrix, axis=0) + \
            np.diag(self.confusionMatrix) + np.diag(self.confusionMatrix) ) / self.confusionMatrix.sum()
        
        Mpa = np.nanmean(Cpa)  # Find the average Cpa for each category
        return Mpa, Cpa  # Returned is a list of values, e.g., [0.90, 0.80, 0.96], indicating the prediction accuracy for categories 1 2 3 each category

    def meanPixPrecision(self):
        """Calculate the average precision"""
        classPre = self.classPixelPrecision()
        meanPre = np.nanmean(classPre)
        return meanPre, classPre

    def meanPixRecall(self):
        """Calculate the average recall rate"""
        classRec = self.classPixelRecall()
        meanRec = np.nanmean(classRec)
        return meanRec, classRec

    def meanf1_score(self):
        """Calculate the average f1"""
        classf1 = self.f1_score()
        meanf1 = np.nanmean(classf1)
        return meanf1, classf1

    def meanIntersectionOverUnion(self):
        """Calculate the IntersectionOverUnion"""
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # TP:Take the value of the diagonal element and return the list.
        # Union = FN + FP - TP
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1 confuses the values of the rows of the matrix and returns a list; axis = 0 confuses the values of the columns of the matrix and returns a list.
        IoU = np_divide(intersection, union)  # Returns a list whose values are the IoUs for each category
        # Ciou = (intersection / np.maximum((1.0, union)))
        mIoU = np.nanmean(IoU)  # Find the average IoU for each category
        return mIoU, IoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        """Remove unlabeled pixels from images and predictions"""
        pred = imgPredict.flatten()
        label = imgLabel.flatten()
        confusionMatrix = confusion_matrix(pred, label, labels=list(range(self.numClass)))

        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        """Accumulation of the confusion matrix after removing unlabeled pixels"""
        assert imgPredict.shape == imgLabel.shape
        # if np.all(imgLabel) != 0:
        # print(self.genConfusionMatrix(imgPredict, imgLabel))
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
        # else:
        #     self.confusionMatrix += 0
        return self.confusionMatrix

    def reset(self):
        """Initializing the confusion matrix"""
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


def calc_confusionMatrix_results(segMetric):
    """Computing Confusion Matrix Results:Accuracy, Precision, Recall, f1 Score, IOU"""
    """
    [[Accuracy, Precision, Recall, F1, iou],
    [Accuracy, Precision, Recall, F1, iou],
    [Accuracy, Precision, Recall, F1, iou],
    [MAccuracy, MPrecision, MRecall, MF1, Miou]]
    """
    
    # Mpa, Cpa = segMetric.meanPixelAccuracy()

    Pa, Accuracy = segMetric.meanPixelAccuracy()
    P, Precision = segMetric.meanPixPrecision()
    R, Recall = segMetric.meanPixRecall()
    F, F1 = segMetric.meanf1_score()
    Miou, Ciou = segMetric.meanIntersectionOverUnion()


    # results =[Accuracy, Precision, Recall, F1, Ciou]
    results = [[round(Pa, 4), round(P,4), round(R,4), round(F,4), round(Miou, 4)]]
    for index, (accuracy, precision, recall, f1, iou) in enumerate(zip(Accuracy, Precision, Recall, F1, Ciou)):
        results.append([round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1, 4), round(iou, 4)])
    # results = [[Pa, P, R, F, Miou]]
    # for index, (accuracy, precision, recall, f1, iou) in enumerate(zip(Accuracy, Precision, Recall, F1, Ciou)):
    #     results.append([accuracy, precision, recall, f1, iou])
    return results