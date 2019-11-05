import math
import pickle
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif

class ViolaJones:
    def __init__(self, T = 10):

        self.T = T
        self.alphas = []
        self.clfs = []

    def training(self, training, pos_num, neg_num):

        weights = np.zeros(len(training))
        training_data = []
        for x in range(len(training)):
            training_data.append((integral_image(training[x][0]), training[x][1]))
            if training[x][1] == 1:
                weights[x] = 1.0 / (2 * pos_num)
            else:
                weights[x] = 1.0 / (2 * neg_num)

        features = self.feature_building(training_data[0][0].shape)
        X, y = self.feature_applying(features, training_data)
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
        X = X[indices]
        features = features[indices]

        for t in range(self.T):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.weak_training(X, y, features, weights)
            clf, error, accuracy = self.best_selecting(weak_classifiers, weights, training_data)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)

    def weak_training(self, X, y, features, weights):

        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        classifiers = []
        total_features = X.shape[0]
        for index, feature in enumerate(X):
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w

            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers

    def feature_building(self, image_shape):

        height, width = image_shape
        features = []
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i+w, j, w, h)
                        if i + 2 * w < width:
                            features.append(([right], [immediate]))

                        bottom = RectangleRegion(i, j+h, w, h)
                        if j + 2 * h < height:
                            features.append(([immediate], [bottom]))

                        right_2 = RectangleRegion(i+2*w, j, w, h)
                        if i + 3 * w < width:
                            features.append(([right], [right_2, immediate]))

                        bottom_2 = RectangleRegion(i, j+2*h, w, h)
                        if j + 3 * h < height:
                            features.append(([bottom], [bottom_2, immediate]))

                        bottom_right = RectangleRegion(i+w, j+h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right, bottom], [immediate, bottom_right]))
                        j += 1
                    i += 1
        return np.array(features)

    def best_selecting(self, classifiers, weights, training_data):

        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.category(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(training_data)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy

    def feature_applying(self, features, training_data):

        X = np.zeros((len(features), len(training_data)))
        Y = np.array(list(map(lambda data: data[1], training_data)))
        i = 0
        for positive_regions, negative_regions in features:
            feature = lambda integral_img: sum([pos.feature_computing(integral_img) for pos in positive_regions]) - sum([neg.feature_computing(integral_img) for neg in negative_regions])
            X[i] = list(map(lambda data: feature(data[0]), training_data))
            i += 1
        return X, Y

    def category(self, image):

        total = 0
        ii = integral_image(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.category(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0

    def save(self, filename):

        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):

        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)

class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):

        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity

    def category(self, x):

        feature = lambda integral_img: sum([pos.feature_computing(integral_img) for pos in self.positive_regions]) - sum([neg.feature_computing(integral_img) for neg in self.negative_regions])
        return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0

    def __str__(self):
        return "Weak Clf (threshold=%d, polarity=%d, %s, %s" % (self.threshold, self.polarity, str(self.positive_regions), str(self.negative_regions))

class RectangleRegion:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.width = w
        self.height = h

    def feature_computing(self, integral_img):
        return integral_img[self.y+self.height][self.x+self.width] + integral_img[self.y][self.x] - (integral_img[self.y+self.height][self.x]+integral_img[self.y][self.x+self.width])

    def __str__(self):
        return "(x= %d, y= %d, width= %d, height= %d)" % (self.x, self.y, self.width, self.height)
    def __repr__(self):
        return "RectangleRegion(%d, %d, %d, %d)" % (self.x, self.y, self.width, self.height)

def integral_image(img):
    integral_img = np.zeros(img.shape)
    sp = np.zeros(img.shape)
    for y in range(len(img)):
        for x in range(len(img[y])):
            sp[y][x] = sp[y-1][x] + img[y][x] if y-1 >= 0 else img[y][x]
            integral_img[y][x] = integral_img[y][x-1]+sp[y][x] if x-1 >= 0 else sp[y][x]
    return integral_img
