from catboost import CatBoostClassifier
from sklearn.utils import safe_mask
import numpy as np

class CatBoostClassifierExtended:
    def __init__(self, cats, *args, **kwargs):
        self.cats = cats
        self.classifier = CatBoostClassifier(*args, **kwargs)
        
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifier.fit(X, y, self.cats, verbose = False)
    
    def predict(self, X):
        return self.classifier.predict(X)
        
    def predict_proba(self, X):
        return self.classifier.predict_proba(X)
        


class SelfTrainingClassifierExtended:

    def __init__(
        self,
        base_estimator,
        threshold=0.75,
        criterion="threshold",
        k_best=10,
        max_iter=10,
        verbose=False,
    ):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.criterion = criterion
        self.k_best = k_best
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y):

        self.base_estimator_ = self.base_estimator

        if y.dtype.kind in ["U", "S"]:
            raise ValueError(
                "y has dtype string. If you wish to predict on "
                "string targets, use dtype object, and use -1"
                " as the label for unlabeled samples."
            )

        has_label = y != -1

        if np.all(has_label):
            warnings.warn("y contains no unlabeled samples", UserWarning)

        if self.criterion == "k_best" and (
            self.k_best > X.shape[0] - np.sum(has_label)
        ):
            warnings.warn(
                "k_best is larger than the amount of unlabeled "
                "samples. All unlabeled samples will be labeled in "
                "the first iteration",
                UserWarning,
            )

        self.transduction_ = np.copy(y)
        self.labeled_iter_ = np.full_like(y, -1)
        self.labeled_iter_[has_label] = 0

        self.n_iter_ = 0

        while not np.all(has_label) and (
            self.max_iter is None or self.n_iter_ < self.max_iter
        ):
            self.n_iter_ += 1
            self.base_estimator_.fit(
                X[safe_mask(X, has_label)], self.transduction_[has_label]
            )

            # Predict on the unlabeled samples
            prob = self.base_estimator_.predict_proba(X[safe_mask(X, ~has_label)])
            pred = self.base_estimator_.classes_[np.argmax(prob, axis=1)]
            max_proba = np.max(prob, axis=1)

            # Select new labeled samples
            if self.criterion == "threshold":
                selected = max_proba > self.threshold
            else:
                n_to_select = min(self.k_best, max_proba.shape[0])
                if n_to_select == max_proba.shape[0]:
                    selected = np.ones_like(max_proba, dtype=bool)
                else:
                    # NB these are indices, not a mask
                    selected = np.argpartition(-max_proba, n_to_select)[:n_to_select]

            # Map selected indices into original array
            selected_full = np.nonzero(~has_label)[0][selected]

            # Add newly labeled confident predictions to the dataset
            self.transduction_[selected_full] = pred[selected]
            has_label[selected_full] = True
            self.labeled_iter_[selected_full] = self.n_iter_

            if selected_full.shape[0] == 0:
                # no changed labels
                self.termination_condition_ = "no_change"
                break

            if self.verbose:
                print(
                    f"End of iteration {self.n_iter_},"
                    f" added {selected_full.shape[0]} new labels."
                )

        if self.n_iter_ == self.max_iter:
            self.termination_condition_ = "max_iter"
        if np.all(has_label):
            self.termination_condition_ = "all_labeled"

        self.base_estimator_.fit(
            X[safe_mask(X, has_label)], self.transduction_[has_label]
        )
        self.classes_ = self.base_estimator_.classes_
        return self

    def predict(self, X):
        return self.base_estimator_.predict(X)

    def predict_proba(self, X):
        return self.base_estimator_.predict_proba(X)

    def score(self, X, y):
        return self.base_estimator_.score(X, y)

