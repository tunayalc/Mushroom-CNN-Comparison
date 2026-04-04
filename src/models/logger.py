import json
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

class Logger:
    def __init__(self, log_dir='logs', class_names=None):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        self.train_log = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "epoch_times": []
        }

        self.test_log = {
            "metrics": {},
            "confusion_matrix": [],
            "roc_data": {
                "y_true": [],
                "y_probs": []
            },
            "test_time": None,
            "class_names": class_names if class_names else []
        }

    def log_epoch(self, train_loss, val_loss, train_acc, val_acc, epoch_time):
        self.train_log["train_loss"].append(train_loss)
        self.train_log["val_loss"].append(val_loss)
        self.train_log["train_accuracy"].append(train_acc)
        self.train_log["val_accuracy"].append(val_acc)
        self.train_log["epoch_times"].append(epoch_time)

    def save_train_log(self, filename='train_log.json'):
        path = os.path.join(self.log_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.train_log, f, indent=4)

    def log_test(self, y_true, y_pred, y_probs, class_names=None):
        start_time = time.time()

        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)

        num_classes = len(np.unique(y_true))
        specificity_list = []
        for i in range(num_classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            specificity = TN / (TN + FP + 1e-7)
            specificity_list.append(specificity)
        specificity = np.mean(specificity_list)

        gmean = np.sqrt(recall * specificity)

        try:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except:
            auc = None

        self.test_log["metrics"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "mcc": mcc,
            "gmean": gmean,
            "auc": auc
        }

        self.test_log["confusion_matrix"] = cm.tolist()
        self.test_log["roc_data"]["y_true"] = y_true
        self.test_log["roc_data"]["y_probs"] = y_probs
        self.test_log["test_time"] = time.time() - start_time

        if class_names:
            self.test_log["class_names"] = class_names

    def save_test_log(self, filename='test_log.json'):
        import numpy as np

        def convert(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_log = json.loads(json.dumps(self.test_log, default=convert))

        path = os.path.join(self.log_dir, filename)
        with open(path, 'w') as f:
            json.dump(serializable_log, f, indent=4)