from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Union, List
import torch
import numpy as np

class Evaluate(object):
    def __init__(self):
        self.precision_score = None
        self.recall_score = None
        self.f1_score = None

    def __call__(
            self,
            total_predicted_label: Union[List, np.ndarray, torch.Tensor],
            total_original_label: Union[List, np.ndarray, torch.Tensor]
            ) -> float:

        total_predicted_label = total_predicted_label.cpu().numpy()

        total_original_label = total_original_label.cpu().numpy()

        self.precision_score = precision_score(total_original_label, total_predicted_label, average='micro', labels=[1,2,3,4,5,6,7,8])

        self.recall_score = recall_score(total_original_label, total_predicted_label, average='micro', labels=[1,2,3,4,5,6,7,8])

        self.f1_score = f1_score(total_original_label, total_predicted_label, average='micro', labels=[1,2,3,4,5,6,7,8])

        return self.f1_score
