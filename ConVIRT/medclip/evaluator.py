import pandas as pd
import numpy as np
import torch

class Evaluator:
    '''do evaluation on chexpert5x200 zero-shot classification
    '''
    def __init__(self,
        medclip_clf,
        eval_dataloader=None,
        ) -> None:
        self.clf = medclip_clf
        self.eval_dataloader = eval_dataloader

    def evaluate(self, eval_dataloader=None):
        self.clf.eval()
        if self.eval_dataloader is None and eval_dataloader is not None: eval_dataloader = eval_dataloader
        else: eval_dataloader = self.eval_dataloader
        pred_list = []
        label_list = []
        for data in eval_dataloader:
            with torch.no_grad():
                outputs = self.clf(**data)
                pred = outputs['logits']
            pred_list.append(pred)
            label_list.append(data['labels'])
        pred_list = torch.cat(pred_list, 0)
        label_list= np.concatenate(label_list, 0)

        pred_df = pd.DataFrame(
            pred_list.cpu().numpy(),
            columns = outputs['class_names'],
        )

        # evaluate accuracy
        from sklearn.preprocessing import OrdinalEncoder
        class_names = outputs['class_names']
        enc = OrdinalEncoder(categories=[outputs['class_names']])
        labels = enc.fit_transform(label_list[:,None]).flatten()
        acc = (pred_df.to_numpy().argmax(1) == labels).mean()
        return {'acc':acc}