import pdb
from collections import defaultdict

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

from .eval_utils import nlp_evaluate

class Evaluator:
    _report_sections_ = ['findings','impression']
    def __init__(self,
        model,
        eval_dataloader=None,
        report=None,
        sentence_dataloader=None,
        topk=1,
        ) -> None:
        '''args:
        model: MedClipModel
        eval_dataloader: IUXRayDataset+IUXRayImageTextCollator
        sentecen_dataloader: DataLoader(IUXRaySentenceDataset)
        report: input report csv with columns reports, and index uid
        topk: retrieve top k images or sentences
        '''
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.sentence_dataloader = sentence_dataloader
        self.sentence_emb_bank = defaultdict() # {sentence texts: embeddings}
        self.report_bank = report
        if report is not None: self.update_report(report) # pd.Series({uid: report texts})
        self.image_emb_bank = defaultdict(list) # {uid: uid_list, embedding: image embeddings}
        self.topk = topk

    # def evaluate(self, eval_dataloader=None):
    #     self.model.eval()
    #     if self.eval_dataloader is None and eval_dataloader is not None:
    #         eval_dataloader = eval_dataloader
    #     else:
    #         eval_dataloader = self.eval_dataloader
        
    #     all_sent_embs = self.sentence_emb_bank['embedding']
    #     all_sent = self.sentence_emb_bank['sentence']
    #     pred_report = []
    #     gts_report = []
    #     for batch in tqdm(eval_dataloader, desc='evaluate encode image into embeddings'):
    #         '''batch: a dict with keys: pixel_values, input_ids, attention_mask, report(groundtruth)
    #         '''
    #         with torch.no_grad():
    #             img_embs = self.model.encode_image(batch['pixel_values']).cpu()
            
    #         report = batch['report']
    #         gts_report.extend(report)

    #         # compute cosine similarity
    #         sim_ts = torch.matmul(img_embs,all_sent_embs.T)
    #         topk_sent = torch.argsort(sim_ts,1)[:, -10:].numpy()            
    #         pred_report.extend(['. '.join(row) for row in all_sent[topk_sent]])
        
    #     pred_report = [[r] for r in pred_report]
    #     gts_report = [[r] for r in gts_report]

    #     report_indices = list(range(len(pred_report)))
    #     gts = dict(zip(report_indices, gts_report))
    #     res = dict(zip(report_indices, pred_report))
    #     score = nlp_evaluate(gts, res)
    #     return score

    def evaluate(self, eval_dataloader=None):
        self.model.eval()
        if self.eval_dataloader is None and eval_dataloader is not None:
            eval_dataloader = eval_dataloader
        else:
            eval_dataloader = self.eval_dataloader
        
        all_img_embs = torch.cat(self.image_emb_bank['embedding'], 0)
        all_uids = np.array(self.image_emb_bank['uid'])

        pred_report = []
        gts_report = []
        for batch in tqdm(eval_dataloader, desc='evaluate encode image into embeddings'):
            '''batch: a dict with keys: pixel_values, input_ids, attention_mask, report(groundtruth)
            '''
            with torch.no_grad():
                img_embs = self.model.encode_image(batch['pixel_values']).cpu()
            
            report = batch['report']
            gts_report.extend(report)

            # compute cosine similarity
            sim_ts = torch.matmul(img_embs, all_img_embs.T)
            topk_img = all_uids[torch.argsort(sim_ts,1)[:, -self.topk:].numpy()][:,::-1] # start from the most similar
            for img_idxs in topk_img:
                img_idxs = np.unique(img_idxs)
                pred_report.append(' '.join(self.report_bank.loc[img_idxs].tolist()))
        
        pred_report = [[r] for r in pred_report]
        gts_report = [[r] for r in gts_report]

        report_indices = list(range(len(pred_report)))
        gts = dict(zip(report_indices, gts_report))
        res = dict(zip(report_indices, pred_report))
        score = nlp_evaluate(gts, res)
        return score

    def update_memory(self, inputs):
        '''update image embeddings during training
        '''
        uid = np.array(inputs['uid'])
        self.image_emb_bank['uid'].extend(uid.astype(int))
        self.image_emb_bank['embedding'].append(inputs['image_embedding'].cpu())

    def update_sentence_memory(self):
        self.sentence_emb_bank = defaultdict(list)
        self.model.eval()
        # go through all stored sentences, encode by model and store their embeddings
        for batch_sent in tqdm(self.sentence_dataloader, desc='update sentence embeddings'):
            with torch.no_grad(): sent_embeds = self.model.encode_text(batch_sent['input_ids'], batch_sent['attention_mask'])
            self.sentence_emb_bank['sentence'].extend(batch_sent['sentence'])
            self.sentence_emb_bank['embedding'].append(sent_embeds.cpu())
        self.sentence_emb_bank['embedding'] = torch.cat(self.sentence_emb_bank['embedding'], axis=0)
        self.sentence_emb_bank['sentence'] = np.array(self.sentence_emb_bank['sentence'])
    
    def update_report(self, report):
        report = report.copy()
        report_agg = report.fillna('')[self._report_sections_].agg(' '.join,1)
        report_agg.index = report.uid.copy()
        report_agg.name = 'report'
        self.report_bank = report_agg

    def reset_memory(self):
        '''delete the saved image embeddings
        '''
        self.image_emb_bank = defaultdict(list)
    
