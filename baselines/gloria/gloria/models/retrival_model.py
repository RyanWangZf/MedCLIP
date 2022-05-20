import torch
import numpy as np
from .. import utils
from ..loss.gloria_loss import attention_fn, cosine_similarity
from sklearn import metrics


class Retriver:
    def __init__(
        self, ckpt_path, targets=None, target_classes=None, device=None, top_k=5
    ):

        # create device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # load gloria model
        self.gloria = utils.load_gloria(ckpt_path).to(device=self.device)
        self.top_k = top_k

        # create targets
        self.targets = self._process_targets(targets)
        self.targets_classes = np.array(target_classes)

    def _process_targets(self, target):

        text_tensors = self.gloria.process_text(target)
        caption_ids = (
            torch.stack([x["input_ids"] for x in text_tensors])
            .squeeze()
            .to(device=self.device)
        )
        attention_mask = (
            torch.stack([x["attention_mask"] for x in text_tensors])
            .squeeze()
            .to(device=self.device)
        )
        token_type_ids = (
            torch.stack([x["token_type_ids"] for x in text_tensors])
            .squeeze()
            .to(device=self.device)
        )

        with torch.no_grad():
            text_emb_l, text_emb_g, sents = self.gloria.text_encoder_forward(
                caption_ids, attention_mask, token_type_ids
            )

        # get cap_lens
        self.cap_lens = [
            len([w for w in sent if not w.startswith("[")]) for sent in sents
        ]

        # remove [CLS] token
        text_emb_l = text_emb_l[:, :, 1:]

        return {
            "global_embeddings": text_emb_g.detach().cpu(),
            "local_embeddings": text_emb_l.detach().cpu(),
            "processed_input": target,
        }

    def _process_source(self, img):
        with torch.no_grad():
            imgs = self.gloria.process_img(img)
            img_tensors = torch.stack(imgs).to(device=self.device)
            img_emb_l, img_emb_g = self.gloria.image_encoder_forward(img_tensors)

        del img_tensors

        return {
            "global_embeddings": img_emb_g.detach().cpu(),
            "local_embeddings": img_emb_l.detach().cpu(),
            "processed_input": imgs,
        }

    def retrieve(self, source, similarity_type="both"):

        if similarity_type not in ["both", "local", "global"]:
            raise Exception(
                "similarity_type must be one of ['both', 'local', 'global']"
            )

        # process source
        self.source = self._process_source(source)

        # compute similarities
        local_similarities = self._compute_local_similarity(
            self.source["local_embeddings"],
            self.targets["local_embeddings"],
            self.cap_lens,
            self.gloria.temp1,
            self.gloria.temp2,
            self.gloria.temp3,
        )
        # agg='mean')
        # local_similarities = local_similarities.transpose(0, 1)

        global_similarities = metrics.pairwise.cosine_similarity(
            self.source["global_embeddings"], self.targets["global_embeddings"]
        )
        global_similarities = global_similarities[0]

        if similarity_type == "local":
            similarities = local_similarities
        elif similarity_type == "global":
            similarities = global_similarities
        else:
            norm = lambda x: (x - x.mean(axis=0)) / (x.std(axis=0))
            similarities = np.stack(
                [norm(local_similarities), norm(global_similarities)]
            )
            # similarities = np.stack([local_similarities, global_similarities])
            similarities = similarities.mean(axis=0)

        sorted_idx = np.argsort(similarities)[::-1][: self.top_k]

        if self.targets_classes is not None:
            retrived_cls = self.targets_classes[sorted_idx]
        else:
            retrived_cls = None

        return np.array(self.targets["processed_input"])[sorted_idx], retrived_cls

    def _compute_local_similarity(
        self,
        img_features,
        words_emb,
        cap_lens,
        temp1=4.0,
        temp2=5.0,
        temp3=10.0,
        agg="sum",
    ):

        batch_size = words_emb.shape[0]

        similarities = []
        for i in range(batch_size):

            words_num = cap_lens[i]
            word = words_emb[i, :, 1 : words_num + 1].unsqueeze(0).contiguous()
            context = img_features

            weiContext, _ = attention_fn(word, context, temp1)

            word = word.transpose(1, 2).contiguous()
            weiContext = weiContext.transpose(1, 2).contiguous()

            word = word.squeeze()
            weiContext = weiContext.squeeze()

            row_sim = cosine_similarity(word, weiContext)

            row_sim.mul_(temp2).exp_()
            if agg == "sum":
                row_sim = row_sim.sum()
            else:
                row_sim = row_sim.mean()
            row_sim = torch.log(row_sim)

            similarities.append(row_sim.item())

        return np.array(similarities) * temp3
