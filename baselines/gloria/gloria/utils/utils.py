"""Adapted from: https://github.com/mrlibw/ControlGAN"""

import numpy as np
import torch
import torch.nn as nn
import skimage.transform


from PIL import Image, ImageDraw, ImageFont


def normalize(similarities, method="norm"):

    if method == "norm":
        return (similarities - similarities.mean(axis=0)) / (similarities.std(axis=0))
    elif method == "standardize":
        return (similarities - similarities.min(axis=0)) / (
            similarities.max(axis=0) - similarities.min(axis=0)
        )
    else:
        raise Exception("normalizing method not implemented")


# For visualization ################################################
COLOR_DIC = {
    0: [128, 64, 128],
    1: [244, 35, 232],
    2: [70, 70, 70],
    3: [102, 102, 156],
    4: [190, 153, 153],
    5: [153, 153, 153],
    6: [250, 170, 30],
    7: [220, 220, 0],
    8: [107, 142, 35],
    9: [152, 251, 152],
    10: [70, 130, 180],
    11: [220, 20, 60],
    12: [255, 0, 0],
    13: [0, 0, 142],
    14: [119, 11, 32],
    15: [0, 60, 100],
    16: [0, 80, 100],
    17: [0, 0, 230],
    18: [0, 0, 70],
    19: [0, 0, 0],
    20: [128, 64, 128],
    21: [244, 35, 232],
    22: [70, 70, 70],
    23: [102, 102, 156],
    24: [190, 153, 153],
    25: [153, 153, 153],
    26: [250, 170, 30],
    27: [220, 220, 0],
    28: [107, 142, 35],
    29: [152, 251, 152],
    30: [70, 130, 180],
    31: [220, 20, 60],
    32: [255, 0, 0],
    33: [0, 0, 142],
    34: [119, 11, 32],
    35: [0, 60, 100],
    36: [0, 80, 100],
    37: [0, 0, 230],
    38: [0, 0, 70],
    39: [0, 0, 0],
    40: [128, 64, 128],
    41: [244, 35, 232],
    42: [70, 70, 70],
    43: [102, 102, 156],
    44: [190, 153, 153],
    45: [153, 153, 153],
    46: [250, 170, 30],
    47: [220, 220, 0],
    48: [107, 142, 35],
    49: [152, 251, 152],
    50: [70, 130, 180],
    51: [220, 20, 60],
    52: [255, 0, 0],
    53: [0, 0, 142],
    54: [119, 11, 32],
    55: [0, 60, 100],
    56: [0, 80, 100],
    57: [0, 0, 230],
    58: [0, 0, 70],
    59: [0, 0, 0],
    60: [128, 64, 128],
    61: [244, 35, 232],
    62: [70, 70, 70],
    63: [102, 102, 156],
    64: [190, 153, 153],
    65: [153, 153, 153],
    66: [250, 170, 30],
    67: [220, 220, 0],
    68: [107, 142, 35],
    69: [152, 251, 152],
    70: [70, 130, 180],
    71: [220, 20, 60],
    72: [255, 0, 0],
    73: [0, 0, 142],
    74: [119, 11, 32],
    75: [0, 60, 100],
    76: [0, 80, 100],
    77: [0, 0, 230],
    78: [0, 0, 70],
    79: [0, 0, 0],
    80: [128, 64, 128],
    81: [244, 35, 232],
    82: [70, 70, 70],
    83: [102, 102, 156],
    84: [190, 153, 153],
    85: [153, 153, 153],
    86: [250, 170, 30],
    87: [220, 220, 0],
    88: [107, 142, 35],
    89: [152, 251, 152],
    90: [70, 130, 180],
    91: [220, 20, 60],
    92: [255, 0, 0],
    93: [0, 0, 142],
    94: [119, 11, 32],
    95: [0, 60, 100],
    96: [0, 80, 100],
    97: [0, 0, 230],
    98: [0, 0, 70],
    99: [0, 0, 0],
}
FONT_MAX = 50


def drawCaption(convas, vis_size, sents, off1=2, off2=2):

    img_txt = Image.fromarray(convas)
    fnt = ImageFont.truetype("./FreeMono.ttf", 45)
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    word_index_list = []
    for i in range(len(sents)):
        # cap = captions[i].data.cpu().numpy()
        cap = [w for w in sents[i] if not w.startswith("[")]
        cap = ["[CLS]"] + cap
        sentence = []
        word_index = []
        word = ""
        for j in range(len(cap)):

            word += sents[i][j].strip("#")

            if j == (len(cap)) - 1:
                word_index.append(j)
            else:
                if sents[i][j + 1].startswith("#"):
                    continue
                else:
                    word_index.append(j)

            d.text(
                ((len(sentence) + off1) * (vis_size + off2), i * FONT_MAX),
                "%s" % (word),
                font=fnt,
                fill=(255, 255, 255, 255),
            )
            sentence.append(word)
            word = ""

        sentence_list.append(sentence)
        word_index_list.append(word_index)

    return img_txt, sents, word_index_list


def build_attention_images(
    real_imgs,
    attn_maps,
    max_word_num=None,  # TODO: remove
    nvis=8,
    rand_vis=False,
    sentences=None,
):

    att_sze = attn_maps[0].shape[-1]
    batch_size = real_imgs.shape[0]

    word_counts = []
    for sent in sentences:
        sent = [s for s in sent if (not s.startswith("#")) and (not s.startswith("["))]
        word_counts.append(len(sent) + 1)
    max_word_num = max(word_counts)

    if rand_vis:
        loop_idx = np.random.choice(len(real_imgs), size=nvis, replace=False)
    else:
        loop_idx = np.arange(nvis)

    if (att_sze == 17) or (att_sze == 19):
        vis_size = att_sze * 16
    else:
        vis_size = real_imgs.size(2)

    text_convas = np.ones(
        [batch_size * FONT_MAX, (max_word_num + 2) * (vis_size + 2), 3], dtype=np.uint8
    )

    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]

    real_imgs = nn.Upsample(size=(vis_size, vis_size), mode="bilinear")(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences, word_index_list = drawCaption(text_convas, vis_size, sentences)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in loop_idx:

        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        # --> 1 x 1 x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True)
        attn = torch.cat([attn_max[0], attn], 1)

        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = attn.shape[0]
        #
        img = real_imgs[i]
        lrI = img
        row = [lrI, middle_pad]
        row_merge = [img, middle_pad]
        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0

        # including first max attention index
        word_end_list = [0] + [idx + 1 for idx in word_index_list[i]]
        word_level_attn = []

        for j in range(num_attn):

            one_map = attn[j]
            if (vis_size // att_sze) > 1:
                one_map = skimage.transform.pyramid_expand(
                    one_map, sigma=20, upscale=vis_size // att_sze, multichannel=True
                )

            word_level_attn.append(one_map)
            if j in word_end_list:
                one_map = np.mean(word_level_attn, axis=0)
                word_level_attn = []
            else:
                continue

            row_beforeNorm.append(one_map)
            minV = one_map.min()
            maxV = one_map.max()

            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV

        for j in range(seq_len + 1):
            if j < len(row_beforeNorm):
                one_map = row_beforeNorm[j]
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                one_map *= 255

                PIL_im = Image.fromarray(np.uint8(img))
                PIL_att = Image.fromarray(np.uint8(one_map))
                merged = Image.new("RGBA", (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new("L", (vis_size, vis_size), (210))
                merged.paste(PIL_im, (0, 0))
                merged.paste(PIL_att, (0, 0), mask)
                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad

            row.append(one_map)
            row.append(middle_pad)
            row_merge.append(merged)
            row_merge.append(middle_pad)

        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[i * FONT_MAX : (i + 1) * FONT_MAX]

        if txt.shape[1] != row.shape[1]:
            print("txt", txt.shape, "row", row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None
