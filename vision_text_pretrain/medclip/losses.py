from torch import nn

class ImageTextContrastiveLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, 
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        **kwargs,
        ):
        outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                return_loss=True,
                )
        return_res = {
            'loss_value': outputs['loss_value'],
        }
        return return_res