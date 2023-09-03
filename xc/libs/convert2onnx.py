import torch
from xc.libs.custom_dtypes import MultiViewData


def convert2onnx(network, max_length=16, device="cuda:0", model_path=""):
    class OnlyEncode(torch.nn.Module):
        def __init__(self, net):
            super(OnlyEncode, self).__init__()
            self.net = net.item_encoder
                
        def forward(self, txt_input_ids, txt_attention_mask):
            txt_data = MultiViewData({"mask": txt_attention_mask,
                                      "index": txt_input_ids})
            return torch.normalize(self.net({"txt": txt_data, "img": None},
                                            True, False)[0])

    omodel = torch.onnx.export(OnlyEncode(network))
    omodel.eval()
    dummy_input = torch.randint(0, 10, (2, max_length))
    dummy_masks = torch.ones_like(dummy_input)
    torch.onnx.export(
        omodel, (dummy_input.to(device), dummy_masks.to(device)),
        f"{model_path}/encoder_onnx_model.onnx",
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=11,          # the ONNX version to export the model to
        input_names = ['txt_input_ids', 'txt_attention_mask'],
        output_names = ['embeddings'],
        dynamic_axes={'txt_input_ids' : {0:'batch_size', 1:'max_length'},
                      'txt_attention_mask' : {0:'batch_size', 1:'max_length'},}
    )
