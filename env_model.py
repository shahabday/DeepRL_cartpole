import torch

model = torch.load('env_model/my_model_full.pth', weights_only=False)
dummy_input = outputs_array[0,:5].copy()
dummy_input = torch.from_numpy(dummy_input).float()
dummy_input = dummy_input.unsqueeze(0)  # add batch dimension
results = model(dummy_input)
print(results)