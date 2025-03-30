import torch 

model = torch.load('env_model/my_model_full.pth' , weights_only=False)
model.eval()

#mock input 

dummy_input = torch.randn(1,4)

state_and_action = torch.tensor([0.,0.,0.,0.,0])
dummy_input= state_and_action.unsqueeze(0)
'''
model = torch.load('env_model/my_model_full.pth', weights_only=False)
dummy_input = outputs_array[0,:5].copy()
dummy_input = torch.from_numpy(dummy_input).float()
dummy_input = dummy_input.unsqueeze(0)  # add batch dimension
results = model(dummy_input)
'''


with torch.no_grad():
  output = model(dummy_input)

print(output.numpy()[0])


                   
