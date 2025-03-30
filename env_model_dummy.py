import torch 

model = torch.load('model')
model.eval()

#mock input 

dummy_input = torch.randn(1,4)

with torch.no_grad():
  output = model(dummy_input)


                   
