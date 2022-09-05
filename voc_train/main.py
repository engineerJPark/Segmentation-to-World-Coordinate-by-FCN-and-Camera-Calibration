from train import train
from predict import predict 

if __name__ = '__main__': 
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  torch.manual_seed(1)
  if device == 'cuda:0':
      torch.cuda.manual_seed_all(1)

  print(torch.__version__)
  print(device)
  ROOT_DIR = 'voc_data'

  history = train(model, epochs, optimizer, criterion, device=device, verbos_iter=False)
  plt.plot(history)
  plt.show()  
