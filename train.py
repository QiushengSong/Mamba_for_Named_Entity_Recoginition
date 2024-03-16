from dataset.CoNLL import CoNLL2003
from torch.utils.data import DataLoader
from mamba_ssm.models.config_mamba import MambaConfig
from model import MambaForNER
from tqdm import tqdm
from evaluate import Evaluate
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.nn.init as init

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dtype', default=torch.float32, help='data type')
parser.add_argument('--device', default='cuda:0', help='device')
parser.add_argument('--epochs', default=50, type=int, help='number of train epochs')
parser.add_argument('--resume', default=False, type=bool, help='resume from checkpoint')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--max_length', default=512, type=int,
                    help='Controls the maximum length to use by one of the truncation/padding parameters.')
args = parser.parse_args()

checkpoint_path = 'checkpoint.pth'
dtype = args.dtype
device = torch.device(args.device)
data_dir_path = 'data/CoNLL2003'
batch_size = args.batch_size
max_length = args.max_length

dataset = {phase: CoNLL2003(data_dir_path, phase, args.max_length)
           for phase in ['train', 'valid']}
dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=False, pin_memory=True)
              for x in ['train', 'valid']}
mambaconfig = MambaConfig()
mambaconfig.d_model = 1024
mambaconfig.n_layer = 48
print('Loading model ...')
model = MambaForNER(mambaconfig, num_class=9, device=device, dtype=dtype)
#for name, param in model.backbone.lm_head.named_parameters():
#    init.xavier_normal_(param.data)

learning_rate = args.lr

print('Loading loss function ...')
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)

print('Loading optimizer ...')
optimizer = optim.Adam(model.backbone.lm_head.parameters(), lr=learning_rate)

print('Loading learning rate scheduler')
#scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
epochs = args.epochs

if args.resume:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    checkpoint = {'model_state_dict': None,
                  'optimizer_state_dict': None,
                  'lr_scheduler_state_dict': None,
                  'train_loss': list(),
                  'valid_loss': list(),
                  'epoch': 0,
                  "best_loss": 100.0,
                  }
    start_epoch = 0

print('The model is being trained')
for epoch in range(start_epoch, epochs):
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.to(device)
            model.train()
            train_loss = 0.0
            with tqdm(total=len(dataloader[phase]),
                      desc=f"Epoch: {epoch + 1}/{epochs}",
                      dynamic_ncols=True
                      ) as pbar:

                for batch_tokens, batch_label in dataloader[phase]:
                    optimizer.zero_grad()

                    batch_tokens = batch_tokens.to(device)
                    batch_label = batch_label.to(device).view(-1)
                    train_output = model(batch_tokens)
                    train_output = train_output.logits.view(-1, 9)
                    # lm_logits = self.lm_head(hidden_states)
                    # CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
                    # return CausalLMOutput(logits=lm_logits)

                    loss = criterion(train_output,
                                     batch_label)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss
                    pbar.update()
                    pbar.set_postfix_str(f'loss: {loss.item()}')
            checkpoint['train_loss'].append(train_loss / len(next(iter(dataset[phase]))[0]))
        else:
            model.to(torch.device("cuda:1"))
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                # total_pred_entity = list()
                total_predicted_label = None
                total_original_label = None
                with tqdm(total=len(dataloader[phase]),
                      desc=f'Epoch: {epoch + 1}/{epochs}',
                      dynamic_ncols=True
                      ) as pbar:
                    for batch_tokens, batch_label in dataloader[phase]:
                        batch_tokens = batch_tokens.to(torch.device("cuda:1"))
                        batch_label = batch_label.to(torch.device("cuda:1")).view(-1)
                        valid_output = model(batch_tokens)
                        valid_output = valid_output.logits.view(-1, 9)
                        loss = criterion(valid_output,
                                     batch_label)

                        pred_label = torch.argmax(torch.softmax(valid_output, dim=1), dim=1)
                        if total_predicted_label is None:
                            total_predicted_label = pred_label
                        else:
                            total_predicted_label = torch.cat((total_predicted_label, pred_label), dim=0)
                        # entity = list(filter(lambda x: x != 0, pred_label))
                        if total_original_label is None:
                            total_original_label = batch_label
                        else:
                            total_original_label = torch.cat((total_original_label, batch_label), dim=0)
                        valid_loss += loss.item()
                        pbar.update()
                        pbar.set_postfix_str(f'loss: {loss.item()}')
                if checkpoint["best_loss"]:
                    if (valid_loss / len(dataset[phase])) < checkpoint["best_loss"]:
                        checkpoint['model_state_dict'] = model.state_dict()
                        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

                checkpoint['valid_loss'].append(valid_loss / len(next(iter(dataset[phase]))[0]))
                evaluate = Evaluate()
                F1_score = evaluate(total_predicted_label,
                                    total_original_label)
                print(f'F1 score: {F1_score}')
    # scheduler.step()
    checkpoint["epoch"] = epoch
    # checkpoint['lr_scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, checkpoint_path)
torch.save(model.state_dict(), 'last.pth')
