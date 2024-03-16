import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset.CoNLL import CoNLL2003
from model import MambaForNER
from mamba_ssm.models.config_mamba import MambaConfig
from evaluate import Evaluate

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup():
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            valid_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.valid_data = valid_data
        self.optimizer = optimizer
        self.evaluate = Evaluate()
        self.save_every = save_every
        self.epochs_run = 0
        self.train_loss = list()
        self.valid_loss = list()
        self.f1_score = list()
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
    
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        train_loss = 0.0
        valid_loss = 0.0
        total_predicted_label = None
        total_original_label = None

        self.train_data.sampler.set_epoch(epoch)
        self.valid_data.sampler.set_epoch(epoch)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.model.train()
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id).view(-1)
            self.optimizer.zero_grad()
            output = self.model(source)
            output = output.view(-1, 9)
            loss = F.cross_entropy(output, targets, reduction='mean', ignore_index=-1)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        self.train_loss.append(train_loss / len(next(iter(self.train_data))[0]))
        for source, targets in self.valid_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id).view(-1)
            output = self.model(source)
            output = output.view(-1, 9)
            loss = F.cross_entropy(output, targets, reduction='mean', ignore_index=-1)
            pred_label = torch.argmax(torch.softmax(output, dim=1), dim=1)
            if total_predicted_label is None:
                total_predicted_label = pred_label
            else:
                total_predicted_label = torch.cat((total_predicted_label, pred_label), dim=0)

            if total_original_label is None:
                total_original_label = targets
            else:
                total_original_label = torch.cat((total_original_label, targets), dim=0)

            valid_loss += loss.item()
        self.valid_loss.append(valid_loss / len(next(iter(self.valid_data))[0]))
        F1_score = self.evaluate(total_predicted_label,
                                 total_original_label)
        self.f1_score.append(F1_score)
        print(f'F1 score: {F1_score}')

        
            
    def _save_snapshot(self, epoch):
        snapshot = dict()
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        snapshot["TRAIN_LOSS"] = self.train_loss
        snapshot["VALID_LOSS"] = self.valid_loss
        snapshot["F1_SCORE"] = self.f1_score
        torch.save(snapshot, "snapshot.pt")
        print(f"Epoch {epoch} | Training snapshot saved at snapshot.pt")
        
    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.train_loss = snapshot["TRAIN_LOSS"]
        self.valid_loss = snapshot["VALID_LOSS"]
        self.f1_score = snapshot["F1_SCORE"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
        
    def train(self, max_epochs: int):

        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            # if epoch % self.save_every == 0:
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

def load_train_objs():
    model_config = MambaConfig()
    model_config.d_model = 1024
    model_config.n_layer = 48
    train_set = CoNLL2003(dir_path='data/CoNLL2003', phase='train', max_length=48)  # load your dataset
    valid_set = CoNLL2003(dir_path='data/CoNLL2003', phase='valid', max_length=48)
    model = MambaForNER(model_config, num_class=9 ,dtype=torch.float32)  # load your model
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-6)
    return train_set,  valid_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


# def main(device, total_epochs, save_every, batch_size):
def main(total_epochs: int, save_every: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    train_set, valid_set, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_set, batch_size=3)
    valid_data = prepare_dataloader(valid_set, batch_size=3)
    trainer = Trainer(model, train_data, valid_data,
                      optimizer, save_every, snapshot_path=snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=100, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=5, type=int, help='How often to save a snapshot')
    # parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    # device = 0  # shorthand for cuda:0
    # world_size = torch.cuda.device_count()
    # main(device, args.total_epochs, args.save_every, args.batch_size)
    # mp.spawn(main, args=(world_size, args.total_epochs, args.save_every,), nprocs=world_size)
    main(save_every=args.save_every, total_epochs=args.total_epochs)
    
