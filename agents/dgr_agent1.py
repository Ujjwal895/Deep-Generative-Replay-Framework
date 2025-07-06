
import torch
from torch.utils import data
import numpy as np
from agents.base import ContinualLearner
from models.generator import CVAE
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter

class DGRAgent1(ContinualLearner):
    def __init__(self, model, opt, params):
        super().__init__(model, opt, params)
        self.device = params.device if hasattr(params, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.opt = opt
        self.params = params
        self.mem_iters = params.mem_iters


        self.generator = CVAE(params).to(self.device)
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.params.gen_lr)
        self.trained_gen = False

        flat_dim = params.input_dim
        side = int((flat_dim / 3) ** 0.5)
        self.input_shape = (3, side, side)

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)

        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0, drop_last=True)

        self.model.train()
        losses_real = AverageMeter()
        losses_gen = AverageMeter()
        acc_real = AverageMeter()
        acc_gen = AverageMeter()

        for ep in range(self.epoch):
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = maybe_cuda(batch_x, self.cuda), maybe_cuda(batch_y, self.cuda)

                for _ in range(self.mem_iters):
                    logits = self.model(batch_x)
                    loss_real = self.criterion(logits, batch_y)

                    _, preds = torch.max(logits, 1)
                    acc_real.update((preds == batch_y).float().mean().item(), batch_y.size(0))
                    losses_real.update(loss_real.item(), batch_y.size(0))

                    self.opt.zero_grad()
                    loss_real.backward()
                    self.opt.step()

                    x_flat = batch_x.view(batch_x.size(0), -1) / 255.0
                    self.gen_optimizer.zero_grad()
                    loss_gen = self.generator.loss_function(x_flat.detach(), batch_y)
                    loss_gen.backward()
                    self.gen_optimizer.step()
                    self.trained_gen = True

                    if self.trained_gen:
                        x_gen_flat, y_gen = self.generator.sample(self.params.replay_batch_size)
                        x_gen = (x_gen_flat * 255.0).view(-1, *self.input_shape).to(self.device)
                        y_gen = y_gen.to(self.device)

                        gen_logits = self.model(x_gen)
                        loss_gen_replay = self.criterion(gen_logits, y_gen)

                        _, preds_gen = torch.max(gen_logits, 1)
                        acc_gen.update((preds_gen == y_gen).float().mean().item(), y_gen.size(0))
                        losses_gen.update(loss_gen_replay.item(), y_gen.size(0))

                        self.opt.zero_grad()
                        loss_gen_replay.backward()
                        self.opt.step()

                if i % 100 == 1 and self.verbose:
                    print(
                        f"==>>> ep: {ep}, it: {i}, real loss: {losses_real.avg():.6f}, "
                        f"real acc: {acc_real.avg():.3f}, gen loss: {losses_gen.avg():.6f}, gen acc: {acc_gen.avg():.3f}"
                    )

        self.after_train()
