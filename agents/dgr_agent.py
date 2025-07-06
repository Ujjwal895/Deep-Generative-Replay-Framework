import torch
from torch.utils import data
import numpy as np
from agents.base import ContinualLearner
from models.generator import CVAE
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter


class DGRAgent(ContinualLearner):
    def __init__(self, model, opt, params):
        super().__init__(model, opt, params)
        self.device = params.device if hasattr(params, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mem_iters = getattr(params, 'mem_iters', 1)  # Default to 1 if not found
        
        self.model = model.to(self.device)
        self.opt = opt
        self.params = params
        
        # Set up generator
        self.generator = CVAE(params).to(self.device)
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.params.gen_lr)
        self.trained_gen = False

        # Infer input shape
        flat_dim = params.input_dim
        num_channels = 3
        side = int((flat_dim / num_channels) ** 0.5)
        self.input_shape = (num_channels, side, side)

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)

        # Setup data loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0, drop_last=True)

        # Set model to train mode
        self.model.train()

        # Trackers
        losses_real = AverageMeter()
        losses_gen = AverageMeter()
        acc_real = AverageMeter()
        acc_gen = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                for _ in range(self.mem_iters):
                    # 1. Train classifier on real data
                    logits = self.model(batch_x)
                    loss_real = self.criterion(logits, batch_y)

                    # Apply KD trick if enabled
                    if self.params.trick.get('kd_trick', False):
                        loss_real = 1 / (self.task_seen + 1) * loss_real + \
                            (1 - 1 / (self.task_seen + 1)) * self.kd_manager.get_kd_loss(logits, batch_x)
                    if self.params.trick.get('kd_trick_star', False):
                        alpha = 1 / ((self.task_seen + 1) ** 0.5)
                        loss_real = alpha * loss_real + (1 - alpha) * self.kd_manager.get_kd_loss(logits, batch_x)

                    _, preds = torch.max(logits, 1)
                    acc_real.update((preds == batch_y).float().mean().item(), batch_y.size(0))
                    losses_real.update(loss_real.item(), batch_y.size(0))

                    self.opt.zero_grad()
                    loss_real.backward()
                    self.opt.step()

                    # 2. Train generator on real data
                    x_flat = batch_x.view(batch_x.size(0), -1) / 255.0
                    self.gen_optimizer.zero_grad()
                    loss_gen = self.generator.loss_function(x_flat.detach(), batch_y)
                    loss_gen.backward()
                    self.gen_optimizer.step()
                    self.trained_gen = True

                    # 3. Train classifier on generated data (generative replay)
                    if self.trained_gen:
                        x_gen_flat, y_gen = self.generator.sample(self.params.replay_batch_size)
                        x_gen = (x_gen_flat * 255.0).view(-1, *self.input_shape).to(self.device)
                        y_gen = y_gen.to(self.device)

                        gen_logits = self.model(x_gen)
                        loss_gen_replay = self.criterion(gen_logits, y_gen)

                        # Apply KD tricks to generative replay if needed
                        if self.params.trick.get('kd_trick', False):
                            loss_gen_replay = 1 / (self.task_seen + 1) * loss_gen_replay + \
                                (1 - 1 / (self.task_seen + 1)) * self.kd_manager.get_kd_loss(gen_logits, x_gen)
                        if self.params.trick.get('kd_trick_star', False):
                            alpha = 1 / ((self.task_seen + 1) ** 0.5)
                            loss_gen_replay = alpha * loss_gen_replay + \
                                (1 - alpha) * self.kd_manager.get_kd_loss(gen_logits, x_gen)

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

    def _prep_xy(self, x, y):
        # NHWC→NCHW, numpy→Tensor, move to device
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().permute(0, 3, 1, 2).to(self.device)
        else:
            x = x.to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long().to(self.device)
        else:
            y = y.to(self.device)
        return x, y
