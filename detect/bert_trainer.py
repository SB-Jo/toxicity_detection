import torch
import torch.nn.utils as torch_utils

from ignite.engine import Events

from detect.utils import get_grad_norm, get_parameter_norm
from detect.trainer import Trainer, IgniteEngine

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

class BertEngine(IgniteEngine):
    def __init__(self, func, model, crit, optimizer, scheduler, config):
        self.scheduler = scheduler

        super().__init__(func, model, crit, optimizer, config)

    @staticmethod
    def train(engine, mini_batch):
        engine.model.train()
        engine.optimizer.zero_grad()

        x, mask, y = mini_batch['input_ids'], mini_batch['attention_mask'], mini_batch['targets']

        x, mask, y = x.to(engine.device), mask.to(engine.device), y.to(engine.device)
        
        y_hat = engine.model(x, attention_mask=mask, return_dict=True).logits

        loss = engine.crit(y_hat, y)
        loss.backward()

        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        engine.optimizer.step()
        engine.scheduler.step()

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, mask, y = mini_batch['input_ids'], mini_batch['attention_mask'], mini_batch['targets']

            x, mask, y = x.to(engine.device), mask.to(engine.device), y.to(engine.device)

            # Take feed-forward
            y_hat = engine.model(x, attention_mask=mask, return_dict=True).logits

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
        }



class BertTrainer(Trainer):
    def __init__(self, config):
        self.config = config

    def train(
        self,
        model, crit, optimizer, scheduler, train_loader, valid_loader
    ):
        train_engine = BertEngine(
            BertEngine.train,
            model, crit, optimizer, scheduler, self.config
        )

        validation_engine = BertEngine(
            BertEngine.validate,
            model, crit, optimizer, scheduler, self.config
        )

        BertEngine.attach(
            train_engine,
            validation_engine,
            verbose = self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            run_validation, # function
            validation_engine, valid_loader, # arguments
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            BertEngine.check_best, # function
        )

        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            BertEngine.save_model,
            train_engine, self.config
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        model.load_state_dict(validation_engine.best_model)

        return model