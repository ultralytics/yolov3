import math

from sparseml.pytorch.optim import ScheduledModifierManager, ScheduledOptimizer
from sparseml.pytorch.utils import SparsificationGroupLogger

from utils.torch_utils import is_parallel


class SparseMLWrapper(object):
    def __init__(self, model, recipe, rank, start_epoch):
        self.enabled = bool(recipe)
        self.model = model
        self.recipe = recipe
        self.rank = rank
        self.start_epoch = start_epoch

        self.manager = ScheduledModifierManager.from_yaml(recipe) if self.enabled else None
        self.logger = None
        self.tb_writer = None
        self.wandb_logger = None

        if self.qat_active(start_epoch):
            # enable the quantization modifiers to start immediately if they're scheduled
            for quant_mod in self.manager.quantization_modifiers:
                quant_mod.enable_on_initialize = True
            self.manager.initialize(model, None)

    def state_dict(self):
        return {
            'recipe': str(self.manager) if self.enabled else None
        }

    def initialize_loggers(self, logger, tb_writer, wandb_logger):
        self.logger = logger
        self.tb_writer = tb_writer
        self.wandb_logger = wandb_logger

        if self.enabled and wandb_logger.wandb:
            artifact = wandb_logger.wandb.Artifact('recipe', type='recipe')
            with artifact.new_file('recipe.yaml') as file:
                file.write(str(self.manager))
            wandb_logger.wandb.log_artifact(artifact)

    def setup_optimizer(self, optimizer, model, dataloader):
        if not self.enabled:
            return optimizer

        def _logging_lambda(log_tag, log_val, log_vals, step, walltime):
            if not self.wandb_logger or not self.wandb_logger.wandb:
                return

            if log_val is not None:
                self.wandb_logger.log({log_tag: log_val})

            if log_vals:
                self.wandb_logger.log(log_vals)

        return ScheduledOptimizer(
            optimizer,
            model if not is_parallel(model) else model.module,
            self.manager,
            steps_per_epoch=len(dataloader),
            loggers=[SparsificationGroupLogger(
                lambda_func=_logging_lambda,
                python=self.logger,
                tensorboard=self.tb_writer,
                enabled=self.rank in [-1, 0]
            )]
        )

    def check_lr_override(self, scheduler):
        # Override lr scheduler if recipe makes any LR updates
        if self.enabled and self.manager.learning_rate_modifiers:
            self.logger.info('Disabling LR scheduler, managing LR using SparseML recipe')
            scheduler = None

        return scheduler

    def check_epoch_override(self, epochs):
        # Override num epochs if recipe explicitly modifies epoch range
        if self.enabled and self.manager.epoch_modifiers and self.manager.max_epochs:
            epochs = self.manager.max_epochs or epochs  # override num_epochs
            self.logger.info(f'Overriding number of epochs from SparseML manager to {epochs}')

        return epochs

    def qat_active(self, epoch):
        if not self.enabled or not self.manager.quantization_modifiers:
            return False

        qat_start = max([mod.start_epoch for mod in self.manager.quantization_modifiers])

        return qat_start < epoch + 1

    def reset_best(self, epoch):
        if not self.enabled:
            return False

        # if pruning is active or quantization just started, need to reset best checkpoint
        # this is in case the pruned and/or quantized model do not fully recover
        pruning_start = math.floor(max([mod.start_epoch for mod in self.manager.pruning_modifiers])) \
            if self.manager.pruning_modifiers else -1
        pruning_end = math.ceil(max([mod.end_epoch for mod in self.manager.pruning_modifiers])) \
            if self.manager.pruning_modifiers else -1
        qat_start = math.floor(max([mod.start_epoch for mod in self.manager.quantization_modifiers])) \
            if self.manager.quantization_modifiers else -1

        return (pruning_start <= epoch <= pruning_end) or epoch == qat_start
