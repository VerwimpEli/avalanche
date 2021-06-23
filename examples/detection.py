import torchvision.transforms
from avalanche.benchmarks.utils import AvalancheDataset, AvalancheDatasetType
from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy

import torch
from torch.utils.data import DataLoader
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from typing import Optional, List
from avalanche.training.plugins.evaluation import default_logger, EvaluationPlugin

import hdata as haitain
from htransforms import get_transform


def collate_fn(batch):
    return list(zip(*batch))


def empty(*args, **kwargs):
    return torch.tensor(0.0)


class DetectionStrategy(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer,
                 criterion=CrossEntropyLoss(),
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def make_train_dataloader(self, num_workers=0, shuffle=True,
                              pin_memory=True, **kwargs):
        """
        Necessary (?) because default collate_fn doesn't work with detection datasets.
        Maybe there's a better way of doing this.
        """
        self.dataloader = DataLoader(
            self.adapted_dataset,
            batch_size=self.train_mb_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

    def _unpack_minibatch(self):
        '''
        Necessary because in detection, x and y aren't
        tensors but lists (of tensors, and dicts)
        '''
        assert len(self.mbatch) >= 3
        self.mbatch[0] = list(sample.to(self.device) for sample in self.mbatch[0])
        self.mbatch[1] = [{k: v.to(self.device) for k, v in t.items()} for t in self.mbatch[1]]
        self.mbatch[-1] = torch.tensor(self.mbatch[-1]).to(self.device)
        # TODO: there could be more tensors in the mbatch

    def forward(self):
        """
        Override necessary because the torchvision models
        calculate losses inside the forward call of the model.
        """
        loss_dict = model(self.mb_x, self.mb_y)
        self.loss += sum(loss_value for loss_value in loss_dict.values())
        return loss_dict


device = 'cuda' if torch.cuda.is_available() else 'cpu'

root = "/home/eli/Documents/Doctoraat/code/data/huawei/dataset/labeled"
splits = ['train', 'val', 'val', 'val']

task_dicts = [{'city': 'Shanghai', 'location': 'Citystreet', 'period': 'Daytime', 'weather': 'Clear'},
              {'city': 'Shanghai', 'location': 'Highway', 'period': 'Daytime', 'weather': ['Clear', 'Overcast']},
              {'period': 'Night'},
              {'city': 'Guangzhou', 'location': 'Citystreet', 'period': 'Daytime', 'weather': 'Rainy'}]

match_fns = [haitain.create_match_fn_from_dict(td) for td in task_dicts]
train_datasets = [haitain.get_matching_set(root, split, match_fn, get_transform(True)) for match_fn, split in
                  zip(match_fns, splits)]
test_datasets = haitain.get_domain_sets(root, 'test', keys=['location', 'period', 'weather'],
                                        transform=get_transform(False))
train_transform = None
eval_transform = None


transform_groups = dict(
    train=(train_transform, None),
    eval=(eval_transform, None)
)

train_dataset = AvalancheDataset(
    train_datasets[0],
    transform_groups=transform_groups,
    initial_transform_group='train',
    dataset_type=AvalancheDatasetType.REGRESSION)

test_dataset = AvalancheDataset(
    test_datasets[0],
    transform_groups=transform_groups,
    initial_transform_group='eval',
    dataset_type=AvalancheDatasetType.REGRESSION)

benchmark = create_multi_dataset_generic_benchmark(
        train_datasets=[train_dataset],
        test_datasets=[test_dataset])

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# I think there's no option currently to not specify a criterion.
# Didn't search well though, so placeholder for now.
criterion = empty

# Current metrics don't work, but passing no evaluator will default to
# some basic metrics which should be prevented
eval_plugin = EvaluationPlugin([])

strategy = DetectionStrategy(
    model, optimizer, criterion, train_mb_size=1, train_epochs=1,
    eval_mb_size=1, device=device, eval_every=-1, evaluator=eval_plugin)


for experience in benchmark.train_stream:
    strategy.train(experience)
