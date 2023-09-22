from ray import air, tune
from ray.tune import Stopper
from collections import defaultdict

class EarlyStopper(Stopper):
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = defaultdict(lambda: 0)
        self.best_score = defaultdict(lambda: None)
        self.early_stop = False
        self.delta = delta

    def __call__(self, trial_id: str, result: dict) -> bool:
        score = -result["total_loss"]

        if self.best_score[trial_id] is None:
            self.best_score[trial_id] = score

        elif score < self.best_score[trial_id] + self.delta:
            self.counter[trial_id] += 1

            if self.counter[trial_id] >= self.patience:
                return True
        else:
            self.best_score[trial_id] = score
            self.counter[trial_id] = 0

        return False

    def stop_all(self) -> bool:
        return False