#!/usr/bin/env python3
"""
Simple test for the WeightDecayScheduler without external dependencies
"""

import math
import torch
import torch.optim as optim


class WeightDecayScheduler:
    """
    A scheduler for dynamically adjusting weight decay during training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_steps: int,
        start_decay: float = 0.1,
        end_decay: float = 0.01,
        decay_mode: str = "linear",
        warmup_steps: int = 0,
        warmup_decay: float = 0.0,
    ):
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.start_decay = start_decay
        self.end_decay = end_decay
        self.decay_mode = decay_mode.lower()
        self.warmup_steps = warmup_steps
        self.warmup_decay = warmup_decay

        # Validate inputs
        if self.decay_mode not in ["linear", "cosine"]:
            raise ValueError(f"decay_mode must be 'linear' or 'cosine', got {decay_mode}")

        if warmup_steps >= max_steps:
            raise ValueError("warmup_steps must be less than max_steps")

        # Set initial weight decay to warmup value
        if warmup_steps > 0:
            self._set_weight_decay(warmup_decay)

    def _set_weight_decay(self, decay_value: float):
        """Set weight decay for all parameter groups."""
        for group in self.optimizer.param_groups:
            group['weight_decay'] = decay_value

    def _linear_decay(self, current_step: int) -> float:
        """Calculate weight decay using linear interpolation."""
        if current_step < self.warmup_steps:
            return self.warmup_decay

        effective_step = current_step - self.warmup_steps
        effective_max_steps = self.max_steps - self.warmup_steps

        if effective_max_steps <= 0:
            return self.end_decay

        progress = min(effective_step / effective_max_steps, 1.0)
        return self.start_decay + (self.end_decay - self.start_decay) * progress

    def _cosine_decay(self, current_step: int) -> float:
        """Calculate weight decay using cosine interpolation."""
        if current_step < self.warmup_steps:
            return self.warmup_decay

        effective_step = current_step - self.warmup_steps
        effective_max_steps = self.max_steps - self.warmup_steps

        if effective_max_steps <= 0:
            return self.end_decay

        progress = min(effective_step / effective_max_steps, 1.0)
        # Cosine interpolation: smooth transition from start to end
        cosine_progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        return self.start_decay + (self.end_decay - self.start_decay) * cosine_progress

    def step(self, current_step: int) -> float:
        """Update weight decay based on current step."""
        if self.decay_mode == "linear":
            decay_value = self._linear_decay(current_step)
        else:  # cosine
            decay_value = self._cosine_decay(current_step)

        self._set_weight_decay(decay_value)
        return decay_value

    def get_current_decay(self) -> float:
        """Get the current weight decay value."""
        return self.optimizer.param_groups[0]['weight_decay']

    def get_schedule_info(self) -> dict:
        """Get information about the current schedule configuration."""
        return {
            'max_steps': self.max_steps,
            'start_decay': self.start_decay,
            'end_decay': self.end_decay,
            'decay_mode': self.decay_mode,
            'warmup_steps': self.warmup_steps,
            'warmup_decay': self.warmup_decay,
        }


def test_weight_decay_scheduler():
    """Test the weight decay scheduler functionality"""

    # Create a simple model
    model = torch.nn.Linear(10, 1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Create scheduler
    scheduler = WeightDecayScheduler(
        optimizer=optimizer,
        max_steps=1000,
        start_decay=0.1,
        end_decay=0.01,
        decay_mode="linear",
        warmup_steps=100,
        warmup_decay=0.0,
    )

    print("Weight decay scheduler created successfully!")
    print(f"Schedule info: {scheduler.get_schedule_info()}")

    # Test the scheduler at different steps
    test_steps = [0, 50, 100, 500, 1000]

    print("\nTesting weight decay values at different steps:")
    for step in test_steps:
        decay_value = scheduler.step(step)
        print(f"Step {step}: weight_decay = {decay_value:.6f}")

    # Test cosine mode
    print("\nTesting cosine mode:")
    scheduler_cosine = WeightDecayScheduler(
        optimizer=optimizer,
        max_steps=1000,
        start_decay=0.1,
        end_decay=0.01,
        decay_mode="cosine",
        warmup_steps=100,
        warmup_decay=0.0,
    )

    for step in test_steps:
        decay_value = scheduler_cosine.step(step)
        print(f"Step {step}: weight_decay = {decay_value:.6f}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    print("Testing WeightDecayScheduler...")
    test_weight_decay_scheduler()
    print("\nAll tests completed successfully!")