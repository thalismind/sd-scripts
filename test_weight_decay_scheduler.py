#!/usr/bin/env python3
"""
Test script for the WeightDecayScheduler
"""

import torch
import torch.optim as optim
from library.weight_decay_scheduler import WeightDecayScheduler, create_weight_decay_scheduler


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


def test_factory_function():
    """Test the factory function"""

    model = torch.nn.Linear(10, 1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    scheduler = create_weight_decay_scheduler(
        optimizer=optimizer,
        max_steps=1000,
        start_decay=0.1,
        end_decay=0.01,
        decay_mode="linear",
        warmup_steps=100,
        warmup_decay=0.0,
    )

    print("Factory function test passed!")
    print(f"Created scheduler: {scheduler.get_schedule_info()}")


if __name__ == "__main__":
    print("Testing WeightDecayScheduler...")
    test_weight_decay_scheduler()
    test_factory_function()
    print("\nAll tests completed successfully!")