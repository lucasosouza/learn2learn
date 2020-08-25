#!/usr/bin/env python3

"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load Omniglot, and
    * sample tasks and split them in adaptation and evaluation sets.
"""

import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim



def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        train_error /= len(adaptation_data)
        learner.adapt(train_error, freeze_feature_extraction=freeze_feature_extraction)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy, evaluation_data, evaluation_labels

def sample_random_set(batch, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    random_indices = np.zeros(data.size(0), dtype=bool)
    random_indices[np.arange(shots*ways) * 2] = True
    random_indices = torch.from_numpy(random_indices)
    random_data, random_labels = data[random_indices], labels[random_indices]

    return random_data, random_labels

def main(
        ways=5,
        shots=1,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=60000,
        cuda=True,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Load train/validation/test tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets('omniglot',
                                                  train_ways=ways,
                                                  train_samples=2*shots,
                                                  test_ways=ways,
                                                  test_samples=2*shots,
                                                  num_tasks=20000,
                                                  root='~/data',
    )

    # Create model
    model = l2l.vision.models.OmniglotFCOML(28 ** 2, ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0

        learner = maml.clone()
        evaluation_data_all_tasks = []
        evaluation_labels_all_tasks = []

        for task in range(meta_batch_size):
            # Compute meta-training loss
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy, evaluation_data, evaluation_labels = \
                fast_adapt(batch,
                          learner,
                          loss,
                          adaptation_steps,
                          shots,
                          ways,
                          device,
                          freeze_feature_extraction=True)
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            evaluation_data_all_tasks.append(evaluation_data)
            evaluation_labels_all_tasks.append(evaluation_labels)

            # Compute meta-validation loss
            learner = maml.clone()
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy, _, _ = \
                fast_adapt(batch,
                           learner,
                           loss,
                           adaptation_steps,
                           shots,
                           ways,
                           device,
                           freeze_feature_extraction=True)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # sample remember/random set
        random_batch = tasksets.train.sample()
        random_data, random_labels = sample_random_set(
            random_batch, shots, ways, device
        )
        # concatenate with trajectory
        meta_training_test_data = torch.stack(evaluation_data_all_tasks, random_data)
        meta_training_test_labels = torch.stack(evaluation_labels_all_tasks, random_labels)

        # learn - outer loop
        loss = loss(learner(meta_training_test_data), meta_training_test_labels)
        loss.backward()
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


if __name__ == '__main__':
    main(meta_batch_size=5, shots=5)
