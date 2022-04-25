"""Classification for MPHRQE."""

import logging
from typing import Tuple
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mphrqe.data.loader import QueryGraphBatch
from mphrqe.models import QueryEmbeddingModel
from mphrqe.similarity import Similarity


def get_class_metrics(
    distances: np.ndarray, 
    answers: np.ndarray, 
    threshold: float
) -> Tuple[float, float, float, float]:

    epsilon = 1e-6
    
    selection_mask = (distances < threshold)

    tp = np.sum(np.where(selection_mask, answers, False), axis=1)
    fp = np.sum(np.where(selection_mask, ~answers, False), axis=1)
    tn = np.sum(np.where(~selection_mask, ~answers, False), axis=1)
    fn = np.sum(np.where(~selection_mask, answers, False), axis=1)

    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = (2 * precision * recall) / (precision + recall + epsilon)

    # Return the average of the metrics for the whole batch
    avg_acc = np.mean(accuracy).item()
    avg_prec = np.mean(precision).item()
    avg_rec = np.mean(recall).item()
    avg_f1 = np.mean(f1).item()
    return avg_acc, avg_prec, avg_rec, avg_f1


def find_best_threshold(
    distances: np.ndarray, 
    answers: np.ndarray, 
    struct_str: str = None, 
    num_steps: int = 50
) -> Tuple[float, float, float, float, float]:

    pos_dists = np.where(answers, distances, 0)
    pos_dists[pos_dists==0] = np.nan
    pos_dists_mean = np.nanmean(pos_dists)
    pos_dists_3std = np.nanstd(pos_dists) * 3

    pbounds = {'threshold': (pos_dists_mean - pos_dists_3std, pos_dists_mean + pos_dists_3std)}

    def objective(threshold):
        accuracy, precision, recall, f1 = get_class_metrics(distances, answers, threshold)
        return f1
    
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        n_iter=num_steps,
    )

    # save figure if needed
    if struct_str is not None:
        x = np.array([step['params']['threshold'] for step in optimizer.res])
        y = np.array([step['target'] for step in optimizer.res])
        x_order = np.argsort(x)
        x = x[x_order]
        y = y[x_order]
        plt.plot(x, y, 'x-', label=struct_str)

    best_threshold = optimizer.max['params']['threshold']
    best_accuracy, best_precision, best_recall, best_f1 = get_class_metrics(distances, answers, best_threshold)

    return best_threshold, best_accuracy, best_precision, best_recall, best_f1


@torch.no_grad()
def find_val_thresholds(
    data_loader: torch.utils.data.DataLoader[QueryGraphBatch],
    model: QueryEmbeddingModel,
    similarity: Similarity,
):
    model.eval()
    
    # tracking thresholds and scores
    thresholds = {}
    metrics = {}

    step = 0
    total_steps = len(data_loader)

    # track queries, distances and answers
    all_query_stuctures = []
    all_distances = torch.empty((0, model.x_e.size(0)))
    all_answers = torch.empty((0, model.x_e.size(0)))

    for batch in tqdm(data_loader, desc="Evaluation", unit="batch", unit_scale=True):
        # embed query
        x_query = model(batch)
        # compute pairwise similarity to all entities, shape: (batch_size, num_entities)
        scores = similarity(x=x_query, y=model.x_e)
        # get answers
        answers = torch.zeros((torch.max(batch_id) + 1, model.x_e.size(0))) # initialize with zeros, size: (batch_size, num_ents)
        batch_id, entity_id = batch.targets
        for batch_id, entity_id in zip(batch_id, entity_id):
            answers[batch_id, entity_id] = 1
        # add to tracking
        all_query_stuctures.extend(query_structures) # TODO: find query structures
        all_distances = torch.cat((all_distances, scores.cpu()), dim=0)
        all_answers = torch.cat((all_answers, answers.cpu()), dim=0)

        if step % 10 == 0:
            logging.info('Gathering predictions of batches... (%d/%d) ' % (step, total_steps))
            step += 1

    # Define plot
    plt.figure(figsize=(10,10))
        
    # find best threshold for each query structure
    for struct in set(all_query_stuctures):
        logging.info(f"Finding best threshold for structure: {struct}")

        # select data for current structure
        struct_idx = torch.tensor(np.where(np.array(all_query_stuctures) == struct)[0])
        str_distances = all_distances[struct_idx, :]
        str_answers = all_answers[struct_idx, :]

        # find best threshold and metrics
        accuracy, precision, recall, f1 = find_best_threshold(
            str_distances.numpy(), 
            str_answers.bool().numpy(), 
            thresholds[struct],
            num_steps=50
        )

        # save threshold and metrics
        metrics[struct] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Save figure
    plt.xlabel('Distance threshold')
    plt.ylabel('f1-score')
    plt.legend()
    plt.savefig("threshold_search.png", facecolor='w')

    return metrics


@torch.no_grad()
def evaluate_with_thresholds(
    data_loader: torch.utils.data.DataLoader[QueryGraphBatch],
    model: QueryEmbeddingModel,
    similarity: Similarity,
):
    model.eval()
    
    # tracking thresholds and scores
    thresholds = {}
    metrics = {}

    step = 0
    total_steps = len(data_loader)

    # track queries, distances and answers
    all_query_stuctures = []
    all_distances = torch.empty((0, model.x_e.size(0)))
    all_answers = torch.empty((0, model.x_e.size(0)))

    for batch in tqdm(data_loader, desc="Evaluation", unit="batch", unit_scale=True):
        # embed query
        x_query = model(batch)
        # compute pairwise similarity to all entities, shape: (batch_size, num_entities)
        scores = similarity(x=x_query, y=model.x_e)
        # get answers
        answers = torch.zeros((torch.max(batch_id) + 1, model.x_e.size(0))) # initialize with zeros, size: (batch_size, num_ents)
        batch_id, entity_id = batch.targets
        for batch_id, entity_id in zip(batch_id, entity_id):
            answers[batch_id, entity_id] = 1
        # add to tracking
        all_query_stuctures.extend(query_structures) # TODO: find query structures
        all_distances = torch.cat((all_distances, scores.cpu()), dim=0)
        all_answers = torch.cat((all_answers, answers.cpu()), dim=0)

        if step % 10 == 0:
            logging.info('Gathering predictions of batches... (%d/%d) ' % (step, total_steps))
            step += 1
        
    # find best threshold for each query structure
    for struct in set(all_query_stuctures):
        logging.info(f"Calculating metrics for structure: {struct}")

        # select data for current structure
        struct_idx = torch.tensor(np.where(np.array(all_query_stuctures) == struct)[0])
        str_distances = all_distances[struct_idx, :]
        str_answers = all_answers[struct_idx, :]

        # find best threshold and metrics
        accuracy, precision, recall, f1 = get_class_metrics(
            str_distances.numpy(), 
            str_answers.bool().numpy(), 
            thresholds[struct]
        )

        # save threshold and metrics
        metrics[struct] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return metrics
