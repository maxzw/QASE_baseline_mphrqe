"""Classification for MPHRQE."""

import pathlib
import pickle
import logging
from pathlib import Path
from pickle import HIGHEST_PROTOCOL
from typing import Dict, Tuple
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from gqs.loader import QueryGraphBatch
from mphrqe.models import QueryEmbeddingModel
from mphrqe.similarity import Similarity


def get_class_metrics(
    distances: np.ndarray, 
    easy_answers: np.ndarray, 
    hard_answers: np.ndarray, 
    threshold: float
) -> Tuple[float, float, float, float]:

    epsilon = 1e-6
    
    selection_mask = (distances < threshold)

    tp = np.sum(np.where(selection_mask & ~easy_answers, hard_answers, False), axis=1)
    fp = np.sum(np.where(selection_mask & ~easy_answers, ~hard_answers, False), axis=1)
    tn = np.sum(np.where(~selection_mask & ~easy_answers, ~hard_answers, False), axis=1)
    fn = np.sum(np.where(~selection_mask & ~easy_answers, hard_answers, False), axis=1)

    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = (2 * precision * recall) / (precision + recall + epsilon)

    weights = 1 / hard_answers.sum(axis=1)

    # Return the average of the metrics for the whole batch
    avg_acc = np.average(accuracy, weights=weights).item()
    avg_prec = np.average(precision, weights=weights).item()
    avg_rec = np.average(recall, weights=weights).item()
    avg_f1 = np.average(f1, weights=weights).item()

    return avg_acc, avg_prec, avg_rec, avg_f1


def find_best_threshold(
    distances: np.ndarray, 
    easy_answers: np.ndarray,
    hard_answers: np.ndarray, 
    save_path: pathlib.Path,
    struct_str: str = None, 
    num_steps: int = 50,
    model_name: str = "StarQE",
    dataset_name: str = None,
    ) -> Tuple[float, float, float, float, float]:

    # track precision and recall
    precisions = []
    recalls = []

    pos_dists = np.where((easy_answers | hard_answers), distances, np.nan) # find thresholds based on valid easy answers
    pos_dists_mean = np.nanmean(pos_dists)
    pos_dists_std3 = np.nanstd(pos_dists)
    
    pbounds = {'threshold': (pos_dists_mean - pos_dists_std3*5, pos_dists_mean + pos_dists_std3*5)}
    logging.info("Using the following bounds: {}".format(pbounds))

    def objective(threshold):
        accuracy, precision, recall, f1 = get_class_metrics(distances, easy_answers, hard_answers, threshold)
        precisions.append(precision)
        recalls.append(recall)
        return f1
    
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=20,
        n_iter=num_steps,

    )

    if (dataset_name is not None) and (struct_str is not None):
        # create a new figure with a random hash
        x = np.array([step['params']['threshold'] for step in optimizer.res])
        y = np.array([step['target'] for step in optimizer.res])
        x_order = np.argsort(x)
        x = x[x_order]
        y = y[x_order]
        r = np.array(recalls)[x_order]
        p = np.array(precisions)[x_order]
        plt_int = int(hash(struct_str) % 256)
        plt.figure(plt_int)
        plt.plot(x, y, '-', label="f1")
        plt.plot(x, r, '-', label="recall")
        plt.plot(x, p, '-', label="precision")
        plt.legend()
        plt.xlabel("Distance threshold")
        plt.ylabel("Score")
        plt.title(f"{model_name}_{dataset_name}_{struct_str}")
        plt.savefig(save_path / f"{model_name}_{dataset_name}_{struct_str}.png", facecolor='w', bbox_inches='tight')
        temp_figure = plt.gcf()
        pickle.dump(temp_figure, open(save_path / f"{model_name}_{dataset_name}_{struct_str}.pkl", 'wb'))
        plt.clf()

        # save figure to collective f1 plot (1) if needed
        x = np.array([step['params']['threshold'] for step in optimizer.res])
        y = np.array([step['target'] for step in optimizer.res])
        x_order = np.argsort(x)
        x = x[x_order]
        y = y[x_order]
        plt.figure(1)
        plt.plot(x, y, 'x-', label=struct_str)

    best_threshold = optimizer.max['params']['threshold']
    best_accuracy, best_precision, best_recall, best_f1 = get_class_metrics(distances, easy_answers, hard_answers, best_threshold)

    return best_threshold, best_accuracy, best_precision, best_recall, best_f1


@torch.no_grad()
def find_val_thresholds(
    data_loader: torch.utils.data.DataLoader[QueryGraphBatch],
    model: QueryEmbeddingModel,
    similarity: Similarity,
    dataset: str,
    save_path: pathlib.Path,
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
    all_easy_answers = torch.empty((0, model.x_e.size(0)))
    all_hard_answers = torch.empty((0, model.x_e.size(0)))

    batch: QueryGraphBatch
    for batch in tqdm(data_loader, desc="Evaluation", unit="batch", unit_scale=True):
        
        # embed query
        x_query = model(batch)
        
        # compute pairwise similarity to all entities, shape: (batch_size, num_entities)
        scores = similarity(x=x_query, y=model.x_e)
        
        # get easy answers
        easy_answers = torch.zeros_like(scores) # initialize with zeros, size: (batch_size, num_ents)
        batch_id, entity_id = batch.easy_targets
        for batch_id, entity_id in zip(batch_id, entity_id):
            easy_answers[batch_id, entity_id] = 1
        
        # get hard answers
        hard_answers = torch.zeros_like(scores) # initialize with zeros, size: (batch_size, num_ents)
        batch_id, entity_id = batch.hard_targets
        hard_answers[batch_id, entity_id] = 1
        
        # add to tracking
        all_query_stuctures.extend(batch.query_structures)
        all_distances = torch.cat((all_distances, scores.cpu()), dim=0)
        all_easy_answers = torch.cat((all_easy_answers, easy_answers.cpu()), dim=0)
        all_hard_answers = torch.cat((all_hard_answers, hard_answers.cpu()), dim=0)

        if step % 10 == 0:
            logging.info('Gathering predictions of batches... (%d/%d) ' % (step, total_steps))
            step += 1

    # Define plot
    plt.figure(1, figsize=(10,10))

    Path(save_path).mkdir(parents=True, exist_ok=True)
    torch.save(all_distances, save_path / f"{dataset}_distances.pt", pickle_protocol=HIGHEST_PROTOCOL)
    torch.save(all_easy_answers, save_path / f"{dataset}_easy_answers_mask.pt", pickle_protocol=HIGHEST_PROTOCOL)
    torch.save(all_hard_answers, save_path / f"{dataset}_hard_answers_mask.pt", pickle_protocol=HIGHEST_PROTOCOL)
    torch.save(all_query_stuctures, save_path / f"{dataset}_query_structures.pt", pickle_protocol=HIGHEST_PROTOCOL)
    # all_distances = torch.load(save_path / f"{dataset}_distances.pt")
    # all_easy_answers = torch.load(save_path / f"{dataset}_easy_answers_mask.pt")
    # all_hard_answers = torch.load(save_path / f"{dataset}_hard_answers_mask.pt")
    # all_query_stuctures = torch.load(save_path / f"{dataset}_query_structures.pt")
        
    # find best threshold for each query structure
    for struct in set(all_query_stuctures):
        logging.info(f"Finding best threshold for structure: {struct}")

        # select data for current structure
        struct_idx = torch.tensor(np.where(np.array(all_query_stuctures) == struct)[0])
        str_distances = all_distances[struct_idx, :]
        str_easy_answers = all_easy_answers[struct_idx, :]
        str_hard_answers = all_hard_answers[struct_idx, :]

        # find best threshold and metrics
        best_threshold, accuracy, precision, recall, f1 = find_best_threshold(
            -str_distances.numpy(), 
            str_easy_answers.bool().numpy(), 
            str_hard_answers.bool().numpy(),
            save_path=save_path,
            num_steps=50,
            struct_str=struct,
            dataset_name=dataset,
        )

        logging.info(f"Threshold: {best_threshold:.4f}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        # save threshold and metrics
        thresholds[struct] = best_threshold
        metrics[struct] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Save figure
    plt.figure(1)
    plt.title("Optimization results")
    plt.xlabel('Distance threshold')
    plt.ylabel('f1-score')
    plt.legend()
    plt.savefig(save_path / f"threshold_search_{dataset}.png", facecolor='w', bbox_inches='tight')
    opt_figure = plt.gcf()
    pickle.dump(opt_figure, open(save_path / f"threshold_search_{dataset}.pkl", 'wb'))
    plt.clf()

    return thresholds, metrics


@torch.no_grad()
def evaluate_with_thresholds(
    data_loader: torch.utils.data.DataLoader[QueryGraphBatch],
    model: QueryEmbeddingModel,
    similarity: Similarity,
    thresholds: Dict[str, float],
):
    model.eval()
    
    # tracking scores
    metrics = {}

    step = 0
    total_steps = len(data_loader)

    # track queries, distances and answers
    all_query_stuctures = []
    all_distances = torch.empty((0, model.x_e.size(0)))
    all_easy_answers = torch.empty((0, model.x_e.size(0)))
    all_hard_answers = torch.empty((0, model.x_e.size(0)))
    
    batch: QueryGraphBatch
    for batch in tqdm(data_loader, desc="Evaluation", unit="batch", unit_scale=True):
        
        # embed query
        x_query = model(batch)
        
        # compute pairwise similarity to all entities, shape: (batch_size, num_entities)
        scores = similarity(x=x_query, y=model.x_e)
        
        # get easy answers
        easy_answers = torch.zeros_like(scores) # initialize with zeros, size: (batch_size, num_ents)
        batch_id, entity_id = batch.easy_targets
        for batch_id, entity_id in zip(batch_id, entity_id):
            easy_answers[batch_id, entity_id] = 1
        
        # get hard answers
        hard_answers = torch.zeros_like(scores) # initialize with zeros, size: (batch_size, num_ents)
        batch_id, entity_id = batch.hard_targets
        hard_answers[batch_id, entity_id] = 1
        
        # add to tracking
        all_query_stuctures.extend(batch.query_structures)
        all_distances = torch.cat((all_distances, scores.cpu()), dim=0)
        all_easy_answers = torch.cat((all_easy_answers, easy_answers.cpu()), dim=0)
        all_hard_answers = torch.cat((all_hard_answers, hard_answers.cpu()), dim=0)

        if step % 10 == 0:
            logging.info('Gathering predictions of batches... (%d/%d) ' % (step, total_steps))
            step += 1
        
    struct_sizes = {}
    # find best threshold for each query structure
    for struct in set(all_query_stuctures):
        logging.info(f"Calculating metrics for structure: {struct}")

        # select data for current structure
        struct_idx = torch.tensor(np.where(np.array(all_query_stuctures) == struct)[0])
        str_distances = all_distances[struct_idx, :]
        str_easy_answers = all_easy_answers[struct_idx, :]
        str_hard_answers = all_hard_answers[struct_idx, :]

        # track size of current structure for weighted metrics
        struct_sizes[struct] = len(struct_idx)

        # find best threshold and metrics
        accuracy, precision, recall, f1 = get_class_metrics(
            -str_distances.numpy(), 
            str_easy_answers.bool().numpy(), 
            str_hard_answers.bool().numpy(),
            thresholds[struct]
        )

        # save threshold and metrics
        metrics[struct] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    metrics['macro'] = {
        'accuracy': np.mean([metrics[struct]['accuracy'] for struct in metrics]),
        'precision': np.mean([metrics[struct]['precision'] for struct in metrics]),
        'recall': np.mean([metrics[struct]['recall'] for struct in metrics]),
        'f1': np.mean([metrics[struct]['f1'] for struct in metrics])
    }

    metrics['weighted'] = {
        'accuracy': np.average(
            [metrics[struct]['accuracy'] for struct in metrics if struct != 'macro'],
            weights=[struct_sizes[struct] for struct in metrics if struct != 'macro']
        ),
        'precision': np.average(
            [metrics[struct]['precision'] for struct in metrics if struct != 'macro'],
            weights=[struct_sizes[struct] for struct in metrics if struct != 'macro']
        ),
        'recall': np.average(
            [metrics[struct]['recall'] for struct in metrics if struct != 'macro'],
            weights=[struct_sizes[struct] for struct in metrics if struct != 'macro']
        ),
        'f1': np.average(
            [metrics[struct]['f1'] for struct in metrics if struct != 'macro'],
            weights=[struct_sizes[struct] for struct in metrics if struct != 'macro']
        )
    }

    return metrics
