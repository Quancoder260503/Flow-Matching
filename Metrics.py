import argparse 
import io 
import os 
import random 
import warnings 
import zipfile 
from typing import Iterable, Optional, Tuple

import numpy as np
import requests
import tensorflow.compat.v1 as tf
from scipy import linalg
from tqdm.auto import tqdm

INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
INCEPTION_V3_PATH = "classify_image_graph_def.pb"


def frechet_distance(mu_1, sigma_1, mu_2, sigma_2, eps = 1e-6):
  mu_1 = np.atleast_1d(mu_1) 
  mu_2 = np.atleast_1d(mu_2)

  sigma_1 = np.atleast_2d(sigma_1)
  sigma_2 = np.atleast_2d(sigma_2)

  assert mu_1.shape == mu_2.shape, \
    "Ground truth and Sample mean vectors have different dimensions"
  assert sigma_1.shape == sigma_2.shape, \
    "Ground truth and Sample covariance matrices have different dimensions"
  
  dist = np.sum((mu_1 - mu_2) ** 2)

  covmean, _ = linalg.sqrtm(sigma_1.dot(sigma_2), disp = False)
  if not np.isfinite(covmean).all():
    messasge = (
      "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
    )
    warnings.warn(messasge)
    offset = np.eye(sigma_1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma_1 + offset).dot(sigma_2 + offset))

  return dist + np.trace(sigma_1) + np.trace(sigma_2) - 2 * np.trace(covmean)


class Evaluator(object):
  def __init__(
    self, 
    session, 
    batch_size, 
    softmax_batch_size, 
  ):
    self.batch_size = batch_size 
    self.manifold_estimator = ManifoldEstimator()
    self.session = session
  

  def read_activations(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open_npz_array(npz_path, "arr_0") as reader:
      return self.compute_activations(reader.read_batches(self.batch_size))
    
  def read_statistics(
    self, npz_path: str, activations: Tuple[np.ndarray, np.ndarray]
  ) -> Tuple[FIDStatistics, FIDStatistics]:
    obj = np.load(npz_path)
    if "mu" in list(obj.keys()):
      return FIDStatistics(obj["mu"], obj["sigma"]), FIDStatistics(
        obj["mu_s"], obj["sigma_s"]
      )
    return tuple(self.compute_statistics(x) for x in activations)  

  def compute_activations(self, batches : Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    activations = [] 
    spatial_activations = []

    for batch in tqdm(batches, desc = "Computing activations"):
      batch = batch.astype(np.float32)
      batch_activations, batch_spatial = self.session.run(
        [self.pool_features, self.spatial_features], 
        {self.image_input : batch}, 
      )
      activations.append(batch_activations.reshape([batch_activations.shape[0], -1]))
      spatial_activations.append(batch_spatial.reshape([batch_spatial.shape[0], -1]))

    return (
      np.concatenate(activations, axis = 0), 
      np.concatenate(spatial_activations, axis = 0), 
    )
  
  def compute_stastics(self, activations_a : np.ndarray, activations_b : np.ndarray) -> float : 
    mu_a = np.mean(activations_a, axis = 0)
    sigma_a = np.cov(activations_a, rowvar = False)

    mu_b = np.mean(activations_b, axis = 0)
    sigma_b = np.cov(activations_b, rowvar = False)

    return frechet_distance(mu_a, sigma_a, mu_b, sigma_b)


class ManifoldEstimator(object):
  pass


class DistanceBlock(object):
  def __init__(self, session): 
    self.session = session 

    with session.graph.as_default(): 
      self.features_batch_1 = tf.placeholder(tf.float32, shape = [None, None])
      self.features_batch_2 = tf.placeholder(tf.float32, shape = [None, None])


def batch_pairwise_distance(batch_a, batch_b):
  with tf.variable_scope('pairwise_dist_block'):
    norm_a = tf.reduce_sum(tf.square(batch_a), 1)
    norm_b = tf.reduce_sum(tf.square(batch_b), 1)

    norm_a = tf.reshape(norm_a, [-1, 1])
    norm_b = tf.reshape(norm_b, [1, -1])

    dist = tf.maximum(norm_a - 2 * tf.matmul(norm_a, norm_b, False, True) + norm_b, 0.)
  return dist 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("references_batch", help = "path to reference ground truth batch npz file")
    parser.add_argument("sample_batch",     help = "path to sample batch npz file")
    args = parser.parse_args()

    config = tf.ConfigProto(
        allow_soft_placement = True  # allows DecodeJpeg to run on CPU in Inception graph
    )
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config = config))

    print("warming up TensorFlow...")
    evaluator.warmup()

    print("computing reference batch activations...")
    ref_acts = evaluator.read_activations(args.ref_batch)
    print("computing/reading reference batch statistics...")
    ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)

    print("computing sample batch activations...")
    sample_acts = evaluator.read_activations(args.sample_batch)
    print("computing/reading sample batch statistics...")
    sample_stats, sample_stats_spatial = evaluator.read_statistics(args.sample_batch, sample_acts)

    print("Computing evaluations...")
    print("Inception Score:", evaluator.compute_inception_score(sample_acts[0]))
    print("FID:", sample_stats.frechet_distance(ref_stats))
    print("sFID:", sample_stats_spatial.frechet_distance(ref_stats_spatial))
    prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
    print("Precision:", prec)
    print("Recall:", recall)