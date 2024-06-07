import cupy as cp
import numpy as np
import pandas as pd
import os
from cuml.neighbors import NearestNeighbors, kneighbors_graph

from constants import Column, NTC, VAL, TEST, TRAIN


def normalize_by_NTC(df, fnames, metadata_names, use_std=True, use_median=False):
    NTC_df = df[df[Column.gene.value] == NTC]
    NTC_features = NTC_df[fnames].values
    NTC_features_gpu = cp.asarray(NTC_features)
    values_gpu = cp.asarray(df[fnames].values)
    if not use_median:
        if use_std:
            values_scaled = (values_gpu - NTC_features_gpu.mean(axis=0)) / NTC_features_gpu.std(axis=0)
        else:
            values_scaled = values_gpu - NTC_features_gpu.mean(axis=0)
    else:
        median = cp.median(values_gpu, axis=0)
        if use_std:
            mad = cp.median(cp.absolute(values_gpu - median), axis=0)
            values_scaled = (values_gpu - median) / mad
        else:
            values_scaled = values_gpu - median
    df_features = pd.DataFrame(values_scaled.get(), columns=fnames)
    df = pd.concat([df[metadata_names].reset_index(drop=True), df_features], axis=1)
    return df


def standardize_per_catX(df, feature_names, metadata_names, use_std=True, use_median=False):
    df_group = df.groupby([Column.plate.name, Column.well.name], group_keys=False)
    df_scaled = df_group.apply(lambda x: normalize_by_NTC(x, feature_names, metadata_names, use_std, use_median))
    return df_scaled


def one_hot_encoding(num_classes, class_vector):
    return np.eye(num_classes)[class_vector].astype(bool)


def KNN_classifier(df, feature_names, class_type, ntc_mean=None, ntc_std=None, k=20,
                   temperature=1, metric='cosine', use_float16=False, return_all=False):
    labels = pd.factorize(df[class_type])[0].astype(np.int16)
    num_classes = df[class_type].nunique()
    label_arr = one_hot_encoding(num_classes, labels)

    centroids = df[feature_names].values
    if use_float16:
        centroids = centroids.astype(np.float16)
    X_gpu = cp.asarray(centroids)

    if ntc_mean is not None and ntc_std is not None:
        ntc_mean_gpu, ntc_std_gpu = cp.asarray(ntc_mean), cp.asarray(ntc_std)
        X_gpu = (X_gpu - ntc_mean_gpu[cp.newaxis, :]) / ntc_std_gpu[cp.newaxis, :]

    y_arr_gpu = cp.asarray(label_arr)
    y_gpu = cp.asarray(labels)
    model = NearestNeighbors(n_neighbors=k, metric=metric)
    model.fit(X_gpu)
    csr_mat = model.kneighbors_graph(
        X=None, n_neighbors=None,
        mode="distance",
    )

    indice_mat = csr_mat.indices.reshape([-1, k])
    dist_mat = csr_mat.data.reshape([-1, k])
    if use_float16:
        dist_mat = dist_mat.astype(np.float16)
    distances_transform = cp.exp(dist_mat / temperature)
    if use_float16:
        distances_transform = distances_transform.astype(np.float16)

    retrieved_neighbors = cp.take(y_arr_gpu, indice_mat.get(), axis=0)
    probs = cp.sum(retrieved_neighbors * distances_transform[:, :, cp.newaxis], axis=1)
    predictions = cp.argsort(probs, axis=1)[:, ::-1]
    correct = predictions == y_gpu[:, cp.newaxis]
    top1_acc = cp.sum(correct[:, 0:1]) / correct.shape[0]
    top5_acc = cp.sum(correct[:, 0:5]) / correct.shape[0]

    if return_all:
        return float(top1_acc.get()), float(top5_acc.get()), num_classes, correct[:, 0:1].get()
    else:
        return float(top1_acc.get()), float(top5_acc.get()), num_classes


def standardize(df, feature_names, metadata_names):
    values_gpu = cp.asarray(df[feature_names])
    values_scaled = (values_gpu - values_gpu.mean(axis=0)) / values_gpu.std(axis=0)
    df_features = pd.DataFrame(values_scaled.get(), columns=feature_names)
    df = pd.concat([df[metadata_names].reset_index(drop=True), df_features], axis=1)
    return df


def sampled_df_by_dataset(df, data_set):
    if data_set == 'val':
        sel_list = VAL
    elif data_set == 'test':
        sel_list = TEST
    elif data_set == 'train':
        sel_list = TRAIN
    elif isinstance(data_set, list):
        sel_list = data_set
    else:
        raise ValueError

    df_group = df.groupby(['plate', 'well'])
    df_list = []
    for key, group in df_group:
        if key in sel_list:
            df_list.append(group)

    df_sampled = pd.concat(df_list)
    return df_sampled
