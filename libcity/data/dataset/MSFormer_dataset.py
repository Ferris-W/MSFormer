import os
import numpy as np
from libcity.data.dataset import TrafficStatePointDataset
from libcity.data.utils import generate_dataloader
from tslearn.clustering import TimeSeriesKMeans, KShape
from sklearn.mixture import GaussianMixture
import networkx as nx
import pymetis

class MSFormerDataset(TrafficStatePointDataset):

    def __init__(self, config):
        self.type_short_path = config.get('type_short_path', 'hop')
        super().__init__(config)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'pdformer_point_based_{}.npz'.format(self.parameters_str))
        self.points_per_hour = 3600 // self.time_intervals  # 12
        self.sem_clus_num = config.get("sem_clus_num", 8)
        self.geo_clus_num = config.get("geo_clus_num", 8)
        self.sem_clus_proj = self._get_sem_clus_proj()
        self.geo_clus_proj = self._get_geo_clus_proj()
        self.points_per_day = 24 * 3600 // self.time_intervals
        self.cand_key_days = config.get("cand_key_days", 14)
        self.s_attn_size = config.get("s_attn_size", 3)
        self.n_cluster = config.get("n_cluster", 16)
        self.cluster_max_iter = config.get("cluster_max_iter", 5)
        self.cluster_method = config.get("cluster_method", "kshape")

    def _get_sem_clus_proj(self):
        self._logger.info('Loading semantic clusters......')
        for ind, filename in enumerate(self.data_files):
            if ind == 0:
                df = self._load_dyna(filename)
            else:
                df = np.concatenate((df, self._load_dyna(filename)), axis=0)
        # df.shape 17856, 170, 1
        train_data = df[:int(df.shape[0] * self.train_rate)].squeeze().T  # 170 10713
        features = np.column_stack(
            [np.mean(train_data, axis=-1), np.median(train_data, axis=-1), np.std(train_data, axis=-1)])
        gmm = GaussianMixture(n_components=self.sem_clus_num, covariance_type='full', max_iter=100)
        gmm.fit(features)
        sem_labels = gmm.predict(features)
        clus2node = {i: np.argwhere(sem_labels == i).squeeze() for i in range(self.sem_clus_num)}
        sem_clus_proj = {'c2n': clus2node,
                         'n2c': sem_labels}
        return sem_clus_proj

    def _get_geo_clus_proj(self):
        self._logger.info('Loading geographic clusters......')
        (edgecuts, parts) = pymetis.part_graph(self.geo_clus_num, self.adj_mx)
        parts = np.array(parts)
        clus2node = {i: np.argwhere(parts == i).squeeze() for i in range(self.geo_clus_num)}
        geo_clus_proj = {'c2n': clus2node,
                         'n2c': parts}
        return geo_clus_proj

    def _load_rel(self):
        self.sd_mx = None
        super()._load_rel()
        self._logger.info('Max adj_mx value = {}'.format(self.adj_mx.max()))
        self.sh_mx = self.adj_mx.copy()
        if self.type_short_path == 'hop':
            self.sh_mx[self.sh_mx > 0] = 1
            self.sh_mx[self.sh_mx == 0] = 511
            for i in range(self.num_nodes):
                self.sh_mx[i, i] = 0
            for k in range(self.num_nodes):
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        self.sh_mx[i, j] = min(self.sh_mx[i, j], self.sh_mx[i, k] + self.sh_mx[k, j], 511)
            np.save('{}.npy'.format(self.dataset), self.sh_mx)

    def _calculate_adjacency_matrix(self):
        self._logger.info("Start Calculate the weight by Gauss kernel!")
        self.sd_mx = self.adj_mx.copy()
        distances = self.adj_mx[~np.isinf(self.adj_mx)].flatten()
        std = distances.std()
        self.adj_mx = np.exp(-np.square(self.adj_mx / std))
        self.adj_mx[self.adj_mx < self.weight_adj_epsilon] = 0
        if self.type_short_path == 'dist':
            self.sd_mx[self.adj_mx == 0] = np.inf
            for k in range(self.num_nodes):
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        self.sd_mx[i, j] = min(self.sd_mx[i, j], self.sd_mx[i, k] + self.sd_mx[k, j])

    def get_data(self):
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test = self._load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test = self._generate_train_val_test()
        self.feature_dim = x_train.shape[-1]

        self.ext_dim = self.feature_dim - self.output_dim
        self.scaler = self._get_scalar(self.scaler_type,
                                       x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        self.ext_scaler = self._get_scalar(self.ext_scaler_type,
                                           x_train[..., self.output_dim:], y_train[..., self.output_dim:])
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        if self.normal_external:
            x_train[..., self.output_dim:] = self.ext_scaler.transform(x_train[..., self.output_dim:])
            y_train[..., self.output_dim:] = self.ext_scaler.transform(y_train[..., self.output_dim:])
            x_val[..., self.output_dim:] = self.ext_scaler.transform(x_val[..., self.output_dim:])
            y_val[..., self.output_dim:] = self.ext_scaler.transform(y_val[..., self.output_dim:])
            x_test[..., self.output_dim:] = self.ext_scaler.transform(x_test[..., self.output_dim:])
            y_test[..., self.output_dim:] = self.ext_scaler.transform(y_test[..., self.output_dim:])
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample,
                                distributed=self.distributed)
        self.num_batches = len(self.train_dataloader)

        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "sd_mx": self.sd_mx, "sh_mx": self.sh_mx,
                "ext_dim": self.ext_dim, "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches,
                "sem_clus_proj": self.sem_clus_proj, "sem_clus_num": self.sem_clus_num,
                "geo_clus_proj": self.geo_clus_proj}
