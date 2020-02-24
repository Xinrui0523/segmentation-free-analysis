def spherical_kmeans(self, 
                         n_clusters = 10):
        self.n_clusters = n_clusters
        self.model = SphericalKMeans(n_clusters)
        

    def spectralClustering(self,
                           n_clusters = 10,
                           eigen_solver = None, 
                           random_state = None, 
                           n_init = 10,
                           gamma = 1,
                           affinity = 'nearest_neighbors',
                           n_neighbors = 10,
                           eigen_tol = 0.0,
                           assign_labels = 'discretize',
                           degree = 3,
                           coef0 = 1,
                           kernel_parmas = None, 
                           n_jobs = None,
                           ):
        """
        assign_labels: "discretize" | "kmeans"
        """
        self.n_clusters = n_clusters
        self.model = SpectralClustering(n_clusters=n_clusters, 
                                        affinity = 'nearest_neighbors', 
                                        random_state = random_state)




def _runPCA(self,
                n_components=0.95,
                svd_solver = 'auto',
                ):
        self.pca = PCA(n_components=n_components, 
                       svd_solver = svd_solver)
        self.pca.fit(self.data_pixel[self.genes])
    
    def _plot_cum_explained_var(self,
                                savedir,
                                savename,
                                dpi=300):
        
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');
        plt.savefig(os.path.join(savedir, savename), 
                    dpi=dpi)
        plt.close()
        
    def plot_pc1_pc2(self,
                     savedir, 
                     savename,
                     dpi=300):
        projected = self.pca_.transform(self.train_data)
        plt.scatter(projected[:,0], projected[:,1],
                    cmap=plt.cm.get_cmap('spectral', 10),
                    )
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.colorbar()
        plt.savefig(os.path.join(savedir, savename),
                    dpi=dpi)

        
        
    def _customizePCA(self, metric = "cosine"):
        
        assert metric in ["cosine", "covariance"]
        if metric == "cosine":
            self._cosince_dist()
        elif metric == "covariance":
            self._cov_dist()
        else:
            warnings.warn(f"NOT IMPLEMENTED METRICS\n"
                          f"Using Covariance as default")
            self._cov_dist()
            
        a, b, c = np.linalg.svd(self.gg_matrix)
        pc = np.matmul(self.data_pixel[self.genes].values, a[:,:3])
        self.data_pixel["pc1"] = pc[:,0]
        
        max_min = np.percentile(self.data_pixel["pc1"], 99) - np.percentile(self.data_pixel["pc1"], 1)
        
        self.data_pixel["pc1"] = (self.data_pixel["pc1"] - np.percentile(self.data_pixel["pc1"], 1)) / max_min
        self.data_pixel["pc1"] = np.clip(self.data_pixel["pc1"], 0, 1)
        
    def _cosince_dist(self):
        self.gg_matrix = cosine(self.gene_pixel, self.gene_pixel)
    
    def _cov_dist(self):
        self.gg_matrix = np.cov(self.gene_pixel)
        
    # Clustering methods ----------------------------
    def createModel(self, 
                    model_name = "kmeans",
                    n_clusters = 10,
                    **kwargs):
        assert model_name in implemented_models
        self.n_clusters = n_clusters
        self.model_name = model_name
        
        if model_name == 'kmeans':
            self.model = KMeans(n_clusters = n_clusters,
                                **kwargs)
        elif model_name == 'sphericalKMeans':
            self.model = SphericalKMeans(n_clusters = n_clusters,
                                         **kwargs)
        elif model_name == 'spectral':
            self.model = SpectralClustering(n_clusters = n_clusters,
                                            **kwargs)
        
    def trainModel(self):
        print(f"Training model {self.model_name}...")
        x_tr = self.train_data[self.genes]
        self.model.fit(x_tr)
        y_tr = self.model.labels_
                
        print(f"Predicting labels ...")
        clf = LinearSVC().fit(x_tr, y_tr)
        self.data_pixel["pred"] = clf.predict(self.data_pixel[self.genes])
        
        self.data_pixel["cluster"]=np.zeros((len(self.data_pixel["pred"]==0),1))
        
        
        print("--- Calculating silhouette scores ---")
        self.data_train_pred_labels = self.data_pixel.iloc[::self.train_idx]["cluster"]
        
        self.silhouette_vals = silhouette_samples(x_tr, y_tr, metric='euclidean')
        
        # spectral
        # self.affinity_matrix = self.model.affinity_matrix_
                
        # kmeans
        self.centroids = self.model.cluster_centers_
        self.model_inertia_ = self.model.inertia_
    
    def _runCustomizePCA(self):
        print(f"Getting correlation of each cluster ...")
        self._customizePCA(metric = "covariance")
        
        temp = self.data_pixel[["pred","pc1"]].groupby("pred").apply(np.mean).reset_index(drop=1)
        temp.rename(columns={"pc1":"pc1_mean"},inplace=True)
        self.data_pixel = pd.merge(self.data_pixel,temp,how="left",on="pred")
        
        uq =  np.unique(self.data_pixel.pc1_mean)
        for i in range(len(uq)):
           self.data_pixel.loc[self.data_pixel["pc1_mean"]==uq[i],"cluster"] = (i+1)%self.n_clusters
    
    