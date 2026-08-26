from rpy2.robjects import r
from rpy2.robjects import r, numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
import numpy as np

r("library(grf)")

def fit_grf(
        X: np.ndarray,
        Y: np.ndarray,
        W: np.ndarray,
        num_trees = 500,
        min_node_size = 5,
        clusters: np.ndarray | None = None,
        seed: int = 1234
    ) -> tuple:

    with localconverter(numpy2ri.converter):
        r_X = numpy2ri.py2rpy(X)
        r_Y = numpy2ri.py2rpy(Y)
        r_W = numpy2ri.py2rpy(W)

    r.assign("X", r_X)
    r.assign("Y", r_Y)
    r.assign("W", r_W)

    if clusters is not None:
        with localconverter(numpy2ri.converter):
            r_clusters = numpy2ri.py2rpy(clusters)
        r.assign("clusters", r_clusters)
        clusters_arg = "clusters = clusters"
    else:
        clusters_arg = "clusters = NULL"

    r.assign("seed", seed)

    r(f"""
    library(grf)
    source("src/save_causal_forests.R")
      
    X <- as.data.frame(X)
    Y <- as.numeric(Y)
    W <- as.numeric(W)

    cf <- causal_forest(
        X,
        Y,
        W,
        num.trees = {num_trees},
        min.node.size = {min_node_size},
        {clusters_arg},
        num.threads=1,
        seed=seed
    )

    tau_hat <- predict(cf)$predictions
    forest_df <- grf_forest_to_df(cf)
    """)

    forest_df = pandas2ri.rpy2py(r["forest_df"])
    tau_hat = np.array(r["tau_hat"])

    return (forest_df, tau_hat)