suppressPackageStartupMessages({library(tidyverse)})


grf_tree_to_df <- function(tree, tree_index, Y_resid, W_resid) {
  nodes <- tree$nodes
  n <- length(nodes)

  Tree <- rep.int(tree_index, n)
  Node <- seq_len(n)

  Is_leaf <- logical(n)
  Feature <- rep(NA_integer_, n)
  Split <- rep(NA_real_, n)
  Yes <- rep(NA_integer_, n)
  No <- rep(NA_integer_, n)
  Value <- rep(NA_real_, n)
  num <- rep(NA_real_, n)
  den <- rep(NA_real_, n)
  n_samples <- integer(n)

  for (i in seq_len(n)) {
    node <- nodes[[i]]

    if (!node$is_leaf) {
      Is_leaf[i] <- FALSE
      Feature[i] <- node$split_variable - 1
      Split[i] <- node$split_value
      Yes[i] <- node$left_child - 1
      No[i] <- node$right_child - 1
    } else {
      Is_leaf[i] <- TRUE

      samples <- node$samples
      n_samples[i] <- length(samples)

      if (length(samples) > 0) {
        Y <- Y_resid[samples]
        W <- W_resid[samples]

        num_i <- mean(Y * W)
        den_i <- mean(W * W)

        Value[i] <- num_i / den_i
      }
    }
  }

  data.frame(
    Tree,
    Node,
    Is_leaf,
    Feature,
    Split,
    Yes,
    No,
    Value,
    n_samples,
    check.names = FALSE
  )
}

grf_forest_to_df <- function(cf) {
  num_trees <- cf$`_num_trees`
  Y_resid <- cf$Y.orig - cf$Y.hat
  W_resid <- cf$W.orig - cf$W.hat
  
  all_trees_df <- data.table::rbindlist(
  lapply(seq_len(num_trees), function(t) {
    grf_tree_to_df(
      get_tree(cf, t),
      t,
      Y_resid,
      W_resid
    )
  })
)

  all_trees_df$Split <- all_trees_df$Split + 1e-12
  all_trees_df$Tree <- all_trees_df$Tree - 1
  
  return(all_trees_df)
}