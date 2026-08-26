import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.ensemble import RandomForestRegressor

def predict_tree(tree_df: pd.DataFrame, X: np.ndarray) -> np.ndarray:

    n_samples = X.shape[0]

    features    = tree_df['Feature'].fillna(-1).to_numpy(dtype=int)
    cuts        = tree_df['Split'].to_numpy()
    values      = tree_df['Value'].to_numpy()
    left_child  = tree_df['Yes'].to_numpy()
    right_child = tree_df['No'].to_numpy()
    is_leaf     = tree_df['Is_leaf'].to_numpy()

    current_nodes = np.zeros(n_samples, dtype=int)
    done_mask = np.zeros(n_samples, dtype=bool)

    while not np.all(done_mask):
        active = ~done_mask
        node_idxs = current_nodes[active]

        leaves = is_leaf[node_idxs]
        done_mask[active] = leaves

        move = ~leaves
        if np.any(move):
            move_idx = np.where(active)[0][move]
            node_move = node_idxs[move]
            feat = features[node_move]
            thr  = cuts[node_move]
            x_vals = X[move_idx, feat]
            left_mask = x_vals <= thr
            right_mask = ~left_mask
            current_nodes[move_idx[left_mask]]  = left_child[node_move[left_mask]].astype(int)
            current_nodes[move_idx[right_mask]] = right_child[node_move[right_mask]].astype(int)

    preds = values[current_nodes]
    return preds

def process_tree_df(forest_df):
    """
    Convert forest_df directly to the sklearn-style dataframe format.
    """
    df = forest_df.reset_index(drop=True)

    is_leaf = df["Is_leaf"].to_numpy()

    result = pd.DataFrame({
        "Tree": df["Tree"],
        "Node": range(len(df)),
        "Feature": df.apply(lambda row: "Leaf" if row["Is_leaf"] else f"X{int(row['Feature'])}", axis=1),
        "Split": np.where(is_leaf, np.nan, df["Split"]),
        "Yes": df.apply(lambda row: np.nan if row["Is_leaf"] else f"0-{int(row['Yes'])}", axis=1),
        "No":  df.apply(lambda row: np.nan if row["Is_leaf"] else f"0-{int(row['No'])}", axis=1),
        "Value": np.where(is_leaf, df["Value"], np.nan),
    })

    return result

def build_random_forest_from_df(forest_df, n_features):
    """
    Build a sklearn RandomForestRegressor from a full forest dataframe
    """
    rf = RandomForestRegressor(n_estimators=forest_df['Tree'].nunique())
    trees = []
    for _, tree_df_group in forest_df.groupby('Tree', sort=True):
        dt = build_tree_from_df(tree_df_group.reset_index(drop=True), n_features)
        trees.append(dt)
    
    rf.estimators_ = trees
    rf.n_estimators = len(trees)
    rf.n_features_in_ = n_features
    rf.feature_names_in_ = np.array([f"X{i}" for i in range(n_features)])
    return rf

def build_tree_from_df(tree_df, n_features):

    n_nodes = len(tree_df)

    children_left = np.full(n_nodes, -1, dtype=np.int64)
    children_right = np.full(n_nodes, -1, dtype=np.int64)
    feature = np.full(n_nodes, _tree.TREE_UNDEFINED, dtype=np.int64)
    threshold = np.zeros(n_nodes, dtype=np.float64)
    value = np.zeros((n_nodes, 1, 1), dtype=np.float64)

    is_leaf = tree_df["Is_leaf"].to_numpy()
    feat_col = tree_df["Feature"].to_numpy()
    split_col = tree_df["Split"].to_numpy()
    yes_col = tree_df["Yes"].to_numpy()
    no_col = tree_df["No"].to_numpy()
    tau_col = tree_df["Value"].to_numpy()

    internal = ~is_leaf

    feature[internal] = feat_col[internal].astype(np.int64)
    children_left[internal] = yes_col[internal].astype(np.int64)
    children_right[internal] = no_col[internal].astype(np.int64)
    threshold[internal] = split_col[internal]
    value[is_leaf, 0, 0] = tau_col[is_leaf]

    n_node_samples = (
        tree_df["n_samples"]
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )

    weighted_n_node_samples = n_node_samples.astype(np.float64)

    impurity = np.zeros(n_nodes, dtype=np.float64)
    missing = np.zeros(n_nodes, dtype=np.int64)

    nodes_array = np.array(
        list(
            zip(
                children_left,
                children_right,
                feature,
                threshold,
                impurity,
                n_node_samples,
                weighted_n_node_samples,
                missing,
            )
        ),
        dtype=_tree.NODE_DTYPE,
    )

    tree = _tree.Tree(
        n_features=n_features,
        n_classes=np.array([1]),
        n_outputs=1,
    )

    tree.__setstate__(
        {
            "max_depth": int(tree_df["Node"].max()),
            "node_count": n_nodes,
            "nodes": nodes_array,
            "values": value,
        }
    )

    dt = DecisionTreeRegressor()
    dt.tree_ = tree

    return dt