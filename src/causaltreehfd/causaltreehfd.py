import numpy as np
import pandas as pd
from treehfd.tree import TreeHFD

from causaltreehfd.utils import predict_tree, process_tree_df

class CausalTreeHFD:
    """
    CausalTreeHFD of a fitted causal forest model.
    """

    def __init__(self, causal_forest_df: pd.DataFrame, depth_variable: tuple[int, int], max_depth: int = 10) -> None:

        self.forest_structure = causal_forest_df
        self.n_trees = causal_forest_df['Tree'].nunique()
        self.max_depth = max_depth
        self.depth_variable = depth_variable
        self.treehfd_list: list[TreeHFD] = []
        self.tau0: float = 0.0
        
    def fit(self, X: np.ndarray) -> None:

        interaction_list_raw: list[list[list[int]]] = []

        for tree_idx in range(self.n_trees):

            tree_df = self.forest_structure[self.forest_structure['Tree'] == tree_idx]
            preds = predict_tree(tree_df, X)

            treehfd = TreeHFD(
                process_tree_df(tree_df),
                max_depth=self.max_depth,
                depth_variable=self.depth_variable,
                interaction_list=None,
                interaction_order=2
            )
            treehfd.fit(X, preds)

            self.tau0 += treehfd.eta0/self.n_trees
            self.treehfd_list.append(treehfd)
            interaction_list_raw.append(treehfd.interaction_list)

        interaction_list_raw = [x for x in interaction_list_raw if len(x) > 0]
        if len(interaction_list_raw) > 0:
            self.interaction_list = np.unique(np.concatenate(
                                        interaction_list_raw, axis=0), axis=0)

    def predict(self, X: np.ndarray) -> tuple:

        tau_main = np.zeros(shape=(X.shape[0], X.shape[1]))
        tau_order2 = np.zeros((X.shape[0], self.interaction_list.shape[0]))

        for treehfd in self.treehfd_list:
            tau_main_tree, tau_order2_tree = treehfd.predict(X)
            main_variables = treehfd.cartesian_partition.main_variables
            tau_main[:, main_variables] += tau_main_tree/self.n_trees

            interaction_index = []
            for interaction in treehfd.interaction_list:
                idx = np.where(np.all(self.interaction_list == interaction,
                                    axis=1))[0].tolist()
                interaction_index += idx
            tau_order2[:, interaction_index] += tau_order2_tree/self.n_trees

        return (tau_main, tau_order2)
    