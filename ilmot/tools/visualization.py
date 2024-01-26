"""Visualization tools."""
from typing import Dict, Set
import copy
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.manifold import TSNE  # type: ignore


def visualize_embeddings(
    embeddings: npt.NDArray[np.float64],
    class_ids: npt.NDArray[np.int64],
    category_mapping: Dict[str, int],
    output_dir: str = "default",
    max_instance: int = 100,
) -> None:
    """Visualization of embeddings.

    Args:
        embeddings: embeddings of each detection
        class_ids: class ids of each detection
        track_ids: track ids of each detection
        category_mapping: A mapping from category names to their ids.
        output_dir: output image directory
        max_instance: number of instances to visualize

    """
    id_mask: npt.NDArray[np.bool8] = np.zeros(class_ids.shape, dtype=np.bool8)

    arr = np.arange(embeddings.shape[0])
    np.random.shuffle(arr)
    embeddings = embeddings[arr]
    class_ids = class_ids[arr]

    class_count: Dict[int, int] = {cat: 0 for cat in category_mapping.values()}
    class_count[-1] = 0
    category_mapping_reverse = {v: k for k, v in category_mapping.items()}

    for i, class_id in enumerate(class_ids):
        if class_count[class_id] < max_instance:
            id_mask[i] = 1
            class_count[class_id] += 1

    class_legends = []
    for id in np.unique(class_ids):
        if id == -1:
            class_legends.append("background")
        else:
            class_legends.append(category_mapping_reverse[id])

    class_ids = class_ids[id_mask]
    embeddings = embeddings[id_mask]

    size = copy.deepcopy(class_ids)

    size[class_ids > -2] = 3

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(embeddings)

    df = pd.DataFrame()
    df["class_ids"] = class_ids
    df["size"] = size
    df["comp-1"] = z[:, 0] * 2
    df["comp-2"] = z[:, 1]

    plt.figure()
    class_plot = sns.scatterplot(
        x="comp-1",
        y="comp-2",
        hue=df.class_ids.tolist(),
        palette=sns.color_palette(
            "hls", len(np.unique(class_ids))  # type: ignore
        ),
        data=df,
        size="size",
        style="size",
    )
    class_plot.get_legend().remove()
    handles, _ = class_plot.get_legend_handles_labels()
    class_plot.legend(handles, class_legends)
    class_plot.set(title="Class Embeddings T-SNE projection")
    plt.axis("equal")
    plt.axis("off")
    plt.savefig(f"{output_dir}/class_embeddings.png", dpi=2000)
