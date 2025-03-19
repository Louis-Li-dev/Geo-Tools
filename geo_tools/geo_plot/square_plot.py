import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
from typing import List, Optional, Union, Tuple


class plot_and_compare:

    plt.rcParams['font.family'] = 'Times New Roman'

    def __init__(self, results: List[Union[np.ndarray, List]], names: Optional[List[str]] = None, use_seaborn: bool = False):
        """
        Initialize plot_and_compare class.

        Args:
            results (List[Union[np.ndarray, List]]): List containing model predictions or results to compare.
            names (Optional[List[str]]): List of names corresponding to each result. Defaults to None.
            use_seaborn (bool): Whether to use seaborn for plotting (1D and 2D only). Defaults to False.
        """
        if not results:
            raise ValueError("Results list cannot be empty.")

        self.results = [np.array(r) if isinstance(r, list) else r for r in results]

        length_set = {len(r.flatten()) for r in self.results}
        if len(length_set) > 1:
            raise ValueError("All results must have the same number of elements.")

        self.num_results = len(results)
        if names is not None and len(names) != self.num_results:
            raise ValueError("The length of names must match the number of results.")

        self.names = names if names else [f'Result {i+1}' for i in range(self.num_results)]
        self.use_seaborn = use_seaborn

    def plot(self, figsize: Tuple[int, int] = (15, 5), cmap: str = 'viridis', adjust_params: Optional[dict] = None) -> None:
        """
        Plot the comparison of results.

        Args:
            figsize (Tuple[int, int]): Size of the figure. Defaults to (15, 5).
            cmap (str): Colormap used for 2D and 3D plots. Defaults to 'viridis'.
            adjust_params (Optional[dict]): Parameters for subplot adjustment. Defaults to None.
        """
        dim = self.results[0].ndim
        fig, axs = plt.subplots(1, self.num_results, figsize=figsize)

        if self.num_results == 1:
            axs = [axs]

        for idx, (ax, result, name) in enumerate(zip(axs, self.results, self.names)):
            ax.set_title(name)
            if dim == 1:
                if self.use_seaborn:
                    sns.lineplot(x=np.arange(len(result)), y=result, ax=ax)
                else:
                    ax.plot(result)

            elif dim == 2:
                if self.use_seaborn:
                    sns.heatmap(result, ax=ax, cmap=cmap)
                else:
                    cax = ax.imshow(result, cmap=cmap)
                    fig.colorbar(cax, ax=ax)

            elif dim == 3:
                ax.remove()
                ax = fig.add_subplot(1, self.num_results, idx + 1, projection='3d')
                ax.voxels(result, facecolors=colormaps.get_cmap(cmap)(result), edgecolor='k')
                ax.set_title(name)

            else:
                raise ValueError("Only dimensions 1D, 2D, and 3D are supported.")

        if adjust_params:
            plt.subplots_adjust(**adjust_params)
        else:
            plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Dummy examples for users:
    # 1D example
    data_1d_gt = np.sin(np.linspace(0, 2 * np.pi, 100))
    data_1d_pred = np.sin(np.linspace(0, 2 * np.pi, 100) + 0.1)
    plotter1d = plot_and_compare([data_1d_gt, data_1d_pred], names=['Ground Truth', 'Prediction'])
    plotter1d.plot()

    # 2D example
    data_2d_gt = np.random.rand(10, 10)
    data_2d_pred = data_2d_gt + np.random.normal(0, 0.1, (10, 10))
    plotter2d = plot_and_compare([data_2d_gt, data_2d_pred], names=['Ground Truth', 'Prediction'], use_seaborn=True)
    plotter2d.plot()

    # 3D example (voxel visualization)
    data_3d_gt = np.random.rand(5, 5, 5) > 0.7
    data_3d_pred = np.random.rand(5, 5, 5) > 0.7
    data_3d_pred2 = np.random.rand(5, 5, 5) > 0.7
    plotter3d = plot_and_compare([data_3d_gt, data_3d_pred, data_3d_pred2], names=['Ground Truth', 'Prediction', 'Prediction 2'])
    plotter3d.plot()
