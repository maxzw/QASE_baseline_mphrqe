from argparse import ArgumentParser
import pickle
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

if __name__ == '__main__':

    for dataset_name in ["AIFB", "MUTAG"]:
        for model_name in ["vec", "box", "beta", "StarQE"]:

            print(f"Formatting {dataset_name} {model_name}")

            plot_name = f"threshold_search_{model_name}_{dataset_name}"
            
            # Load the data into dummy figure manager
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            with open(plot_name + ".pkl",'rb') as f:
                fig: Figure = pickle.load(file=f)
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)
            
            # Set axis bounds
            ax = fig.gca()
            ax.set_ybound(0, 1)
            # Mutag
            if model_name == "StarQE" and dataset_name == "MUTAG":
                ax.set_xbound(lower=-20, upper=20)
            elif (model_name == "vec" or model_name == "box") and dataset_name == "MUTAG":
                ax.set_xbound(lower=10, upper=60)
            elif model_name == "beta" and dataset_name == "MUTAG":
                ax.set_xbound(lower=40, upper=100)
            # AIFB
            elif model_name == "StarQE" and dataset_name == "AIFB":
                ax.set_xbound(lower=-20, upper=20)
            elif model_name == "vec" and dataset_name == "AIFB":
                ax.set_xbound(lower=15, upper=35)
            elif model_name == "box" and dataset_name == "AIFB":
                ax.set_xbound(lower=10, upper=50)
            elif model_name == "beta" and dataset_name == "AIFB":
                ax.set_xbound(lower=45, upper=90)

            # Order labels
            required_ordered_labels = ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi']
            handles, labels = ax.get_legend_handles_labels()
            if model_name == "StarQE":
                gqs_structures = ['1hop', '2hop', '3hop', '2i', '3i', '2i-1hop', '1hop-2i']
                short_map = {
                    '1hop': '1p',
                    '2hop': '2p',
                    '3hop': '3p',
                    '2i': '2i',
                    '3i': '3i',
                    '2i-1hop': 'ip',
                    '1hop-2i': 'pi'
                }
                transformed_labels = [short_map[label] for label in labels]
                sort_indices = [transformed_labels.index(x) for x in required_ordered_labels]
                labels = [transformed_labels[i] for i in sort_indices]
                handles = [handles[i] for i in sort_indices]
                ax.legend(handles, labels)
            else:        
                gqs_structures = ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi']
                short_map = {
                    '1p': '1p',
                    '2p': '2p',
                    '3p': '3p',
                    '2i': '2i',
                    '3i': '3i',
                    'ip': 'ip',
                    'pi': 'pi'
                }
                transformed_labels = [short_map[label] for label in labels]
                sort_indices = [transformed_labels.index(x) for x in required_ordered_labels]
                labels = [transformed_labels[i] for i in sort_indices]
                handles = [handles[i] for i in sort_indices]
                ax.legend(handles, labels)

            # reset colors
            all_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            leg = ax.get_legend()
            for i, label in enumerate(labels):
                leg.legendHandles[i].set_color(all_colors[i])
                leg.legendHandles[i].set_color(all_colors[i])

            # Set names
            model_names = {
                'StarQE': 'StarQE',
                'vec': 'GQE',
                'box': 'Q2B',
                'beta': 'BetaE'
            }
            ax.set_title(f"Optimization results ({model_names[model_name]} on {dataset_name})")
            ax.set_xlabel("Distance threshold")
            ax.set_ylabel("F1-score")
            fig.set_size_inches(4, 3)

            # Save figure
            fig.savefig(f"opt_{model_name}_{dataset_name}.pdf", facecolor='w', bbox_inches='tight')