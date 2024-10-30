import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'comtb'
trackers.extend(trackerlist(name='ostrack', parameter_name='OSTrack', dataset_name=dataset_name,
                            run_ids=None, display_name='OSTrack256'))

dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
