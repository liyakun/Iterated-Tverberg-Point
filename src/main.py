"""
This is the main function.
"""
from src.model import telescope, skin
import time, os
from src.lib import plot
from src.lib import synthetic

""" run skin test """
"""
number_of_training, number_of_training_instances, number_of_fold, percentage_of_training(e.g. 7) 70%,
number of equally split disjoint subset
we have many single vectors for the equal split, when we want to tes the number of single vector's influence
we can select subset from all the vectors calculated from the equal split

my_skin = skin.Skin()
my_skin.run_skin_n_fold(1000, 100, 1, 0.1)
# for i in range(100):
#    my_skin.test_all_point(10, 0.9, "../resources/skin/All_Point_Test/error_all_half_step"+str(i)+".txt")
"""


""" run bank test """
"""
my_bank = bank.Bank()
my_bank.runbank(1000, 2000)
my_bank = bank.Bank()
my_bank.run_bank_n_fold(10, 4000, 500, 9)
# my_plot = plot.Plot()
# my_plot.box_plot(10, "../resources/bank/result/SecondTest/")
"""


""" test plot error """
"""
my_plot = plot.Plot()
my_plot.plot_error("../resources/skin/result/error_random.txt")
my_plot.plot_error("../resources/skin/result/error_equal.txt")


my_plot = plot.Plot()
#my_plot.plot_comparing_median(30, "../resources/telescope/")
#my_plot.plot_time("../resources/syntheticdata/result/vectors_30_iterations/")
#my_plot.plot_time("../resources/telescope/new_results/")
#for i in range(100, 3000, 100):
#    my_plot.box_plot_random(30, "../resources/telescope/result_30_1000_"+str(i)+"/")


#my_plot.box_plot_different_dimensions(20, "../resources/syntheticdata/result/dimensions/errors/")
#my_plot.plot_time_dimensions("../resources/syntheticdata/result/dimensions_30_iterations/errors/")
for i in range(2, 15, 1):
    my_plot.box_plot_random(30, "../resources/syntheticdata/result/dimensions_30_iterations/errors/"+str(i)+"/")

#my_plot.box_plot_different_vectors( "../resources/syntheticdata/result/vectors_30_iterations/")
#for i in range(1000, 6500, 500):
#    my_plot.box_plot_random(30, "../resources/syntheticdata/result/vectors_30_iterations/"+str(i)+"/")
"""


""" test synthetic data """
"""
my_synthetic_data = synthetic.SyntheticData()
# n_samples, n_features, n_informative, n_classes
# percent_of_train(0~1), number_of_training, number_of_training_instances, number_of_experiments
my_synthetic_data.run_fake_data_n_fold(10000, 9, 9, 2, 0.75, 3500, 1000, 10)

# n_samples, n_features, n_informative, n_classes, percent_train, vectors, instances, iterations
# my_fake_data.dimension_test(20000, 20, 20, 2, 0.75, 1000, 4000, 20)

# n_samples, n_features, n_informative, n_classes, percent_train, vectors, instances, iterations
# my_fake_data.vectors_test(30000, 5, 5, 2, 0.75, 10000, 1000, 20)
"""


""" test shuttle data"""
"""
my_shuttle = shuttle.Shuttle()
my_shuttle.run_shuttle_n_fold(1000, 2000, 10)
"""


""" test protein data"""
"""
n, number_of_training, number_of_training_instances, number_of_equal_disjoint_sets,
                           percent_of_training

# my_protein = protein.Protein()
# my_protein.run_protein_n_fold(20, 1000, 50, 800, 0.7)
"""


""" test telescope data """
"""
100: 652.132999897 seconds
200: 690.535000086 seconds
300: 688.878000021 seconds
400: 693.819999933 seconds
500: 686.217999935 seconds
600: 738.809000015 seconds

for j in range(1, 12):
    for i in range(64, 2048, 2**j):
        start_time = time.time()
        my_telescope = telescope.Telescope()
        my_telescope.run_telescope_n_fold(30, 1000, i, 1000, 0.9, "../resources/telescope/new_est_results/result_30_1000_"+str(i)+"/")
        file = open("../resources/telescope/new_est_results/result_30_1000_"+str(i)+"/time.txt", 'wb')
        file.write(str(time.time() - start_time))
"""


""" test wilt data """
"""
number_of_training, number_of_training_instances, number_of_equal_disjoint_sets, path
my_wilt = wilt.Wilt()
my_wilt.run_wilt_n_fold(1, 350, 200, 100, "../resources/wilt/result/")
"""