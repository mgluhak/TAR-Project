import evaluation.baseline1
import evaluation.test_model1
from evaluation.eval_utils import old_store_result

# store_result((evaluation.baseline1.evaluate("gender")), 'results/baseline5_9_5_2017.pkl', "Gender only")
# store_result((evaluation.baseline1.evaluate("age")), 'results/baseline5_9_5_2017.pkl', "Age only")
# store_result((evaluation.baseline1.evaluate("both")), 'results/baseline5_9_5_2017.pkl', "Both")

old_store_result((evaluation.test_model1.evaluate("gender")), 'results/test2_9_5_2017.pkl', "Gender only")
old_store_result((evaluation.test_model1.evaluate("age")), 'results/test2_9_5_2017.pkl', "Age only")
old_store_result((evaluation.test_model1.evaluate("both")), 'results/test2_9_5_2017.pkl', "Both")