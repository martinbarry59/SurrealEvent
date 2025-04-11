import os

package_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(package_dir)

data_path = os.path.join(parent_dir, 'dataset/')

results_path = os.path.join(parent_dir, 'results/')
if not os.path.exists(results_path):
    os.makedirs(results_path)



checkpoint_path = os.path.join(parent_dir, 'checkpoints/')
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)