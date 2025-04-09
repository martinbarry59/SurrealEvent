from config import results_path
import os
import matplotlib.pyplot as plt

files = ["slurm-627.out", "slurm-624.out"]
for filename in files:
    with open(os.path.join(results_path,filename) , 'r') as file:
        lines = file.readlines()

    errors = [line for line in lines if "Batch Step" in line]
    epoch_errors = [line for line in lines if "Epoch" in line]

    error_values = [float(text.split("Loss:")[-1].strip()) for text in errors]
    training_errors = [error  for error in error_values if error < 0.1]
    validation_errors = [error for error in error_values if error >= 0.1]
    # plt.plot(training_errors, label=f'Training Errors {filename}')
    plt.plot(validation_errors, label=f'Validation Errors {filename}')
plt.legend()
plt.xlabel('Batch Step')
plt.ylabel('Error')
plt.title('Error Plot')
plt.savefig(os.path.join(results_path, "error_plot_test_train.png"))
plt.show()