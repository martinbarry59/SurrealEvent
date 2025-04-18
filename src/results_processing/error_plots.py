from config import results_path
import os
import matplotlib.pyplot as plt

files = ["training_error.txt", "testing_error.txt"]
def find_loss(text):
    text = text.split(",")
    for subtext in text:
        if "Loss:" in subtext:
            return float(subtext.split(":")[-1].strip())
for filename in files:
    print(os.path.join(results_path,filename))
    with open(os.path.join(results_path,filename) , 'r') as file:
        lines = file.readlines()

    errors = [line for line in lines if "Batch" in line]

    error_values = [find_loss(text) for text in errors]
    # plt.plot(training_errors, label=f'Training Errors {filename}')
    ## temporal smoothing of the error
    window = 2
    error_values = [error_values[i] for i in range(len(error_values)) if i % window == 0]
    error_values = [sum(error_values[i:i+window])/window for i in range(len(error_values)-window)]
    error_values = error_values[:len(error_values)-window]

    plt.plot(error_values, label=f'{filename.split("_")[0]}')
plt.legend()
plt.xlabel('Batch Step')
plt.ylabel('Error')
plt.title('Error Plot')
plt.savefig(os.path.join(results_path, "error_plot_test_train.png"))
plt.show()