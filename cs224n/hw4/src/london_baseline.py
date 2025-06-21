# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_corpus_path', type=str, required=True)
    args = parser.parse_args()
    dev_text = open(args.dev_corpus_path, encoding='utf-8').read()
    dev_data = [line for line in dev_text.split('\n') if line.strip()]
    total = len(dev_data)
    correct = sum(1 for line in dev_data if line.split('\t')[1].strip() == "London")
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return accuracy
    ### END YOUR CODE ###
    
    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
