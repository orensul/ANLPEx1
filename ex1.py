import sys
import time
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorWithPadding
from transformers import EvalPrediction
from transformers import set_seed
# import wandb
import os
import torch
import numpy as np

NUM_ARGS = 4
models = ['roberta-base', 'google/electra-base-generator', 'bert-base-uncased']


def preprocess_function(examples, tokenizer):
    # Tokenize the texts with Dynamic Padding
    return tokenizer(examples['sentence'], max_length=512, truncation=True)


def read_args():
    if len(sys.argv) != NUM_ARGS + 1:
        print("Error: " + NUM_ARGS + " arguments are required.")
        print("Usage: python ex1.py num_seeds num_training_examples num_validation_examples num_prediction_samples")
        sys.exit(1)

    # Retrieve the arguments
    num_seeds = sys.argv[1]
    num_training_examples = sys.argv[2]
    num_validation_examples = sys.argv[3]
    num_prediction_samples = sys.argv[4]

    # Print the arguments
    print("num of seeds:", num_seeds)
    print("num of training examples:", num_training_examples)
    print("num of validation examples:", num_validation_examples)
    print("num of prediction samples:", num_prediction_samples)

    return int(num_seeds), int(num_training_examples), int(num_validation_examples), int(num_prediction_samples)


# 5. evaluation metrics
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    accuracy = np.mean(preds == p.label_ids)
    return {'accuracy': accuracy}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load arguments
    num_seeds, num_training_examples, num_validation_examples, num_prediction_samples = read_args()

    # 2. Load dataset
    raw_datasets = load_dataset('glue', 'sst2')
    print("size of training: " + str(len(raw_datasets['train'])))
    print("size of validation: " + str(len(raw_datasets['validation'])))

    train_dataset = raw_datasets['train']
    if num_training_examples != -1:
        train_dataset = train_dataset.select(range(num_training_examples))

    eval_dataset = raw_datasets['validation']
    if num_validation_examples != -1:
        eval_dataset = eval_dataset.select(range(num_validation_examples))

    test_dataset = raw_datasets['test']
    if num_prediction_samples != -1:
        test_dataset = test_dataset.select(range(num_prediction_samples))

    accuracy_values = {}  # Collect accuracy values for each model and for each seed
    total_train_time = 0
    # os.environ["WANDB_API_KEY"] = "8278bdace6809a899287c0488390f5bbc4e200f2"
    tokenizers = {}
    file_path = "/content/drive/MyDrive/ANLPEx1/res.txt"
    res_file = open(file_path, 'w')

    for model_name in models:
        print("Load model: " + model_name)
        # 3. Load models and tokenizers
        config = AutoConfig.from_pretrained(model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model_name not in tokenizers:
            tokenizers[model_name] = tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)

        # 4. Tokenize and process dataset
        train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, )
        eval_dataset = eval_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, )

        # Define data collator with Dynamic Padding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

        # Loop over the seeds
        for seed in range(num_seeds):
            print("Run new seed: " + str(seed) + ", for model: " + model_name)
            set_seed(seed)

            # Initialize Weights & Biases
            # wandb.init(project="ANLP_ex1", name=f"model={model_name}_seed={seed}")

            # 6. Train
            output_dir = '/content/drive/MyDrive/ANLPEx1/output/' + model_name + "/" + str(seed) + "/"
            training_args = TrainingArguments(
                output_dir=output_dir, logging_steps=200, save_steps=0, seed=seed, ) #, report_to='wandb'

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
            )
            start_time = time.time()
            trainer.train()
            end_time = time.time()
            train_time = end_time - start_time
            total_train_time += train_time

            # 7. evaluate
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            # wandb.log(metrics)

            accuracy, eval_runtime = metrics['eval_accuracy'], metrics['eval_runtime']

            if model_name not in accuracy_values:
                accuracy_values[model_name] = {}
            accuracy_values[model_name][seed] = accuracy

            # save the current model
            model_path = '/content/drive/MyDrive/ANLPEx1/output/' + model_name + "/" + str(seed) + "/" + "model.pth"
            torch.save(model, model_path)
            print("saving model in path=" + model_path)

            # wandb.finish()

        # Calculate mean and standard deviation
        model_acc = []
        for seed, acc in accuracy_values[model_name].items():
            model_acc.append(acc)

        highest_mean_accuracy = 0
        model_highest_mean_accuracy = None

        mean_accuracy = np.mean(model_acc)
        if mean_accuracy >= highest_mean_accuracy:
            highest_mean_accuracy = mean_accuracy
            model_highest_mean_accuracy = model_name
        std_accuracy = np.std(model_acc)

        res_file.write(f"{model_name},{mean_accuracy:.3f} +- {std_accuracy:.3f}\n")

    res_file.write("----\n")
    res_file.write("train time,{:.3f}\n".format(total_train_time))

    print("results on validation set:")
    print(accuracy_values)
    print("highest mean accuracy: " + str(highest_mean_accuracy))
    print("model with the highest accuracy: " + str(model_highest_mean_accuracy))

    # select the best seed of the best model (model with the best mean accuracy across all of his seeds)
    best_seed, best_acc = 0, 0
    for seed, acc in accuracy_values[model_highest_mean_accuracy].items():
        if acc >= best_acc:
            best_seed = seed
            best_acc = acc

            # load best model (in terms of mean accuracy) with his best seed
    best_model_path = '/content/drive/MyDrive/ANLPEx1/output/' + model_highest_mean_accuracy + "/" + str(
        best_seed) + "/" + "model.pth"
    best_model = torch.load(best_model_path)
    # Set the model to evaluation mode
    best_model.eval()

    # best model tokenizer
    best_model_tokenizer = tokenizers[model_highest_mean_accuracy]

    file_path = "/content/drive/MyDrive/ANLPEx1/predictions.txt"
    predictions_file = open(file_path, 'w')

    # make predictions
    start_time = time.time()
    for test_sample in test_dataset:
        input_text = test_sample['sentence']
        tokenized_sample = best_model_tokenizer(input_text, return_tensors='pt').to(device)
        res = torch.argmax(best_model(**tokenized_sample).logits)
        predictions_file.write(f"{input_text}###{res}\n")
    end_time = time.time()
    res_file.write(f"predict time, {end_time - start_time:.3f}")

    # close files
    res_file.close()
    predictions_file.close()


if __name__ == "__main__":
    main()