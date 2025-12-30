import argparse
from preprocess import preprocess_dataset
from train import train_model
from test import generate_examples
from evaluate_metrics import evaluate_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--data_dir", default="./LIDC-IDRI/")
    parser.add_argument("--processed_dir", default="./processed_volumes/")
    parser.add_argument("--output_dir", default="./checkpoints/")
    parser.add_argument("--results_dir", default="./results/")
    parser.add_argument("--model_path", default="./checkpoints/best_model.pth")
    parser.add_argument("--num_volumes", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_examples", type=int, default=10)

    args = parser.parse_args()

    if args.preprocess:
        print("[1/4] Preprocessing...")
        preprocess_dataset(args.data_dir, args.processed_dir, args.num_volumes)

    if args.train:
        print("[2/4] Training...")
        train_model(args.processed_dir, args.output_dir, args.epochs, args.batch_size)

    if args.test:
        print("[3/4] Generating examples...")
        generate_examples(
            args.model_path, args.processed_dir, args.results_dir, args.num_examples
        )

    if args.evaluate:
        print("[4/4] Evaluating...")
        evaluate_results(args.results_dir)


if __name__ == "__main__":
    main()
