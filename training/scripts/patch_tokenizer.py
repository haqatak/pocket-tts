from training.tokenizer_patch import extend_tokenizer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_model", required=True)
    parser.add_argument("--output_model", required=True)
    args = parser.parse_args()

    # Common Norwegian characters and subwords
    new_tokens = ["æ", "ø", "å", "Æ", "Ø", "Å", "er", "en", "et", "det"]
    extend_tokenizer(args.original_model, args.output_model, new_tokens)
