from training.data_prep import download_ssc, download_npsc
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssc_dir", default="data/SSC", help="Directory for SSC dataset")
    parser.add_argument("--npsc_dir", default="data/NPSC", help="Directory for NPSC dataset")
    args = parser.parse_args()

    download_ssc(args.ssc_dir)
    download_npsc(args.npsc_dir)
