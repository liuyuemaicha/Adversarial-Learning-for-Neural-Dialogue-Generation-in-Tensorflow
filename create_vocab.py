from tqdm import tqdm

GLOVE_EMB_PATH = "../data/glove.840B.300d.txt"
VOCAB_OUTPUT = "thesis_data/vocab.all"


def extract_vocab(glove_file):
    with open(VOCAB_OUTPUT, "w", encoding="utf8") as vocab:
        vocab.write("_PAD\n")
        vocab.write("_GO\n")
        vocab.write("_EOS\n")
        vocab.write("_UNK\n")
        with open(glove_file, "r", encoding="utf8") as f:
            for line in tqdm(f):
                word = line.split(" ")[0]
                vocab.write(word.lower() + "\n")


if __name__ == '__main__':
    extract_vocab(GLOVE_EMB_PATH)
