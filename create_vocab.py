from tqdm import tqdm
import pandas as pd

GLOVE_EMB_PATH = "../data/glove.840B.300d.txt"
DUPLICATE_TEXT_DATA = "thesis_data/not_duplicates_data.csv"
TARGET_TEXT_DATA = "thesis_data/processed_question_data.csv"
VOCAB_OUTPUT = "thesis_data/vocab.all"


def extract_vocab(glove_file, ):
    with open(VOCAB_OUTPUT, "w", encoding="utf8") as vocab:
        # vocab.write("_PAD\n")
        # vocab.write("_GO\n")
        # vocab.write("_EOS\n")
        # vocab.write("_UNK\n")
        with open(glove_file, "r", encoding="utf8") as f:
            for line in tqdm(f):
                word = line.split(" ")[0]
                vocab.write(word.lower() + "\n")
        print("extracted glove words")
        def write_sentences(sentence):
            try:
                words = sentence.split(" ")
                for word in words:
                    vocab.write(word+"\n")
            except:
                pass
        df = pd.read_csv(DUPLICATE_TEXT_DATA, encoding="utf8")
        print("loaded duplicate data")
        df2 = pd.read_csv(TARGET_TEXT_DATA, encoding="utf8")
        print("loaded question answering data")
        for question in df["question1"]:
            write_sentences(question)
        for question in df["question2"]:
            write_sentences(question)
        print("processed duplicate data")
        for question in df2["q"]:
            write_sentences(question)
        for paragraph in df2["p"]:
            sentences = paragraph.split(".")
            for sentence in sentences:
                write_sentences(sentence)
        print("processed question answering data")

if __name__ == '__main__':
    extract_vocab(GLOVE_EMB_PATH)
