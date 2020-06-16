import glob
import os


def main():
    emails: list = []
    labels: list = []
    file_path = "datasets/enron1/ham"
    for filename in glob.glob(os.path.join(file_path, "*.txt")):
        with open(filename, "r", encoding="ISO-8859-1") as infile:
            emails.append(infile.read())
            labels.append(0)

    return emails, labels


if __name__ == "__main__":
    main()
