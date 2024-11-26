import string

#https://gist.github.com/sebleier/554280
stop_words = {
    "i",        "me",       "my",           "myself",       "we",       "our",      "ours",     "ourselves",    "you",          "your",
    "yours",    "yourself", "yourselves",   "he",           "him",      "his",      "himself",  "she",          "her",          "hers",
    "herself",  "it",       "its",          "itself",       "they",     "them",     "their",    "theirs",       "themselves",   "what",
    "which",    "who",      "whom",         "this",         "that",     "these",    "those",    "am",           "is",           "are",
    "was",      "were",     "be",           "been",         "being",    "have",     "has",      "had",          "having",       "do",
    "does",     "did",      "doing",        "a",            "an",       "the",      "and",      "but",          "if",           "or",
    "because",  "as",       "until",        "while",        "of",       "at",       "by",       "for",          "with",         "about",
    "against",  "between",  "into",         "through",      "during",   "before",   "after",    "above",        "below",        "to",
    "from",     "up",       "down",         "in",           "out",      "on",       "off",      "over",         "under",        "again",
    "further",  "then",     "once",         "here",         "there",    "when",     "where",    "why",          "how",          "all",
    "any",      "both",     "each",         "few",          "more",     "most",     "other",    "some",         "such",         "no",
    "nor",      "not",      "only",         "own",          "same",     "so",       "than",     "too",          "very",         "can",
    "will",     "just",     "don't",        "should",       "now"
}

def process_file(input_file, output_file):

    # Open the input file for reading
    with open(input_file, 'r', encoding='utf-8') as file:
        # Read the content of the file
        content = file.read()

    # Remove punctuation and convert to lowercase
    # str.maketrans() creates a mapping table for translation
    translator = str.maketrans('', '', string.punctuation)
    stripped_content = content.translate(translator).lower()


    """
    # Removing stop words occurs in check and not here
    # Remove stop words
    filtered_words = [
        word for word in stripped_content.split() if word not in stop_words
    ]
    processed_content = ' '.join(filtered_words)
    """

    # Write the processed content to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(stripped_content)

# Specify input and output file names
input_filename = 'OutputTranscriptions/donothing.txt'  # Change this to your input file name
output_filename = 'CleanedOutputTranscriptions/donothing.txt'  # Change this to your desired output file name

# Process the file
process_file(input_filename, output_filename)

print(f"Processed '{input_filename}' and saved to '{output_filename}'.")
