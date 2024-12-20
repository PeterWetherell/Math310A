from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def cosine_similarity_files(file1, file2):
    # Read the files
    text1 = read_file(file1)
    text2 = read_file(file2)

    # Create a CountVectorizer to convert text to vector
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([text1, text2])
    
    # Compute the cosine similarity
    cosine_sim = cosine_similarity(vectors)
    
    sim = cosine_sim[0][1]  # Return the similarity score between the two texts
    
    similarity = sim*100
    
    return similarity

# Specify the input file names
file1 = 'Transcriptions/YWP_cleaned.txt'  # Change to your first file
file2 = 'CleanedOutputTranscriptions/donothing.txt'  # Change to your second file
print(cosine_similarity_files(file1,file2))
# # Calculate and print the similarity

