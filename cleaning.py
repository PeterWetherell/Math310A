import string

def process_file(input_file, output_file):

    # Open the input file for reading
    with open(input_file, 'r', encoding='utf-8') as file:
        # Read the content of the file
        content = file.read()

    # Remove punctuation and convert to lowercase
    # str.maketrans() creates a mapping table for translation
    translator = str.maketrans('', '', string.punctuation)
    stripped_content = content.translate(translator).lower()

    # Write the processed content to the output file
    with open(output_filename, 'w') as file:
        file.write(stripped_content)

# Specify input and output file names
input_filename = 'Transcriptions/pnp_part1_transcription_noisy.txt'  # Change this to your input file name
output_filename = 'Transcriptions/pnp_part1_cleaned_noisy.txt'  # Change this to your desired output file name

# Process the file
process_file(input_filename, output_filename)

print(f"Processed '{input_filename}' and saved to '{output_filename}'.")
