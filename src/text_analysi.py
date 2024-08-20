import os
import re
import nltk
import pandas as pd
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Setup NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Paths for the stopwords and dictionary files
STOPWORDS_PATH = "../data/StopWords/"
MASTER_DICT_PATH = "../data/MasterDictionary/"
ARTICLES_PATH = "../data/output/extracted_articles/"
OUTPUT_PATH = "../data/output/Output Data Structure.xlsx"

# Function to load stopwords from multiple files
def load_stopwords():
    stopwords_files = ['StopWords_Auditor.txt', 'StopWords_Currencies.txt', 'StopWords_DatesandNumbers.txt', 'StopWords_Generic.txt', 'StopWords_Geographic.txt', 'StopWords_Names.txt']
    stopwords_set = set()

    for file in stopwords_files:
        with open(os.path.join(STOPWORDS_PATH, file), 'r') as f:
            for line in f:
                stopwords_set.add(line.strip())

    # Add nltk stopwords too
    stopwords_set.update(stopwords.words('english'))
    
    return stopwords_set

# Function to load positive and negative word dictionaries
def load_master_dictionary():
    positive_words = set()
    negative_words = set()
    
    with open(os.path.join(MASTER_DICT_PATH, 'positive-words.txt'), 'r') as f:
        positive_words.update([line.strip() for line in f if line.strip() and not line.startswith(';')])

    with open(os.path.join(MASTER_DICT_PATH, 'negative-words.txt'), 'r') as f:
        negative_words.update([line.strip() for line in f if line.strip() and not line.startswith(';')])
    
    return positive_words, negative_words

# Function to clean and tokenize the text
def clean_and_tokenize(text, stopwords_set):
    # Lowercase the text and remove stopwords
    words = word_tokenize(text.lower())
    cleaned_words = [word for word in words if word.isalpha() and word not in stopwords_set]
    
    # Sentence tokenization
    sentences = sent_tokenize(text)
    
    return cleaned_words, sentences

# Function to compute sentiment scores
def compute_sentiment_scores(cleaned_words, positive_words, negative_words):
    positive_score = sum(1 for word in cleaned_words if word in positive_words)
    negative_score = sum(1 for word in cleaned_words if word in negative_words)
    
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(cleaned_words) + 0.000001)
    
    return positive_score, negative_score, polarity_score, subjectivity_score

# Function to compute readability metrics
def compute_readability_metrics(text):
    avg_sentence_length = textstat.sentence_count(text)
    fog_index = textstat.gunning_fog(text)
    return avg_sentence_length, fog_index

# Function to count complex words
def count_syllables(word):
    vowels = "aeiou"
    count = sum(1 for char in word if char in vowels)
    if word.endswith("es") or word.endswith("ed"):
        count -= 1
    return count

def count_complex_words(words):
    complex_words_count = sum(1 for word in words if count_syllables(word) > 2)
    return complex_words_count

# Function to count personal pronouns
def count_personal_pronouns(text):
    pronouns = re.findall(r'\b(I|we|my|ours|us)\b', text, re.I)
    return len(pronouns)

# Function to calculate average word length
def avg_word_length(cleaned_words):
    total_characters = sum(len(word) for word in cleaned_words)
    return total_characters / len(cleaned_words) if cleaned_words else 0

# Function to analyze each article
def analyze_article(url_id, text, positive_words, negative_words, stopwords_set):
    cleaned_words, sentences = clean_and_tokenize(text, stopwords_set)
    
    positive_score, negative_score, polarity_score, subjectivity_score = compute_sentiment_scores(cleaned_words, positive_words, negative_words)
    avg_sentence_length, fog_index = compute_readability_metrics(text)
    complex_word_count = count_complex_words(cleaned_words)
    word_count = len(cleaned_words)
    syllable_per_word = sum(count_syllables(word) for word in cleaned_words) / word_count
    personal_pronoun_count = count_personal_pronouns(text)
    avg_word_len = avg_word_length(cleaned_words)
    
    return {
        "URL_ID": url_id,
        "POSITIVE_SCORE": positive_score,
        "NEGATIVE_SCORE": negative_score,
        "POLARITY_SCORE": polarity_score,
        "SUBJECTIVITY_SCORE": subjectivity_score,
        "AVERAGE_SENTENCE LENGTH": avg_sentence_length,
        "FOG INDEX": fog_index,
        "COMPLEX WORD COUNT": complex_word_count,
        "WORD COUNT": word_count,
        "SYLLABLE PER WORD": syllable_per_word,
        "PERSONAL PRONOUN": personal_pronoun_count,
        "AVERAGE WORD LENGTH": avg_word_len
    }

# Main function to iterate over all articles and save results
def main():
    # Load dictionaries and stopwords
    stopwords_set = load_stopwords()
    positive_words, negative_words = load_master_dictionary()
    
    # Initialize a list to store results
    output = []
    
    # Loop through each article and perform analysis
    for txt_file in os.listdir(ARTICLES_PATH):
        url_id = txt_file.split(".")[0]
        with open(os.path.join(ARTICLES_PATH, txt_file), "r", encoding="utf-8") as file:
            text = file.read()
            
            # Analyze the article
            analysis_results = analyze_article(url_id, text, positive_words, negative_words, stopwords_set)
            output.append(analysis_results)
    
    # Convert results to DataFrame and save to Excel
    output_df = pd.DataFrame(output)
    output_df.to_excel(OUTPUT_PATH, index=False)
    
    print(f"Analysis complete! Results saved to {OUTPUT_PATH}")

# Run the main function
if __name__ == "__main__":
    main()
