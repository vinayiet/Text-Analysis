# Text-Analysis Project

## Overview

The Text-Analysis project is designed to perform a detailed analysis of textual data extracted from a large number of links. This project processes and analyzes over 200 articles to compute various text metrics, including sentiment scores, readability indices, and lexical statistics. The results are saved in an Excel file for further review.

## Project Components

1. **Stopwords Management**: Utilizes custom stopwords from multiple files in addition to NLTK’s built-in stopwords to filter out irrelevant words from the text.
2. **Sentiment Analysis**: Computes positive and negative sentiment scores, polarity, and subjectivity of the text.
3. **Readability Metrics**: Evaluates the readability of the text using indices such as the average sentence length and the Gunning Fog Index.
4. **Lexical Analysis**: Counts complex words, calculates syllable counts, and measures word lengths.
5. **Personal Pronouns**: Identifies and counts personal pronouns in the text.

## Features

- **Custom Stopwords Handling**: Load stopwords from specific files and include standard NLTK stopwords.
- **Sentiment Scoring**: Analyze positive and negative sentiment and derive polarity and subjectivity scores.
- **Readability Calculation**: Measure readability with average sentence length and fog index.
- **Lexical Statistics**: Count complex words, syllables per word, and average word length.
- **Personal Pronoun Detection**: Count occurrences of personal pronouns to assess the personal nature of the text.

## File Structure

- `data/StopWords/`: Contains stopwords files used for filtering text.
- `data/MasterDictionary/`: Contains positive and negative word dictionaries for sentiment analysis.
- `data/output/extracted_articles/`: Directory where the extracted articles are stored.
- `data/output/Output Data Structure.xlsx`: Output file where analysis results are saved.
- `data/input.xlsx`: Input file containing URLs associated with unique IDs.

## Dependencies

- `nltk`: For natural language processing tasks.
- `pandas`: For data manipulation and saving results.
- `textstat`: For computing readability metrics.

## Setup

1. **Install Required Libraries**:
   Make sure you have the necessary Python libraries installed. You can install them using pip:
   ```bash
   pip install nltk pandas textstat
   ```

2. **Download NLTK Resources**:
   Ensure that you have the required NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

3. **Directory Structure**:
   Ensure the directory structure matches the paths defined in the script:
   - `../data/StopWords/` - Contains custom stopwords files.
   - `../data/MasterDictionary/` - Contains positive and negative word lists.
   - `../data/output/extracted_articles/` - Contains the text files of the extracted articles.
   - `../data/output/` - Directory for saving the results Excel file.
   - `../data/input.xlsx` - Input file with URLs and IDs.

## Usage

1. **Prepare Your Data**:
   - Place the extracted articles in the `ARTICLES_PATH` directory.
   - Ensure the `INPUT_PATH` file contains the URLs with corresponding IDs.

2. **Run the Script**:
   Execute the script to perform the analysis:
   ```bash
   python text_analysis.py
   ```

   The results will be saved in the `OUTPUT_PATH` Excel file.

3. **Review Results**:
   Open the `Output Data Structure.xlsx` file to review the analysis results, including sentiment scores, readability metrics, and lexical statistics.

## Example Output

The output file contains the following columns:
- `URL_ID`: Identifier for the URL.
- `URL`: URL of the article.
- `POSITIVE_SCORE`: Number of positive words in the text.
- `NEGATIVE_SCORE`: Number of negative words in the text.
- `POLARITY_SCORE`: Polarity score of the text.
- `SUBJECTIVITY_SCORE`: Subjectivity score of the text.
- `AVERAGE_SENTENCE_LENGTH`: Average length of sentences.
- `FOG_INDEX`: Gunning Fog Index.
- `COMPLEX_WORD_COUNT`: Number of complex words.
- `WORD_COUNT`: Total word count.
- `SYLLABLE_PER_WORD`: Average syllables per word.
- `PERSONAL_PRONOUN_COUNT`: Count of personal pronouns.
- `AVERAGE_WORD_LENGTH`: Average length of words.

Thank you for Visiting my GitHub Repository.
