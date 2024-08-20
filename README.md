# Text Analysis Project

## Overview

This project involves extracting textual data from specified URLs, performing text analysis, and computing various textual metrics. The analysis includes sentiment analysis, readability assessment, and various textual statistics. The project utilizes Python and several libraries for data extraction and analysis.

## Project Structure

```
Text-Analysis/
│
├── data/
│   ├── MasterDictionary/
│   │   ├── positive-words.txt
│   │   └── negative-words.txt
│   └── StopWords/
│       ├── stop_words_genreric.txt
│       ├── Stop_dateandNumber.txt
│       └── Stop_wordCurrency.txt
│
├── src/
│   ├── data_extraction.py
│   ├── text_analysis.py
│   └── utils.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── Input.xlsx
```

## Requirements

Ensure you have Python 3.x installed. Install the required libraries using:

```bash
pip install -r requirements.txt
```

### Required Libraries

- `beautifulsoup4`
- `requests`
- `nltk`
- `textblob`
- `pandas`
- `openpyxl`
- `scikit-learn`

## Data Extraction

### Input

- `Input.xlsx`: Contains URLs of articles to be extracted.

### Extraction Script

- `data_extraction.py`: This script extracts article text from the URLs specified in `Input.xlsx`. The text is saved in separate `.txt` files named by their URL IDs.

### Usage

To extract data, run:

```bash
python src/data_extraction.py
```

## Text Analysis

### Input

- Extracted article text files from the `data/` directory.

### Analysis Script

- `text_analysis.py`: This script performs various text analysis tasks including sentiment analysis and readability metrics based on the extracted text.

### Metrics Computed

- **Positive Score**: Sum of positive sentiment scores.
- **Negative Score**: Sum of negative sentiment scores.
- **Polarity Score**: A measure of sentiment polarity.
- **Subjectivity Score**: A measure of subjectivity.
- **Average Sentence Length**: Average number of words per sentence.
- **Percentage of Complex Words**: Percentage of words with more than two syllables.
- **Fog Index**: A readability metric.
- **Average Number of Words Per Sentence**: Average number of words in sentences.
- **Complex Word Count**: Count of complex words.
- **Word Count**: Total number of words.
- **Syllable Per Word**: Average number of syllables per word.
- **Personal Pronouns**: Count of personal pronouns.
- **Average Word Length**: Average length of words.

### Usage

To perform text analysis, run:

```bash
python src/text_analysis.py
```

### Output

The results of the text analysis will be saved in an Excel file formatted according to `Output Data Structure.xlsx`.

## Configuration

- **.gitignore**: Specifies files and directories to be ignored by Git.
- **requirements.txt**: Lists the Python dependencies.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request to contribute to the project.

## Contact

For any questions or feedback, please contact vinayiet435@gmail.com
