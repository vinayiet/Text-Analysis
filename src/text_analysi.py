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

# Initialize stopwords and master dictionary paths (if needed for future use)
STOPWORDS_PATH = "../data/StopWords/"
MASTER_DICT_PATH = "../data/MasterDictionary/"

# Provided text for analysis
text = """
Client Background
Client: A leading insurance firm worldwide

Industry Type: BFSI

Products & Services: Insurance

Organization Size: 10000+

The Problem
The insurance industry, particularly in the context of providing coverage to Public Company Directors against Insider Trading public lawsuits, faces a significant challenge in accurately determining insurance premiums. Traditional methods of premium calculation may lack precision, and there is a growing need for more sophisticated and data-driven approaches. The integration of Artificial Intelligence (AI) and Machine Learning (ML) models in predicting insurance premiums for this specialized coverage is essential to enhance accuracy, fairness, and responsiveness in adapting to evolving risk factors.

The problem at hand involves developing robust AI and ML models that can effectively analyze a multitude of dynamic variables influencing the risk profile of Public Company Directors. These variables include market conditions, regulatory changes, historical legal precedents, financial performance of the insured company, and individual directorial behaviors. The goal is to create a predictive model that not only accurately assesses the risk associated with potential insider trading public lawsuits but also adapts in real-time to new information, ensuring that the insurance premiums charged by the global insurance firm are reflective of the current risk landscape.

Key Challenges:

Data Complexity: The relevant data for predicting insurance premiums in this context is multifaceted, involving financial data, legal precedents, market trends, and individual directorial histories. Integrating and interpreting this diverse set of data poses a significant challenge.
Dynamic Risk Factors: The risk factors influencing insider trading public lawsuits are dynamic and subject to rapid changes. The models must be capable of adapting to evolving market conditions, legal landscapes, and individual company dynamics.
Fairness and Ethics: Ensuring fairness in premium calculation is critical. The models should be designed to avoid biases and discriminatory practices, considering the diverse backgrounds and contexts of Public Company Directors.
Regulatory Compliance: The insurance industry is subject to regulatory frameworks that vary across jurisdictions. The developed models need to comply with these regulations while providing accurate and reliable predictions.
Interpretability: Transparency in model predictions is crucial, especially in an industry where decisions can have significant financial implications. Ensuring that the AI and ML models are interpretable and explainable is vital for gaining the trust of stakeholders.
Addressing these challenges will not only improve the accuracy of insurance premium predictions but also contribute to the overall efficiency and effectiveness of the insurance services provided to Public Company Directors by the leading global insurance firm.

Blackcoffer Solution
To develop an ML and AI-based insurance premium prediction model for Public Company Directors in the USA, safeguarding them against insider trading public lawsuits, we propose a comprehensive solution leveraging advanced machine learning techniques. The goal is to create a model that accurately assesses the risk associated with individual directors and adapts to dynamic market conditions.

Data Collection and Preprocessing:
Financial Data:
Gather financial data related to the insured companies, including revenue, profit margins, and financial stability indicators.
Incorporate stock market data and trading patterns to capture potential insider trading signals.
Legal History:
Collect historical legal cases related to insider trading lawsuits, with a focus on outcomes and financial implications.
Integrate legal precedents to understand patterns and potential future risks.
Directorial Profiles:
Compile individual profiles for each Public Company Director, including their professional history, prior legal involvements, and any relevant affiliations.
Market Trends and Regulatory Changes:
Monitor market trends and regulatory changes affecting the insurance landscape.
Incorporate external data sources for real-time updates on legal and market conditions.
Feature Engineering:
Risk Factors:
Identify key risk factors contributing to the likelihood of insider trading allegations.
Develop features that encapsulate financial stability, market conditions, and individual directorial behaviors.
Sentiment Analysis:
Implement sentiment analysis on news articles and social media to gauge public perception and potential legal scrutiny.
Machine Learning Models:
Supervised Learning:
Employ supervised learning algorithms such as Random Forests, Gradient Boosting, or ensemble models.
Train the model on historical data with labeled outcomes related to insider trading lawsuits.
Anomaly Detection:
Implement anomaly detection techniques to identify unusual patterns that may indicate potential insider trading activities.
Dynamic Risk Assessment:
Real-Time Updates:
Design the model to continuously update with real-time data to adapt to evolving risk factors.
Implement a feedback loop to capture the impact of recent legal cases and market events.
Scenario Analysis:
Develop scenario analysis capabilities to assess the impact of hypothetical events on premium calculations.
Fairness and Transparency:
Fairness Metrics:
Integrate fairness metrics to ensure unbiased predictions across diverse directorial profiles.
Regularly audit and refine the model to address any identified biases.
Explainability:
Implement model explainability tools to provide clear insights into premium calculations.
Ensure transparency in how the model arrives at its predictions.
Model Integration and Deployment:
User-Friendly Interface:
Develop a user-friendly interface for underwriters to interact with the model.
Ensure seamless integration into the existing insurance company workflow.
API Integration:
Provide API endpoints for easy integration with existing insurance systems.
Monitoring and Maintenance:
Model Monitoring:
Implement continuous monitoring to detect model drift and performance degradation.
Regularly update the model with new data and retrain it to maintain accuracy.
Scalability:
Design the solution to scale horizontally to accommodate an increasing volume of data.
By adopting this ML and AI-based approach, the insurance company can enhance its ability to predict insurance premiums accurately, adapt to changing risk landscapes, and provide tailored coverage for Public Company Directors against insider trading public lawsuits in the dynamic environment of the USA.

Solution Architecture Diagram
Data Collection and Integration:
Data Sources: Financial records, legal databases, directorial profiles, market data.
Integration Layer: ETL processes, SQL/NoSQL databases.
Feature Engineering:
Feature Selection and Engineering Module.
Machine Learning Models:
Model Training Module: Scikit-Learn, TensorFlow, or PyTorch.
Model Evaluation Component.
Dynamic Risk Assessment:
Real-Time Data Integration Component: Apache Kafka.
Scenario Analysis Module.
Fairness and Transparency:
Fairness Metrics Integration.
Explainability Module: SHAP or Lime.
Model Integration and Deployment:
API Layer: RESTful API.
User Interface (UI).
Documentation for Integration.
Monitoring and Maintenance:
Monitoring Dashboard: Prometheus, Grafana.
Automated Model Update Pipeline: CI/CD.
General Documentation:
Model Architecture Document.
Technical User Manual.
Compliance Documentation:
Regulatory Compliance Report.
Data Privacy and Security Documentation.
Post-Implementation Support:
Support and Maintenance Plan.
Training and Knowledge Transfer:
Training Sessions.
Knowledge Transfer Documentation.
Scalability and Future-Proofing:
Scalable Infrastructure.
Flexibility for Future Enhancements.
Tools & Technology Used By Blackcoffer
Building an ML and AI-based insurance premium prediction model involves the use of various tools and technologies across different stages of development. Here’s a list of tools and technologies that can be employed for creating such a model for a leading insurance firm in the USA, specifically targeting Public Company Directors against insider trading public lawsuits:

Data Collection and Preprocessing:
Python: A versatile programming language commonly used for data manipulation and preprocessing.
Pandas: A Python library for data manipulation and analysis, useful for handling structured data.
NumPy: A library for numerical operations in Python, often used for efficient array operations.
SQL/NoSQL Databases: To store and retrieve structured and unstructured data efficiently.
Feature Engineering:
Scikit-Learn: A machine learning library in Python that includes tools for feature extraction and preprocessing.
NLTK (Natural Language Toolkit): For processing and analyzing textual data, particularly for sentiment analysis.
Machine Learning Models:
Scikit-Learn: Provides various machine learning algorithms for classification tasks, including Random Forests and Gradient Boosting.
XGBoost or LightGBM: Powerful gradient boosting frameworks for improved predictive performance.
TensorFlow or PyTorch: Deep learning frameworks for building and training neural networks if the complexity of the model demands it.
Dynamic Risk Assessment:
Apache Kafka or RabbitMQ: Message brokers to facilitate real-time data streaming and updates.
Airflow: A platform to programmatically author, schedule, and monitor workflows, useful for scheduling model updates.
Fairness and Transparency:
Aequitas or Fairness Indicators: Libraries for assessing and mitigating bias in machine learning models.
SHAP (SHapley Additive exPlanations): An algorithm for model interpretability.
Model Integration and Deployment:
Flask or Django: Web frameworks for building the model deployment API.
Docker: Containerization tool for packaging the model and its dependencies.
Kubernetes: Container orchestration for deploying and managing containerized applications at scale.
RESTful API: For communication between the model and other components in the insurance company’s infrastructure.
Monitoring and Maintenance:
Prometheus: An open-source monitoring and alerting toolkit.
Grafana: A platform for monitoring and observability with beautiful, customizable dashboards.
Jenkins or GitLab CI/CD: Continuous integration and continuous deployment tools for automating model updates and deployment.
MLflow: An open-source platform to manage the end-to-end machine learning lifecycle.
General Development Environment:
Jupyter Notebooks: Interactive computing environment for exploratory data analysis and model development.
Git: Version control system for collaborative development.
VS Code or PyCharm: Integrated development environments (IDEs) for coding and debugging.
It’s important to note that the choice of specific tools may vary based on the preferences of the data science team, the complexity of the model, and the existing technology stack of the insurance company. Additionally, compliance with regulatory requirements and industry standards should be considered in the selection of tools and technologies.

Blackcoffer Deliverables
The deliverables for an ML and AI-based insurance premium model for Public Company Directors in the USA, aiming to predict premiums for protection against insider trading public lawsuits, would encompass various stages of the development and deployment process. Here is a comprehensive list of deliverables:

1. Project Documentation:
1.1 Project Proposal:

Clearly outlines the objectives, scope, and methodology of the premium prediction model.
1.2 Requirements Document:

Specifies the functional and non-functional requirements of the model, considering the insurance company’s needs and regulatory compliance.
2. Data Collection and Preprocessing:
2.1 Data Collection Report:

Details the sources and types of data gathered, including financial records, legal cases, and directorial profiles.
2.2 Cleaned and Preprocessed Dataset:

A structured dataset ready for model training, containing relevant features and properly handled missing or inconsistent data.
3. Feature Engineering:
3.1 Feature Selection and Engineering Report:

Documents the process of selecting and creating features, highlighting their relevance to the prediction task.
4. Machine Learning Models:
4.1 Trained ML Models:

Includes the serialized models trained on historical data, such as Random Forests, Gradient Boosting, or other chosen algorithms.
4.2 Model Evaluation Report:

Evaluates the performance of the models on validation and test datasets, including metrics like accuracy, precision, recall, and F1-score.
5. Dynamic Risk Assessment:
5.1 Real-Time Integration Component:

Code or module that integrates real-time data for dynamic risk assessment.
5.2 Scenario Analysis Module:

Component allowing the assessment of premium changes based on hypothetical scenarios.
6. Fairness and Transparency:
6.1 Fairness Assessment Report:

Evaluates and mitigates bias, documenting fairness metrics and any adjustments made.
6.2 Explainability Module:

Implementation of tools or methodologies for model interpretability and explanation.
7. Model Integration and Deployment:
7.1 Deployed API:

RESTful API endpoint for seamless integration into the insurance company’s systems.
7.2 User Interface (UI):

User-friendly interface for underwriters to interact with the model, providing insights and entering necessary information.
7.3 Documentation for Integration:

Comprehensive guide on integrating the model into the existing workflow, including API documentation.
8. Monitoring and Maintenance:
8.1 Monitoring Dashboard:

Visual representation of key metrics and alerts for model performance, developed using tools like Grafana.
8.2 Automated Model Update Pipeline:

CI/CD pipeline or automated process for updating and retraining the model with new data.
9. General Documentation:
9.1 Model Architecture Document:

Detailed explanation of the model’s architecture, including components and their interactions.
9.2 Technical User Manual:

Documentation guiding technical users on deploying, maintaining, and troubleshooting the model.
10. Training and Knowledge Transfer:
10.1 Training Sessions:

Conducted for the insurance company’s staff, including underwriters and IT personnel, to ensure effective use and understanding of the model.
10.2 Knowledge Transfer Documentation:

Detailed documentation covering model usage, maintenance procedures, and troubleshooting tips.
11. Compliance Documentation:
11.1 Regulatory Compliance Report:

Ensures that the model adheres to relevant insurance regulations in the USA.
11.2 Data Privacy and Security Documentation:

Outlines measures taken to ensure the privacy and security of sensitive data.
12. Post-Implementation Support:
12.1 Support and Maintenance Plan:

Document outlining the support and maintenance plan for the model post-implementation, including response times and escalation procedures.
By delivering these items, the insurance firm can ensure a thorough and transparent development process, facilitating successful integration and utilization of the ML and AI-based insurance premium prediction model.

Business Impacts
The implementation of an ML and AI-based insurance premium model for Public Company Directors in the USA, specifically tailored to protect them from insider trading public lawsuits, can have significant business impacts for the leading insurance firm. Here are several potential business impacts:

1. Improved Accuracy and Risk Assessment:
Impact: Enhanced accuracy in predicting premiums based on advanced data analysis and machine learning algorithms.
Benefit: Better risk assessment leads to more precise premium calculations, reducing the likelihood of underpricing or overpricing policies.
2. Increased Competitiveness:
Impact: Utilizing cutting-edge technology to provide more accurate and dynamic premium predictions.
Benefit: Positions the insurance firm as a leader in the market, attracting more clients seeking innovative and reliable insurance solutions.
3. Tailored Coverage and Pricing:
Impact: Customizing coverage and premiums based on individual directorial profiles and evolving risk factors.
Benefit: Attracts clients with diverse risk profiles, offering tailored solutions that align with their specific needs.
4. Faster Decision-Making:
Impact: Automation of premium calculations and decision-making processes.
Benefit: Speeds up underwriting processes, enabling quicker responses to client inquiries and facilitating faster policy issuance.
5. Reduced Operational Costs:
Impact: Automation of routine tasks related to premium calculation and risk assessment.
Benefit: Decreases manual workload, leading to operational efficiency and cost savings.
6. Real-Time Adaptation to Market Changes:
Impact: Integration of real-time data for dynamic risk assessment.
Benefit: Enables the insurance firm to adapt quickly to changes in market conditions, ensuring that premiums remain reflective of current risk landscapes.
7. Enhanced Customer Satisfaction:
Impact: Accurate pricing, fair premium calculations, and transparent communication.
Benefit: Increases customer satisfaction by providing a reliable and customer-centric insurance experience.
8. Mitigation of Regulatory Risks:
Impact: Implementation of a solution that complies with insurance regulations and industry standards.
Benefit: Reduces the risk of regulatory non-compliance, protecting the firm from legal and financial repercussions.
9. Data-Driven Decision-Making:
Impact: Utilizing data-driven insights for decision-making processes.
Benefit: Empowers the firm’s leadership with actionable insights, contributing to strategic decision-making and business planning.
10. Brand Reputation and Trust:
Impact: Adoption of fairness-aware and transparent AI models.
Benefit: Builds trust among clients and stakeholders by demonstrating a commitment to fairness, transparency, and ethical AI practices.
11. Risk Mitigation for Clients:
Impact: Providing insurance coverage that reflects the evolving nature of insider trading public lawsuits.
Benefit: Assists Public Company Directors in mitigating financial risks associated with legal actions, enhancing the value proposition for clients.
12. Scalability and Future-Proofing:
Impact: Designing the solution to scale and adapt to future industry developments.
Benefit: Ensures the longevity and relevance of the insurance firm’s technology infrastructure in the face of evolving business and technological landscapes.
13. Revenue Growth:
Impact: Attracting a larger customer base and retaining existing clients through innovative and accurate insurance solutions.
Benefit: Contributes to revenue growth by expanding the firm’s market share and increasing customer loyalty.
By recognizing and leveraging these business impacts, the leading insurance firm can derive significant value from the implementation of an ML and AI-based insurance premium model tailored for Public Company Directors in the USA.

Summarize
Summarized: https://blackcoffer.com/

This project was done by Blackcoffer Team, a Global IT Consulting firm.

Contact Details
This solution was designed and developed by Blackcoffer Team
Here are my contact details:
Firm Name: Blackcoffer Pvt. Ltd.
Firm Website: www.blackcoffer.com
Firm Address: 4/2, E-Extension, Shaym Vihar Phase 1, New Delhi 110043
Email: ajay@blackcoffer.com
Skype: asbidyarthy
WhatsApp: +91 9717367468
Telegram: @asbidyarthy
"""

# Define the paths for data
OUTPUT_PATH = "../src/Output Data Structure.xlsx"

def load_stopwords():
    # Sample stopwords files path (you may need to adjust based on actual files)
    stopwords_files = ['StopWords_Auditor.txt', 'StopWords_Currencies.txt', 'StopWords_DatesandNumbers.txt', 'StopWords_Generic.txt', 'StopWords_Geographic.txt', 'StopWords_Names.txt']
    stopwords_set = set()

    for file in stopwords_files:
        with open(os.path.join(STOPWORDS_PATH, file), 'r') as f:
            for line in f:
                stopwords_set.add(line.strip())

    # Add nltk stopwords too
    stopwords_set.update(stopwords.words('english'))
    
    return stopwords_set

def load_master_dictionary():
    positive_words = set()
    negative_words = set()
    
    with open(os.path.join(MASTER_DICT_PATH, 'positive-words.txt'), 'r') as f:
        positive_words.update([line.strip() for line in f if line.strip() and not line.startswith(';')])

    with open(os.path.join(MASTER_DICT_PATH, 'negative-words.txt'), 'r') as f:
        negative_words.update([line.strip() for line in f if line.strip() and not line.startswith(';')])
    
    return positive_words, negative_words

def clean_and_tokenize(text, stopwords_set):
    # Lowercase the text and remove stopwords
    words = word_tokenize(text.lower())
    cleaned_words = [word for word in words if word.isalpha() and word not in stopwords_set]
    
    # Sentence tokenization
    sentences = sent_tokenize(text)
    
    return cleaned_words, sentences

def compute_sentiment_scores(cleaned_words, positive_words, negative_words):
    positive_score = sum(1 for word in cleaned_words if word in positive_words)
    negative_score = sum(1 for word in cleaned_words if word in negative_words)
    
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(cleaned_words) + 0.000001)
    
    return positive_score, negative_score, polarity_score, subjectivity_score

def compute_readability_metrics(text):
    avg_sentence_length = textstat.sentence_count(text)
    fog_index = textstat.gunning_fog(text)
    return avg_sentence_length, fog_index

def count_syllables(word):
    vowels = "aeiou"
    count = sum(1 for char in word if char in vowels)
    if word.endswith("es") or word.endswith("ed"):
        count -= 1
    return count

def count_complex_words(words):  
    complex_words_count = sum(1 for word in words if count_syllables(word) > 2)
    return complex_words_count

def count_personal_pronouns(text):
    pronouns = re.findall(r'\b(I|we|my|ours|us)\b', text, re.I)
    return len(pronouns)

def avg_word_length(cleaned_words):
    total_characters = sum(len(word) for word in cleaned_words)
    return total_characters / len(cleaned_words) if cleaned_words else 0

def analyze_article(text, positive_words, negative_words, stopwords_set):
    cleaned_words, sentences = clean_and_tokenize(text, stopwords_set)
    positive_score, negative_score, polarity_score, subjectivity_score = compute_sentiment_scores(cleaned_words, positive_words, negative_words)
    avg_sentence_length, fog_index = compute_readability_metrics(text)
    complex_word_count = count_complex_words(cleaned_words)
    word_count = len(cleaned_words)
    syllable_per_word = sum(count_syllables(word) for word in cleaned_words) / word_count
    personal_pronoun_count = count_personal_pronouns(text)
    avg_word_len = avg_word_length(cleaned_words)
    
    return {
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

def main():
    stopwords_set = load_stopwords()
    positive_words, negative_words = load_master_dictionary()
 
    # Analyze the provided text
    analysis_results = analyze_article(text, positive_words, negative_words, stopwords_set)
    
    # Convert to DataFrame and save to Excel
    output_df = pd.DataFrame([analysis_results])
    output_df.to_excel(OUTPUT_PATH, index=False)
    
    print(f"Analysis complete! Results saved to {OUTPUT_PATH}")

# Run the main function
if __name__ == "__main__":
    main()
