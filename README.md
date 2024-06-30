# Amazon Reviews Sentiment Analysis

This is a Streamlit app for sentiment analysis of Amazon reviews using a pre-trained model.

## Setup Instructions

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/my-streamlit-project.git
    cd my-streamlit-project
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Download NLTK data (if not included in the project):
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

5. Run the Streamlit app:
    ```sh
    streamlit run sentiment_analysis.py
    ```

## Files

- `sentiment_analysis.py`: Main Streamlit app script.
- `Sentiment_Analysis.h5`: Trained model file.
- `tokenizer_(1).json`: Tokenizer file.
- `requirements.txt`: Python dependencies.

## Usage

Enter a review into the text box and click "Predict" to get the sentiment analysis result.
