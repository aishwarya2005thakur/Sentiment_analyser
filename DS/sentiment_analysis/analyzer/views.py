from django.shortcuts import render
import string
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Ensure necessary NLTK resources are downloaded
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def analyze_text(text):
    # Text Preprocessing
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

    # Tokenization and Stopword Removal
    tokenized_words = word_tokenize(cleaned_text, "english")
    final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

    # Lemmatization
    lemma_words = [WordNetLemmatizer().lemmatize(word) for word in final_words]

    # Sentiment Analysis using VADER
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment_analyzer.polarity_scores(cleaned_text)

    return sentiment_scores

def sentiment_analyse(sentiment_scores):
    if sentiment_scores['neg'] > sentiment_scores['pos']:
        return "Negative Sentiment"
    elif sentiment_scores['neg'] < sentiment_scores['pos']:
        return "Positive Sentiment"
    else:
        return "Neutral Sentiment"

def generate_graph(sentiment_scores):
    # Prepare data for the graph
    categories = ['Positive', 'Neutral', 'Negative']
    scores = [sentiment_scores['pos'], sentiment_scores['neu'], sentiment_scores['neg']]

    # Create the bar graph
    plt.figure(figsize=(8, 5))
    plt.bar(categories, scores, color=['green', 'yellow', 'red'])
    plt.ylabel('Scores')
    plt.title('Sentiment Analysis Scores')
    plt.ylim(0, 1)  # VADER scores are between 0 and 1

    # Save the graph to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return f"data:image/png;base64,{graph_image}"

def analyze_sentiment(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        sentiment_scores = analyze_text(text)
        overall_sentiment = sentiment_analyse(sentiment_scores)

        # Generate the graph
        graph_image = generate_graph(sentiment_scores)

        # Context to send to the template
        context = {
            'text': text,
            'sentiment_scores': sentiment_scores,
            'overall_sentiment': overall_sentiment,
            'graph_image': graph_image,
        }
        return render(request, 'result.html', context)

    return render(request, 'analyzer.html')
