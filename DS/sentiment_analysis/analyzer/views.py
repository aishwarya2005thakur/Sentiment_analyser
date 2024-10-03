from django.shortcuts import render,HttpResponse
import string
import nltk
from collections import Counter
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def analyze_text(text):
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

    tokenized_words = word_tokenize(cleaned_text, "english")
    final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

    lemma_words = [WordNetLemmatizer().lemmatize(word) for word in final_words]

    emotion_list = []
    with open('static/emotions.txt', 'r') as file: # Make sure emotions.txt is in the same directory
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')
            if word in lemma_words:
                emotion_list.append(emotion)

    emotion_counts = Counter(emotion_list) 

    # Sentiment Analysis (using VADER)
    sentiment_scores = SentimentIntensityAnalyzer().polarity_scores(cleaned_text) 

    return sentiment_scores, emotion_counts

def analyze_sentiment(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        sentiment_scores, emotion_counts = analyze_text(text)

        # Generate the chart
        plt.figure(figsize=(8, 6))
        plt.bar(emotion_counts.keys(), emotion_counts.values())
        plt.xlabel('Emotions')
        plt.ylabel('Count')
        plt.title('Emotion Distribution')
        plt.xticks(rotation=45)
        
        # Encode the plot to embed in HTML
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        context = {
            'sentiment_scores': sentiment_scores,
            'emotion_counts': dict(emotion_counts), # Pass as a dictionary
            'chart_image': f'data:image/png;base64,{image_base64}'
        }
        return render(request, 'result.html', context)
    else:
        return render(request, 'analyzer.html')
