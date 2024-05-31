import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
from deep_translator import GoogleTranslator, exceptions
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

def load_data(dataset_name):
    # Load dataset
    df = pd.read_excel(dataset_name)
    return df

def header():
    st.markdown(
        """
        <style>
        .center {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="center"><h1>Prediksi Pilihan Presiden RI tahun 2024-2029</h1></div>', unsafe_allow_html=True)
    st.image('https://img.okezone.com/content/2018/02/19/337/1861446/sejarah-pemilu-dari-masa-ke-masa-cyx7Flt69A.jpg', width=400)

def display_wordcloud(df):
    tweet_text = ' '.join(df['Tweet'].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=550, background_color ='white').generate(tweet_text)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt, clear_figure=False)

def display_sentiment_distribution(df):
    sentiment_counts = df['sentimen'].value_counts()
    fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index)
    fig.update_layout(width=600, height=400)
    st.plotly_chart(fig, use_container_width=True)

def display_top_usernames(df):
    top_usernames = df['username'].value_counts().head(10)
    fig = px.bar(top_usernames, x=top_usernames.index, y=top_usernames.values,
                 labels={'x':'Username', 'y':'Count'})
    fig.update_layout(width=800, height=400, xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)

import streamlit as st
from deep_translator import GoogleTranslator, exceptions
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

def translate_to_english(text):
    try:
        translator = GoogleTranslator(source='auto', target='en')
        translated_text = translator.translate(text)
        return translated_text
    except exceptions.TranslationNotFound:
        return "Translation service unavailable."

def sentiment_analysis(text):
    # Translate teks ke bahasa Inggris
    english_text = translate_to_english(text)
    
    if english_text == "Translation service unavailable.":
        return {'label': 'ERROR', 'score': 0.0}

    # Membuat instance dari SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # Melakukan analisis sentimen pada teks yang telah diterjemahkan
    scores = analyzer.polarity_scores(english_text)
    compound_score = scores['compound']

    # Menentukan label sentimen berdasarkan nilai compound
    if compound_score >= 0.05:
        label = 'POSITIVE'
    elif compound_score <= -0.05:
        label = 'NEGATIVE'
    else:
        label = 'NEUTRAL'

    return {'label': label, 'score': compound_score}


def display_visualizations(df, visualization_options):
    st.title("Visualizations")
    if len(visualization_options) > 1:  # Jika lebih dari satu opsi dipilih
        num_cols = 2
        cols = st.columns(num_cols)
        
        for i, option in enumerate(visualization_options):
            with cols[i % num_cols]:  # Menggunakan modulus untuk mengatur opsi ke kolom yang benar
                st.subheader(option)
                if option == "Word Cloud":
                    display_wordcloud(df)
                elif option == "Sentiment Distribution":
                    display_sentiment_distribution(df)
                elif option == "Top Usernames":
                    display_top_usernames(df)
                if i < len(visualization_options) - 1:
                    st.write("<hr>", unsafe_allow_html=True)  # Memisahkan visualisasi dengan garis horizontal
    else:  # Jika hanya satu opsi yang dipilih
        option = visualization_options[0]
        st.subheader(option)
        if option == "Word Cloud":
            display_wordcloud(df)
        elif option == "Sentiment Distribution":
            display_sentiment_distribution(df)
        elif option == "Top Usernames":
            display_top_usernames(df)

def main():
    st.set_page_config(page_title='Sentiment Analysis Dashboard')

    header()

    dataset_names = {
        "Anies-CakImin": "Dataset_Anies-CakImin.xlsx",
        "Prabowo-Gibran": "Dataset_Prabowo-Gibran.xlsx",
        "Ganjar-Mahfud": "Dataset_Ganjar-Mahfud.xlsx"
    }

    selected_datasets = st.multiselect("Select Datasets", list(dataset_names.keys()))

    page = st.radio("Navigate", ["Visualizations", "Text Sentiment"])

    dfs = [load_data(dataset_names[dataset]) for dataset in selected_datasets]
    df = pd.concat(dfs) if dfs else None

    if page == 'Visualizations':
        if df is not None:
            visualization_options = st.multiselect("Choose Visualizations", ["Word Cloud", "Sentiment Distribution", "Top Usernames"])
            if visualization_options:
                display_visualizations(df, visualization_options)
            else:
                st.warning("Please select at least one visualization option.")

    elif page == 'Text Sentiment':
        text_sentiment()

if __name__ == "__main__":
    main()
