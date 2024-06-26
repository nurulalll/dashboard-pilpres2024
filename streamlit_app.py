import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator, exceptions
from deep_translator.exceptions import TooManyRequests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)

def load_data(dataset):
    if dataset.name.endswith('.xlsx'):
        df = pd.read_excel(dataset)
    elif dataset.name.endswith('.csv'):
        df = pd.read_csv(dataset)
    else:
        st.error("Unsupported file type. Please upload a .xlsx atau .csv file.")
        return None
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

    st.markdown('<div class="center"><h1>Sentimen Analisis Pemilihan Presiden RI tahun 2024-2029</h1></div>', unsafe_allow_html=True)
    st.image('https://img.okezone.com/content/2018/02/19/337/1861446/sejarah-pemilu-dari-masa-ke-masa-cyx7Flt69A.jpg', width=400)

def display_wordcloud(df):
    tweet_text = ' '.join(df['Tweet'].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=550, background_color='white').generate(tweet_text)
    plt.figure(figsize=(12, 15))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt, use_container_width=True)

def display_sentiment_distribution(df):
    sentiment_counts = df['sentimen'].value_counts()
    fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index)
    fig.update_layout(width=800, height=300)  # Reduce margins to bring the chart closer to content
    st.plotly_chart(fig, use_container_width=True)

def display_top_usernames(df):
    top_usernames = df['username'].value_counts().head(10)
    fig = px.bar(top_usernames, x=top_usernames.index, y=top_usernames.values,
                 labels={'x':'Username', 'y':'Count'})
    fig.update_layout(width=800, height=550, xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)

def display_top_locations(df):
    top_locations = df['location'].value_counts().head(10)
    fig = px.bar(top_locations, y=top_locations.index, x=top_locations.values, orientation='h',
                 labels={'y':'Location', 'x':'Count'})
    fig.update_layout(width=800, height=550, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def translate_to_english(text):
    try:
        translator = GoogleTranslator(source='auto', target='en')
        translated_text = translator.translate(text)
        return translated_text
    except TooManyRequests:
        st.error("Terlalu banyak permintaan telah dilakukan. Silakan coba lagi nanti.")
        return "Translation service unavailable."
    except Exception as e:
        st.error(f"Translation error: {e}")
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

def text_sentiment():
    st.title('Analisis Text Sentiment')
    
    option = st.radio("Pilih metode input:", ["Text Sentiment", "Upload file"])

    if option == "Text Sentiment":
        input_text = st.text_area("Masukkan kalimat yang ingin dianalisis:")
        button = st.button("Analisis")
        if button:
            with st.spinner("Sedang menganalisis..."):
                result = sentiment_analysis(input_text)
            if result['label'] == 'ERROR':
                st.write("Terjadi kesalahan dalam menerjemahkan teks.")
            else:
                sentiment_color = "green" if result['label'] == 'POSITIVE' else "red" if result['label'] == 'NEGATIVE' else "blue"
                st.write(f"**Sentimen:** <span style='color:{sentiment_color}; font-weight:bold;'>{result['label']}</span>", 
                         f"**Score:** {result['score']:.2f}", 
                         unsafe_allow_html=True)
    elif option == "Upload file":
        uploaded_file = st.file_uploader("Upload file .xlsx atau .csv", type=["xlsx", "csv"])
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                if 'Tweet' in df.columns:
                    st.write("Data berhasil diunggah. Berikut beberapa baris pertama data:")
                    st.write(df.head())

                    button = st.button("Analisis Sentimen")
                    if button:
                        with st.spinner("Sedang menganalisis..."):
                            df['sentimen'] = df['Tweet'].apply(lambda x: sentiment_analysis(x)['label'])
                            st.write("Analisis Sentimen selesai. Berikut hasilnya:")
                            st.write(df.head())
                            display_sentiment_distribution(df)
                else:
                    st.error("File tidak memiliki kolom 'Tweet'.")

def display_visualizations(df, visualization_options):
    st.title("")
    num_options = len(visualization_options)

    if num_options > 0:
        num_cols = min(num_options, 2)  # Display max 2 visualizations side-by-side
        cols = st.columns(num_cols)

        for i, option in enumerate(visualization_options):
            with cols[i % num_cols]:  # Distribute options across columns
                st.subheader(option)
                if option == "Word Cloud":
                    display_wordcloud(df)
                elif option == "Sentiment Distribution":
                    display_sentiment_distribution(df)
                elif option == "Top Usernames":
                    display_top_usernames(df)
                elif option == "Top Locations":
                    display_top_locations(df)
                st.write("<hr>", unsafe_allow_html=True)  # Separate visualizations with a horizontal line
    else:
        st.warning("Please select at least one visualization option.")

def main():
    st.set_page_config(page_title='Sentiment Analysis')

    header()

    dataset_names = {
        "Anies-CakImin": "Dataset_Anies-CakImin.xlsx",
        "Prabowo-Gibran": "Dataset_Prabowo-Gibran.xlsx",
        "Ganjar-Mahfud": "Dataset_Ganjar-Mahfud.xlsx"
    }

    selected_datasets = st.multiselect("Select Datasets", list(dataset_names.keys()))

    dfs = [load_data(dataset_names[dataset]) for dataset in selected_datasets if dataset in dataset_names]
    df = pd.concat(dfs) if dfs else None

    page = st.radio("Menu", ["Visualizations", "Text Sentiment"])

    if page == 'Visualizations':
        if df is not None:
            visualization_options = st.multiselect("Choose Visualizations", ["Word Cloud", "Sentiment Distribution", "Top Usernames", "Top Locations"])
            if visualization_options:
                display_visualizations(df, visualization_options)
            else:
                st.warning("Please select at least one visualization option.")
        else:
            st.warning("Please select a dataset.")

    elif page == 'Text Sentiment':
        text_sentiment()

if __name__ == "__main__":
    main()
