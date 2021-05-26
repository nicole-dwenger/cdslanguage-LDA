#!/usr/bin/env python

"""
LDA Topic Modelling on Transcripts of English Ted Talks from 2016-2020.

Steps: 
  - Load data, select talks from 2016-2020, remove non-content parts of the transcripts
  - Extract bigrams and trigrams of texts
  - Preprocess texts: tokenize, remove  stopwords, create bigrams and trigrams, extract only those of POS tag, lemmatize
  - Remove tokens of self-defined stop-word list
  - Create dictionary and corpus
  - Train and evaluate LDA
  - Create outputs: metrics, visualisation of keywords, append dominant topics of documents, 
    save representatives of each topic, plot topics over time
    
Input: 
  - -i, --input_file, str, path to input file of ted talks, optional, default: ../data/ted_talks_en.csv
  - -n, --n_topics, int, number of topics to extract, optional, default: 15
  - -y, --year_above, int, above which year should ted talks be extracted, default: 2015

Outputs saved in out/LDA_{n_topics}_topics/:
  - LDA_metrics.txt: perplexity and coherence score, also printed to command line
  - LDA_keywords.png: visualisation of count and weight of keywords of each topics
  - LDA_dominant_topics.csv: original dataframe, which appended columns of "dominant_topic" and "topic_perc_contib"
  - LDA_representatives.txt: file with titles of 3 talks for each topic, which had the highest topic contribution
  - LDA_topics_over_time.png: temporal development of topics over time  
"""

# LIBRARIES ----------------------------------------------------------

# Basics
import os
import re
import argparse
import pandas as pd

# NLP
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import spacy
nlp = spacy.load("en_core_web_sm")
import gensim.corpora as corpora

# Utils
import sys
sys.path.append(os.path.join(".."))
from utils.LDA_utils import (load_data, extract_grams, process_words, LDA_Model)


# MAIN FUNCTION -------------------------------------------------------

def main():
    
    # --- ARGUMENT PARSER AND OUTPUT DIRCETORY ---
    
    # Initialise argument parser
    ap = argparse.ArgumentParser()
    # Add input option for input file
    ap.add_argument("-i", "--input_file", type=str, required=False, default = "../data/ted_talks_en.csv")
    # Add input option for the number of topics
    ap.add_argument("-n", "--n_topics", type=int, required=False, default=15)
    # Add input option for the years to select talks froms
    ap.add_argument("-y", "--year_above", type=int, required=False, default=2015)
    
    # Retrieve arguments
    args = vars(ap.parse_args())
    input_file = args["input_file"]
    n_topics = args["n_topics"]
    year_above = args["year_above"]
    
    # Prepare output directory
    output_directory = os.path.join("..", "out", f"LDA_{n_topics}_topics")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # -- PREPARE DATA ---
    
    print(f"[INFO] Initialising LDA-topic modelling of {input_file}, for {n_topics} topics.")
    
    # Load dataframe
    df = load_data(input_file, year_above=year_above)
    
    # Save texts column (here description) in list
    texts = df["transcript"].to_list()
    # Remove from texts everything that is in () or []
    texts = [re.sub("[\(\[].*?[\)\]]", "", text) for text in texts]
    
    # --- PROCESS TEXTS ---
    
    print(f"[INFO] Processing texts ...")
    
    # Get bigrams and trigrams
    bigram_mod, trigram_mod = extract_grams(texts, min_count=3, threshold=100)
    
    # Process words: remove stopwords, create bigrams, trigrams, extract POS tokens, lemmatize
    processed_texts = process_words(texts, nlp, bigram_mod, trigram_mod, stop_words, allowed_postags=['NOUN'])
    
    # Remove self-defined list of words
    remove_words = ["people", "world", "talk", "time", "other", "hundred", "one", "life", "thousand", 
                    "number", "way","year", "thing", "story", "day", "lot", "question", "idea", "word"]
    processed_texts = [[word for word in text if word not in remove_words] for text in processed_texts]
    
    # Create dictionary and corpus from token lists
    dictionary = corpora.Dictionary(processed_texts)    
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    # -- TOPIC MODELLING --- 
    
    print(f"[INFO] Training LDA model ...")
    
    # Initialise LDA model class
    LDA = LDA_Model(processed_texts, corpus, dictionary, n_topics)

    # Train model
    LDA.train_model()

    # Evaluate model
    LDA.evaluate_model()

    # Save metrics
    LDA.save_metrics(output_directory, "LDA_metrics.txt")

    # Plot keywords
    LDA.plot_keywords(output_directory, "LDA_keywords.png")

    # Append the dominant topics and their contribution to the original dataframe
    LDA.save_dominant_topics(df, output_directory, "LDA_dominant_topics.csv")

    # Save the "representatives" for each topic
    LDA.save_representatives(output_directory, "LDA_representatives.txt")

    # Plot topics over time
    LDA.plot_topics_over_time(output_directory, "LDA_topics_over_time.png")
    
    print(f"All Done! Output is in {output_directory}")
    
      
if __name__=="__main__":
    main()
    