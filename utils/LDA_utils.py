#!/usr/bin/env python

"""
Utility functions and class for LDA topic modelling on TedTalks.

Functions: 
  - load_data: load tedtalk data and filter based on the year
  - extract_grams: extract bigram and trigram models from input text
  - process_words: process words of each text, i.e. remove stopwords, create bigrams/trigrams, 
                     extract tokens with POS tag, lemmatize

Class LDA_Model: 
  - train_model: train LDA model to extract n topics
  - evaluate_model: evaluate LDA model, using perplexity and coherence
  - save_metrics: save perplexity and coherence metrics in .txt file
  - plot_keywords: plot the count and weight of keywords for each topic
  - save_dominant_topics: append the dominant topic to each talk in the original .csv
  - save_representatives: for each topic save the 3 talk titles with the highest topic contribution
  - plot_topics_over_time: plot the distribution of topics over time
"""

# LIBRARIES ---------------------------------------------------

# Basics
import os
import pandas as pd
from math import ceil
from collections import Counter
from pprint import pprint

# Visualisation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
import seaborn as sns

# Spacy NLP
import spacy
nlp = spacy.load("en_core_web_sm")

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")


# FUNCTIONS -------------------------------------------------------

def load_data(input_file, year_above):
    """
    Load raw data, select relevant columns, and filter by input date
    Input: 
      - input_file: path to input file
      - year_above: above this year, all talks will be selected
    Output: 
      - df: with selected columns, years and fixed index
    """
    df = pd.read_csv(input_file)
    # Select only relevant columns
    df = df.loc[:, ['title', 'published_date', 'topics', 'description', 'transcript']]
    # Turn data column into datetime format
    df["published_date"] = pd.to_datetime(df['published_date'])
    # Filter based on date
    df = df[df["published_date"].dt.year > year_above]
    # Reset index 
    df = df.reset_index()
    
    return df
    
def extract_grams(texts, min_count, threshold):
    """
    Create bigrams and trigrams based on text inputs. 
    Input: 
      - texts: lists of text documents
      - min_count: ignore word which occur less than this value
      - threshold: threshold score for accepting phrase of words, higher threshold, less phrases
    Returns:
      - bigram_mod, trigram_mod: models for bigrams and trigrams in text
    """
    # Define parameters for bigrams
    bigram = gensim.models.Phrases(texts, min_count=min_count, threshold=threshold)
    # Use bigrams to define trigrams
    trigram = gensim.models.Phrases(bigram[texts], threshold=threshold)
    # Define bigram model
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # Define trigram model
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    return bigram_mod, trigram_mod
   
def process_words(texts, nlp, bigram_mod, trigram_mod, stop_words, allowed_postags=['NOUN', "ADJ", "VERB", "ADV"]):
    """
    Process words: remove stopwords, create bigrams and trigrams, extract tokens of post-tags and lemmatize tokens
    Input: 
      - texts: list of text documents
      - nlp: language model, e.g. spacy
      - bigram_mod: bigram model
      - trigram_mod: trigram model
      - stop_words: stop words to remove from tokens
      - allowed_postags: POS to keep for LDA
    Return: 
      - texts_out: processed texts: list of lemmatised tokens of defined POS tag
    """
    # Remove stopwords and use gensim simple process
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    # Create bigrams
    texts = [bigram_mod[doc] for doc in texts]
    # Create trigrams
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    # Create empty target output list
    processed_texts = []
    # For each text
    for text in texts:
        # Apply nlp
        doc = nlp(" ".join(text)) 
        # If token is in POS, lemmatise it and append to list
        processed_texts.append([token.lemma_ for token in doc if token.pos_ in allowed_postags]) 
        
    return processed_texts
        
# LDA CLASS -------------------------------------------------------

class LDA_Model():
    """
    LDA Topic Modelling Class
    Initialised with: 
      - preprocessed texts: list of processed tokens for each text
      - corpus: corpus of tokens of texts
      - dictonary: dictionary of tokens of texts
      - n_topics: number of topics       
    Fucntions: 
      - train_model: train LDA model
      - evaluate_model: get perplextiy and coherence scores
      - save_metrics: save perplextiy and coherence scores in .txt file
      - plot_keywords: plot the count and weight of keywords for each topic
      - save_dominant_topics: append the dominant topic to each talk in the original .csv
      - save_representatives: for each topic save the 3 talk titles with the highest topic contribution
      - plot_topics_over_time: plot the distribution of topics over time
    """
    
    def __init__(self, processed_texts, corpus, dictionary, n_topics):
        """
        Save input variables, assigned to the class
        """
        # Variables defined when initialising class
        self.processed_texts = processed_texts
        self.corpus = corpus
        self.dictionary = dictionary
        self.n_topics = n_topics
      
        # Varaibles, which are added through functios
        self.model = None
        self.metrics = None
        self.df_out = None
        
    def train_model(self):
        """
        Train LDA model on processed texts, and extract defined number of topics
        """
        self.model = gensim.models.LdaMulticore(corpus=self.corpus,id2word=self.dictionary,
                                                num_topics=self.n_topics, 
                                                random_state=100,
                                                chunksize=10,
                                                passes=10,
                                                iterations=100,
                                                per_word_topics=True, 
                                                minimum_probability=0.0)
        
     
    def evaluate_model(self):
        """
        Evaluate LDA, i.e. get coherence and perplexity
        """
        # Get perplexity
        perplexity = self.model.log_perplexity(self.corpus)
        
        # Get coherence
        coherence =  CoherenceModel(model=self.model, 
                                    texts=self.processed_texts, 
                                    dictionary=self.dictionary, 
                                    coherence='c_v').get_coherence()
        
        # Append metrics to self
        self.metrics = {"Perplextity": perplexity, "Coherence": coherence}
        
        # Print metrics
        print(f"[INFO] LDA model metrics: {self.metrics}")
        
    def save_metrics(self, output_directory, filename):
        """
        Save metrics and keywords of topics to .txt file
        """
        # Define output path
        out_path = os.path.join(output_directory, filename)
        
        # Save metrics to file
        with open(out_path, "w") as file:
            print(f"Performence metrics of LDA for {self.n_topics}:\n", file=file)
            print(self.metrics, file=file)
            
    def plot_keywords(self, output_directory, filename):
        """
        Create plot of counts ad weights of keywords for each topic,
        save as one .png file (function adjusted from class content)
        """
        # Get topic names
        topics = self.model.show_topics(formatted=False, num_topics=self.n_topics)
        # Word list of data
        data_flat = [w for w_list in self.processed_texts for w in w_list]
        # Initialise counter 
        counter = Counter(data_flat)
        
        # Get weights and countns for words
        word_counts = []
        for i, topic in topics:
            for word, weight in topic:
                word_counts.append([word, i, weight, counter[word]])
                
        # Create dataframe
        df = pd.DataFrame(word_counts, columns = ['word', 'topic_id', 'weight', 'word_count'])
        
        # Get the maximum count and weight, to adjust plot axes
        max_wc = max(df.word_count) + 200
        max_weight = max(df.weight) + 0.01
        # Define number of rows based on number of topics
        n_rows = (ceil(self.n_topics/3))
        
        # Initialise plot, by defining subplots
        fig, axes = plt.subplots(n_rows, 3, figsize=(18,(n_rows*3)), sharey=True, dpi=160)
        # Define colours, repeat colors 3 times to have enough for 20 topics
        cols = ([color for name, color in mcolors.TABLEAU_COLORS.items()] + 
                [color for name, color in mcolors.TABLEAU_COLORS.items()] +
                [color for name, color in mcolors.TABLEAU_COLORS.items()])
        
        # For each subplot:
        for i, ax in enumerate(axes.flatten()):
            # Plot word counts
            ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], 
                   color=cols[i], width=0.5, alpha=0.3, label='Word Count')
            # Plot weights
            ax_twin = ax.twinx()
            ax_twin.bar(x='word', height="weight", data=df.loc[df.topic_id==i, :], 
                        color=cols[i], width=0.2, label='Weights')
            # Plot labels
            ax.set_ylabel('Word Count', color=cols[i])
            # Set y-axis limits
            ax_twin.set_ylim(0, max_weight); ax.set_ylim(0, max_wc)
            ax.tick_params(axis='y', left=False)
            # Set x-axis labels
            ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
            # Set title
            ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
            # Set legend
            ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

        # Add padding
        fig.tight_layout(w_pad=2)
        # Add overall title
        fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
        
        # Save figure
        plt.savefig(os.path.join(output_directory, filename))
        # Clear figure
        plt.clf()
        
    def save_dominant_topics(self, df, output_directory, filename):
        """
        Append to the original dataframe the dominant topic and the topic_perc_contrib
        and save as .csv file (function adjusted from class content)
        """
        # Initialise empty dataframe to save topic and topic_perc_contrib
        df_topics = pd.DataFrame()
        # Get the dominant topic, percentage contibution and keyword for each text
        for i, row_list in enumerate(self.model[self.corpus]):
            row = row_list[0] if self.model.per_word_topics else row_list  
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0: # = dominant topic
                    wp = self.model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    df_topics = df_topics.append(pd.Series([int(topic_num), 
                                                     round(prop_topic,4)]), ignore_index=True)
                else: 
                    break
         
        # Define column names of topics dataframe
        df_topics.columns = ['dominant_topic', 'topic_perc_contrib']
        # Reset index to append to original dataframe
        df_topics = df_topics.reset_index()
        # Append topics to the original dataframe based on inddexe
        self.df_out = pd.concat([df, df_topics], axis=1)

        # Save in output
        self.df_out.to_csv(os.path.join(output_directory, filename))
        
    def save_representatives(self, output_directory, filename):
        """
        From the topic distributions of the dominant topic for each topic, 
        select all topic with a given dominant topic, get those documents,
        which have the highest contribution, and save as "representatives" in .txt
        """
        # Define path to output file for recommendatios
        out_path = os.path.join(output_directory, filename)
        
        # Into the file
        with open(out_path, "w") as f:
            # Print a title
            print("Top 3 Talks for each Topic\n", file = f)
            # For each of the topic print top 5 titles
            for topic in set(self.df_out["dominant_topic"].tolist()):
                # Create dataframe containing only the data for the given topic
                topic_df = self.df_out[self.df_out["dominant_topic"] == topic]
                # Sort by contribution
                topic_df = topic_df.sort_values("topic_perc_contrib", ascending = False)
                # Print the topics with the highest contribution to the file
                print(f"Topic: {topic}", file = f)
                print(f"{topic_df.title.to_list()[:3]}\n", file = f)
     
    def plot_topics_over_time(self, output_directory, filename):
        """
        Using the percental distiributions of topics for each document 
        and information about the publishing date, plot development
        of topics over time with a 90d rolling average, save as .png
        """
        # Get the topic distributions for each of the documents
        values = list(self.model.get_document_topics(self.corpus))
        
        # Create matrics of topics in rows and documents as columns
        split_topics = []
        for entry in values:
            topic_prevelance = []
            for topic in entry:
                topic_prevelance.append(topic[1])
            split_topics.append(topic_prevelance)
            
        # Save in dataframe to plot
        topic_distributions = pd.DataFrame(map(list,zip(*split_topics)))
        # Get the dates from the original dataframe and append as columnames
        dates = self.df_out.published_date.tolist()
        topic_distributions.columns = pd.to_datetime(dates) 
        # Transpose the dataframe, and make index to datetime for rolling mean
        transposed_df = topic_distributions.T
        transposed_df.index = pd.to_datetime(transposed_df.index)
        # Compute rolling mean over 90 days
        rolling = transposed_df.rolling('90D').mean()
        # Add date as column instead of inddex
        rolling["Date"] = rolling.index
        # Turn data from wide into long format for multiline plot
        reshaped = rolling.melt("Date", var_name="Topic", value_name="Percentage")
        # Make topic column as str
        reshaped['Topic'] = reshaped['Topic'].astype(str)
        # Plot lineplot
        ax = sns.lineplot(x="Date", y="Percentage", hue="Topic", style = "Topic", data=reshaped)
        # Set y-limit to "zoom in"
        ax.set_ylim(0, 0.35)
        # Set title
        ax.set_title("Distribution of Ted Talk Topics Over Time (2016-2020)")
        # Save lineplot
        plt.savefig(os.path.join(output_directory, filename))       
        
if __name__=="__main__":
    pass