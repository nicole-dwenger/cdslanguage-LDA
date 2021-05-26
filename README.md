# LDA Topic Modelling: What Are They TedTalking about?

[Description](#description) | [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion) | [Contact](#contact)

## Description
> This project relates to Assignment 5: (Un)Supervised Machine Learning

TedTalks have become popular as videos in which experts share their knowledge and ideas. What if, in 100 years someone finds all these videos, but has no idea what they are about. It would be tedious to watch all of them, to find out. Thus, it would be great if there was a way to figure out which topics these talks cover based on their contents. Knowing what different talks are about and when they were published, can also help to investigate how topics in these talks developed over time. Lastly, if someone was interested in a specific topic, they could find talks, which cover a given topic to a large extend. 

LDA topic modelling is an unsupervised machine learning algorithm, which can help to extract topics from unlabelled text documents. LDA builds on the assumption that words in documents are derived from different clusters of topics. Put differently, words cluster into different topics, and documents are made up of these words. Thus, the distribution of words (and which cluster they belong to) can help to determine the distribution of topics for each document. The aim of LDA is to reverse engineer the topics (clusters) from which the words were drawn. In this project, each of the talks is considered to be a document in a large corpus of documents (corpus of talks). The aim of this project is to address the following questions:  considered to be a document in a large corpus of documents (corpus of talks).

Specifically, this project aims to address the following questions: 
1. Is it possible to find distinct topics across the transcript of TedTalks? 
2. Are there any temporal developments in the topics of TedTalks?
3. Is it possible to get “representative” talks for each of the topics?


## Methods

### Data and Preprocessing
The data, which was used for this project is a dataset from [Kaggle](https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset), containing information and transcripts of 4005 English TedTalks from 2016 to 2020. To reduce processing time I only used talks from 2016-2020, leading to a total of 2167 talks. These transcripts were preprocessed with the following steps: 
 
The transcripts were preprocessed using the following steps: 
1. Remove all "non-content" aspects of the transcripts, e.g. *(Laugther)*, by removing everything that is in () or [].
2. Lowercase and tokenize all texts using gensim's simple_preprocess function.
3. Remove stopwords using the english stopword list from nltk.
4. Create bigrams and trigrams, using a minimum count of 3 and a threshold of 100. 
5. Extract only tokens which are NOUNS (as determined by spaCy’s POS tag)
6. Lemmatise all tokens.
6. Remove tokens using a self-defined list, which was developed by running multiple LDA analyses, and finding which words occurred often in many topics, and seem to contribute relatively little about the meaning of a topic. This list contained the following tokens: *people, world, talk, time, other, hundred, one, life, thousand, number, way, year, thing, story, day, lot, question, idea, word*.

### LDA Topic Modelling
For topic modelling, a LDAMulticore model was trained on the derived dictionary and corpus of tokens. The model was run for 15 topics and for 20 topics. Based on a higher coherence score for 15 topics (0.45) compared to 20 topics (0.43), the focus in the following will be on the model trained for 15 topics. However, output is provided for both models in the out/ directory of the GitHub repository. Optimally, an exploration of more coherence values for different number of topics should have been conducted, but was omitted for time and processing reasons. Using the derived topics and distributions of topics for each of the texts, the following outputs are generated to address the questions (1)-(3) outlined above: 

1. Metrics: The perplexity and coherence scores of the LDA model are stored. The focus will be on the coherence score, which is a measure of how similar words are in a topic.
2. Keywords: The top 10 keywords for each topic are extracted and their counts and weights are visualised in bar graphs. 
3. Dominant topics:  The dominant topic (based on the highest percentage of contribution) is appended to the original data frame. 
4. Representatives: For each topic, the names of the three talks which have the highest percentage of contribution are saved as representatives for a given topic. In other words, these should represent talks, where most words are coming from the given topic.
5. Topics over time: Using the topic distributions for each document and the published date, the development of topics is plotted with a 90 day rolling average.


## Repository Structure 
```
|-- data/
    |-- ted_talks_en.csv                # Raw data of ted talks
    
|-- out/                                # Directory for output, corresponding to scripts
    |-- LDA_15_topics/                  # Results of LDA topic modelling for 15 topics
        |-- LDA_metrics.txt             # Coherence and Perplexity scores
        |-- LDA_keywords.png            # Visualisation of weight and count of keywords for each topic
        |-- LDA_dominant_topics.csv     # Original .csv with appended columns of dominant_topic and topic_perc_contrib
        |-- LDA_representatives.txt     # 3 talks with highest topic_perc_contrib for each topic
        |-- LDA_topics_over_time.png    # Visualisation of topic development over time
    |-- LDA_20_topics/                  # Results of LDA topic modelling for 20 topics
        |-- ...

|-- src/
    |-- LDA_tedtalks.py                 # Script for LDA topic modelling of TedTalks

|-- utils/
    |-- LDA_utils.py                    # Utils containing preprocessing functions and LDA class, for main script
    
|-- README.md
|-- create_venv.sh                      # Bash script to create virtual environment
|-- requirements.txt                    # Dependencies, installed in virtual environment

```

## Usage 
**!** The scripts have only been tested on Linux, using Python 3.6.9. 

### 1. Cloning the Repository and Installing Dependencies
To run the scripts in this repository, I recommend cloning this repository and installing necessary dependencies in a virtual environment. The bash script `create_venv.sh` can be used to create a virtual environment called `venv_LDA` with all necessary dependencies, listed in the `requirements.txt` file. The following commands can be used:

```bash
# cloning the repository
git clone https://github.com/nicole-dwenger/cdslanguage-LDA.git

# move into directory
cd cdslanguage-LDA/

# install virtual environment
bash create_venv.sh

# activate virtual environment 
source venv_LDA/bin/activate
```

### 2. Data 
The raw data, which was originally downloaded from [Kaggle](https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset) is stored in the `data/` directory of this repository.  Thus, after cloning the repository, it is not necessary to retrieve any additional data.

### 3. Script for LDA Topic Modelling on TedTalk Transcripts
The script `LDA_tedtalks.py` preprocesses the TedTalk transcripts, following the steps above. Subsequently, it will conduct an LDA topic analysis on a given number of topics, and produce the output, described above. The script should be called from the `src/` directory:

```bash
# move into src 
cd src/

# run script with default parameters
python3 LDA_tedtalks.py

# run script for specified input data
python3 LDA_tedtalks.py -n 20
```

__Parameters:__

- `-i, --input_file`: *str, optional, default:*`../data/ted_talks_en.csv`\
   Path to the input file.
   
- `-n, --n_topics`: *int, optional, default:* `15`\
   Number of topics to extract.
 
- `-y, --year_above`: *int, optional, default:* `2015`\
   All talks which were published later than the given year. For 2015 this would be all talks from 2016-2020.  
   
__Output:__

- `LDA_metrics.txt`\
   File containing coherence and perplexity scores. 
   
- `LDA_keywords.png`\
   Visualisation of count and weight of keywords for each topic. 
   
- `LDA_dominant_topics.csv`\
   Origial .csv dataframe, with appended columns of dominant_topic and topic_perc_contib.
   
- `LDA_representatives.txt`\
   For each topic, the three talks with the highest contribution are printed. 
   
- `LDA_topics_over_time.png`\
   Visualisation of development of topics over time. 

  
## Results and Discussion 
Output of the scripts can be found in the corresponding directory in `out/`. The model trained for 15 topics had a perplexity score of -8.69 and a coherence measure of 0.46. Below the results are illustrated and discussed in relation to the three questions posed above: 

__1. Is it possible to find distinct topics across the transcript of TedTalks?__

To answer this question, one can inspect the keywords each of the topics, to see if they relate to the same topic and thus compose a distinct topic. Below are displayed the keywords of the 15 derived topics: 

![](https://github.com/nicole-dwenger/cdslanguage-LDA/blob/master/out/LDA_15_topics/LDA_keywords.png)

Based on these keywords, I could imagine the following fitting topic names: 

- Topic 0: Education
- Topic 1: Climate & Environment
- Topic 2: Technology
- Topic 3: Human Body & Animals
- Topic 4: Psychology & Neuroscience
- Topic 5: Drugs & Sex
- Topic 6: Gender Roles
- Topic 7: Literature & Art
- Topic 8: Health Care
- Topic 9: Governance & Economy
- Topic 10: Family
- Topic 11: Medicine
- Topic 12: Communities & Immigration
- Topic 13: Earth & Space
- Topic 14: Love & Music

Overall, it seems that for some clusters e.g. Psychology & Neuroscience (4) or Health Care (8) it was quite easy to see how the keywords fit into a category. For others, e.g. Human Body & Animals (3) it was more difficult, since words seemed quite different (animal, robot, egg).

__2. Are there any temporal developments in the topics of TedTalks?__

Below, the temporal development of topics from 2016-2020 are displayed. Looking at this development, there seem to be a few topics which are fairly dominant across time, such as Family (10), Climate & Environment (1) and Technology (2). For me, this makes intuitively sense, as these topics have been quite dominant inn the past years. Further, it may also be that many talks relate to these topics in some way.  For the topic of Planet Earth  & Space (13) there seems to be an increase around 2019, which could be due to an increased interest in exploring Mars or other planets. This plot could have been improved by changing its dimensions, i.e. making it wider.

![](https://github.com/nicole-dwenger/cdslanguage-LDA/blob/master/out/LDA_15_topics/LDA_topics_over_time.png)


__3. Is is possible to generate "recommendations" of talks for each of the topics?__ 

The full list of representative talks for each topic can be seen in this [file](https://github.com/nicole-dwenger/cdslanguage-LDA/blob/master/out/LDA_15_topics/LDA_representatives.txt), stored in the `out/` directory. Only a few examples will be discussed in the following. 
For Topic (4) Psychology & Neuroscience, the following titles were extracted: 'How stress affects your brain', 'How memories form and how we lose them', 'What happens when you have a concussion?'. Without knowing anything about the talks, these titles all seem to relate to the given topic name.
For Topic (13) Planet Earth & Space the following titles were extracted: 'A needle in countless haystacks: Finding habitable worlds', '3 moons and a planet that could have alien life', 'Could the Earth be swallowed by a black hole?'. Again, based on their titles, these talks all seem to fit into a theme of exploring space and finding life on another planet. 
For Topic (10) Family the following titles were extracted: 'Why we should take laughter more seriously', 'Love others to love yourself', "I grew up in the Westboro Baptist Church. Here's why I left". Here, it seems that the given title of Family does not really fit, but rather these titles fit into a category of Life Advice or Lifestyle. However, the keywords of child, family, friend, love, conversation seem to fit to the talks, as they relate to the interaction with others. 

## Contact
If you have any questions, feel free to contact me at 201805351@post.au.dk.

