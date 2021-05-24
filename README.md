# LDA Topic Modelling: Ted Talks

[Description](#description) | [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion) | [Contact](#contact)

## Description
> This project relates to Assignment 5: (Un)Supervised Machine Learning

Ted Talks have become quite popular as videos of experts sharing their knowledge and ideas. What if, in 100 years someone finds all these videos, but has no information on what they are about. It would be tedious to watch all of them, to find out what they are about. Thus, it would be great if there was a way to figure out which topics these talks cover only based on their contents. Knowing what different talks are about and when they were published, can also allow to investigate how topics in these talks developed over time. Lastly, if someone was interested in a specific topic, they could find talks, which cover this topic to a large extend. 

LDA topic modelling is an unsupervised machine learning algorithm, which can help to extract topics from unlabelled text documents. LDA builds on the assumption that words in documents are derived from different clusters of topics. Thus words belong to topics, documents are made up of different words, and these words then determine the distribution of topics for each document. The aim of LDA is to reverse engineer the topics (clusters) of which the words are drawn. In this project, each of the talks is considered to be a document in a large corpus of documents (corpus of talks).

Specifically, the aim of this project is to address the following questions: 
1. Is it possible to find distinct topics across the transcript of Ted Talks? 
2. Are there any temporal developments in the topics of Ted Talks?
3. Is it possible to get “representative” talks for each of the topics?


## Methods

### Data and Preprocessing
The data, which was used for this project is a dataset from [Kaggle](https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset), which contains information and transcripts of 4005 English Ted Talks from 2016 to 2020. For this project, I only used talks from 2016-2020, to reduce processing time, which led to a total of 2167 talks. 
 
The transcripts were preprocessed using the following steps: 
1. Remove all "non-content" aspects of the transcripts, e.g. *(Laugther)*, by removing everything that is in "()" or "[]".
2. Lowercase and tokenize all texts using gensim's `simple_process` function.
3. Remove stopwords using the english stopword list from nltk.
4. Create bigrams and trigrams, using a minimum count of 3 and a threshold of 100. 
5. Extract only tokens which are NOUNS (as determined by spaCy’s POS tag)
6. Lemmatise all tokens.
6. Remove tokens using a self-defined list, which was developed by running multiple LDA analyses, and finding which words occurred often in many topics, and seem to contribute relatively little about the meaning of a topic. This list contained the following tokens: "people", "world", "talk", "time", "other", "hundred", "one", "life", "thousand", "number", "way", "year", "thing", "story", "day", "lot", "question", "idea", "word".

### LDA Topic Modelling
For topic modelling, an LDAMulticore model was trained on the derived dictionary and corpus of tokens. The model was run for 15 topics and for 20 topics. Based on a slightly higher coherence score for 15 topics (0.46) compared to 20 topics (0.45), the focus in the results section will focus those of 15 topics. However, output is provided for both models in the `out/` directory. Optimally, an exploration of more coherence values for different number of topics should have been conducted, but was omitted due to time and processing reasons. Using the derived topics and distributions of topics for each of the texts, the following outputs are generated to address the questions (1)-(3) outlined above: 

1. Metrics: coherence and perplexity scores
2. Keywords: visualisation of weight and count of the keywords for each topic
3. Dominant topics: the dominant topic for each document and its percentage of contribution is appended to the original data frame.
4. Representatives: for each topic, the names of three talks with the highest percentage of contribution are printed, as representatives for a given topic
5. Topics over time: using the topic distributions for each document and the published date of the documents, the development of topics is plotted with a rolling mean over 60 days.


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
    |-- LDA_tedtalks.py                 # Script for LDA topic modelling of Ted Talks

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
The raw data, which was originally downloaded from [Kaggle](https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset) is stored in the `data/` directory of this repository. Thus, when the repository is cloned, no additional data needs to be retrieved. 

### 3. Script for LDA Topic Modelling on Ted Talk Transcripts
The script `LDA_tedtalks.py` preprocesses the Ted Talk transcripts, following the steps below. Subsequently, it will either compute coherence values for a set of different number of topics **OR** it will conduct an LDA topic analysis on a given number of topics, and produce the output, described above. The script should be called from the `src/` directory:

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
   Origial .csv dataframe, with appended columns of `dominant_topic` and `topic_perc_contib`
   
- `LDA_representatives.txt`\
   For each topic, the three talks with the highest contribution are printed. 
   
- `LDA_topics_over_time.png`\
   Visualisation of development of topics over time. 

  
## Results and Discussion 
Output of the scripts can be found in the corresponding directory in `out/`. The model had a perplexity score of -8.69 and a coherence measure of 0.46. Below the results are illustrated and discussed in relation to the three questions posed above: 

__1. Is it possible to find distinct topics across the transcript of Ted Talks?__

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

In general it seems like for some e.g. Psychology & Neuroscience (4) or Health Care (8) it was quite easy to see how the keywords fit into a broader category. For others, e.g. Economy (9) or Human Body & Animals (3) it was more difficult, since words seem to come from different topics.

__2. Are there any temoral developments over the years the topics that the talks are adressing?__

Below, the temporal development of topics from 2016-2020 are displayed:

![](https://github.com/nicole-dwenger/cdslanguage-LDA/blob/master/out/LDA_15_topics/LDA_topics_over_time.png)

Looking at this development, there seems a few topics which are fairly dominant across time, which are topic Family (10), Climate & Environment (1) and Technology (2). Intuitively, this makes sense to me since these topics have been quite dominant the past few years. The reason why e.g. Family is quite dominant might also simply be because the words of this topic seem to be quite broad and might thus occur in many different talks. For the topic of Planet Eart and Space (13) there seems to be an increase around 2019, which could be due to an increased interest in exploring Mars or other other planets. Lastly, the topic of Love & Music (14) seems to not be as dominant, which might be related to the fact that the words were very specific (e.g. melody), and might thus not occur in many talks.

__3. Is is possible to generate "recommendations" of talks for each of the topics?__ 

For the full list of the 3 talks which had the highest percentage of contribution for each topic, see the [file](https://github.com/nicole-dwenger/cdslanguage-LDA/blob/master/out/LDA_15_topics/LDA_representatives.txt) in the `out/` directory, only a few examples will be discussed here. 

- For the topic Psychology & Neuroscience (4) the following titles were extracted: 'How stress affects your brain', 'How memories form and how we lose them', 'What happens when you have a concussion?'. Without knowing anything about the talks, these tiles all seem to relate to the given topic name. 

- For the topic Earth & Space (13) the following titles were extracted: 'A needle in countless haystacks: Finding habitable worlds', '3 moons and a planet that could have alien life', 'Could the Earth be swallowed by a black hole?'. These also seem to fit into a general theme of exploring space and finding life on other planets. 

- For the topic Family (10) the following titles were extracted: 'Why we should take laughter more seriously', 'Love others to love yourself', "I grew up in the Westboro Baptist Church. Here's why I left". It seems that based on these titles, the topic should have rather been termed 'Lifestyle' or 'Life Advice', since all of these titles seem to refer to something about the way of living or interaction with others. 

## Contact
If you have any questions, feel free to contact me at 201805351@post.au.dk.

