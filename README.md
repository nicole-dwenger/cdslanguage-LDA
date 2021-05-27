# LDA Topic Modelling: What Are They TedTalking about?

[Description](#description) | [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion) | [Contact](#contact)

## Description
> This project relates to Assignment 5: (Un)Supervised Machine Learning

TedTalks have become popular as videos in which experts share their knowledge and ideas. What if, in 100 years someone finds all these videos, but has no idea what they are about. It would be tedious to watch all of them, to find out. Thus, it would be great if there was a way to figure out which topics these talks cover based on their contents. Consequently, it would also be possible to find talks which cover a specific topic of interest. Lastly, knowing what different talks were about and when they were published could also help to investigate how topics in these talks developed over time, as an interesting indicator from a cultural perspective. This project aimed to investigate topics in transcripts of TedTalks using unsupervised machine learning, specifically LDA topic modelling.  Specifically, the following questions were of interest:
 
1. Is it possible to find topics across the transcripts of TedTalks? 
2. Is it possible to find *representative talks* for each of the topics? 
3. Are there any temporal developments in the topics of TedTalks?


## Methods

### Data and Preprocessing
The data used for this project, was extracted from [Kaggle](https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset), and contains information and transcripts of 4005 English TedTalks from 2006 to 2020. To reduce processing only talks from 2016-2020 were used, leading to a total of 2167 transcripts of talks. Previous to the LDA topic modelling, these transcripts were preprocessed using the following steps
 
1. All non-content aspects of the transcripts, e.g. (Laughter) were removed by removing everything that was in () or [].
2. All texts were tokenised and the tokens were lowered using the `simple_preprocess` function from gensim.
3. Stop words were removed using the english stop-word list from NLTK. 
4. Bigrams and trigrams were extracted using a minimum count of 3 and a threshold of 100. This was done by extracting tokens which frequently occur together and thus can be expected to have a meaning as a combined entity, rather than on their own. For instance, if the tokens *new* and *york* occur together they should be seen as *new york*. 
5. To reduce noise for topic modelling, only nouns (as determined by spaCy’s POS tag of the language model `en_core_web_sm`) were extracted, as they were expected to contribute most to the meaning or content of a talk. 
6. All tokens were lemmatised.
7. Lastly, after running the LDA model multiple times, there were some words which occurred across several topics. Thus, a list was constructed with words which seemed too occur frequently across talks, but tell relatively little about the specific topic of a talk. This list contained the following tokens: *people, world, talk, time, other, hundred, one, life, thousand, number, way, year, thing, story, day, lot, question, idea, word*.

### LDA Topic Modelling
LDA (Latent Direct Allocation) topic modelling is an unsupervised machine learning algorithm, which can help to extract topics from unlabelled text documents. LDA builds on the assumption that words in documents are derived from different clusters of topics. Put differently, words cluster into different topics, and documents are made up of these words. Thus, the distribution of words (and which cluster they belong to) can help to determine the distribution of topics for each document. LDA aims to reverse engineer the topics (clusters) from which the words were drawn. In this project, the transcripts of talks were considered to be the documents in a large corpus of documents (transcripts). After preprocessing the transcripts as described above, the dictionary and corpus of tokens across all documents were extracted. The dictionary contains the mapping of tokens to integers, while the corpus contains these inter-id’s and how often they occur. These were then used to train an LDA Multicore model. The model was once trained to extract 15 topics and once for 20 topics. Based on a higher coherence score for 15 topics (0.46) compared to 20 topics (0.45), the focus in the following will be on the model trained for 15 topics. However, the output is provided for both models in the out/ directory of the GitHub repository. Optimally, an exploration of more coherence values for a different number of topics should have been conducted, but was omitted for time and processing reasons. From the trained model, the following outputs were extracted and generated to answer the questions (1)-(3) outlined above: 

1. Metrics: The perplexity and coherence scores of the LDA model are stored. The focus was on the coherence score, which is a measure of how similar words in a topic are. 
2. Keywords: The top 10 keywords for each topic are extracted and their counts and weights were visualised in bar graphs. 
3. Dominant topics:  The dominant topic and its percental contributution to the given document were appended to the original data frame. The dominant topic was defined as the one which had the highest percentage of contribution to a given document. 
4. Representatives: For each topic, the names of the three talks which had the highest percentage of contribution are saved as representatives for a given topic. In other words, these should represent talks, where most words are coming from the given topic.
5. Topics over time: Using the topic distributions for each document and their published date, the development of topics was plotted with a 90 day rolling average. 


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
To run the scripts in this repository, I recommend cloning this repository and installing necessary dependencies in a virtual environment. The bash script `create_venv.sh` can be used to create a virtual environment called `venv_LDA` with all necessary dependencies, listed in the `requirements.txt` file. This will also load the required language model (`en_core_web_sm`, from spaCy). The following commands can be used:

```bash
# cloning the repository
git clone https://github.com/nicole-dwenger/cdslanguage-LDA.gi`

# move into directory
cd cdslanguage-LDA/

# install virtual environment
bash create_venv.sh

# activate virtual environment 
source venv_LDA/bin/activate
```

### 2. Data 
The raw data, which was originally downloaded from [Kaggle](https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset) is stored in the `data/` directory of this repository.  Thus, after cloning the repository, it is not necessary to retrieve any additional data.

### 3. Script for LDA Topic Modelling on TedTalk Transcripts: LDA_tedtalks.py
The script `LDA_tedtalks.py` preprocesses the TedTalk transcripts, following the steps above. Subsequently, it will conduct an LDA topic analysis on a given number of topics, and produce the outputs, described below. The script should be called from the `src/` directory:

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
   All talks which were published later than the given year. For `2015` this would be all talks from 2016-2020.  
   
__Output saved in__ saved `out/LDA_{n_topics}_topics`:

- `LDA_metrics.txt`\
   File containing coherence and perplexity scores. 
   
- `LDA_keywords.png`\
   Visualisation of count and weight of keywords for each topic. 
   
- `LDA_dominant_topics.csv`\
   Origial .csv dataframe, with appended columns of dominant_topic and topic_perc_contib.
   
- `LDA_representatives.txt`\
   For each topic, the three talks with the highest contribution are saved. 
   
- `LDA_topics_over_time.png`\
   Visualisation of development of topics over time. 

  
## Results and Discussion 
Output of the scripts can be found in the corresponding directory in `out/`. The model trained for 15 topics had a perplexity score of -8.69 and a coherence measure of 0.46. Below the results are illustrated and discussed in relation to the three initial questions posed above. 

### 1. Is it possible to find distinct topics across the transcript of TedTalks?

To get an idea of which topics were extracted from the transcripts, the keywords can be inspected to see if they seem to relate to a topic (based on human judgement). Below the keywords of the 15 derived topics are displayed:

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

Overall, it seems that for some clusters e.g. *Psychology & Neuroscience* (4) or *Health Care* (8) it was quite easy to see how the keywords fit into a category. For others, e.g. *Human Body & Animals* (3) it was more difficult, since words seemed quite different (animal, robot, egg).

### 2. Is it possible to extract representative talks for each of the topics?

The full list of representative talks for each topic can be seen in this [file](https://github.com/nicole-dwenger/cdslanguage-LDA/blob/master/out/LDA_15_topics/LDA_representatives.txt), stored in the `out/` directory. Only a few examples will be discussed in the following. 

- For Topic (4) *Psychology & Neuroscience*, the following titles were extracted: 'How stress affects your brain', 'How memories form and how we lose them', 'What happens when you have a concussion?'. Without knowing anything about the talks, these titles all seem to relate to the given topic name.
- For Topic (13) *Planet Earth & Space* the following titles were extracted: 'A needle in countless haystacks: Finding habitable worlds', '3 moons and a planet that could have alien life', 'Could the Earth be swallowed by a black hole?'. Again, based on their titles, these talks all seem to fit into a theme of exploring space and finding life on another planet. 
- For Topic (10) *Family* the following titles were extracted: 'Why we should take laughter more seriously', 'Love others to love yourself', "I grew up in the Westboro Baptist Church. Here's why I left". Here, it seems that the given title of Family does not really fit, but rather these titles fit into a category of Life Advice or Lifestyle. However, the keywords of child, family, friend, love, conversation seem to fit to the talks, as they relate to the interaction with others. 

### 3. Are there any temporal developments in the topics of TedTalks?

Below, the temporal development of topics from 2016-2020 are displayed (Figure 3.2). Looking at this development, there seem to be a few topics which are fairly dominant across time, such as Topic (10) *Family*, Topic (1) *Climate & Environment* and Topic (2) *Technology*. For me, this makes intuitively sense, as these topics have been quite dominant in the past years. Further, it may also be that many talks relate to these topics in some way. For Topic (13) *Planet Earth & Space* there seems to be an increase around 2019, which could be due to an increased interest in exploring Mars or other planets.

![](https://github.com/nicole-dwenger/cdslanguage-LDA/blob/master/out/LDA_15_topics/LDA_topics_over_time.png)


## Contact
If you have any questions, feel free to contact me at 201805351@post.au.dk.

