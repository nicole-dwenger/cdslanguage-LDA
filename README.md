# LDA Topic Modelling: Ted Talks

[Description](#description) | [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion) | [Contact](#contact)

## Description
> This project relates to Assignment 5: (Un)Supervised Machine Learning

TedTalks are videos of talks of experts from different fields. Imagine in 20 years someone finds all these videos, but has no way of knowing what they are about, from only looking at the title. Instead of looking through all the videos and assigning labels, it would be great if this could be solved in a more efficient way. This way, one could also look at the temoral development of when different topics were more or less prevelant and one could find talks which focus on a specific topic faster. 

LDA topic modelling is an unsupervides machine learning algorithm, which can help to extract topics from unlabelled text documents. LDA builds on the assumption that words come from different clusters (i.e topics), and that every document thus adresses these different topics in different proportions, based on the words that it contains. The aim of LDA is then to extract the topics (clusters) from which the words are drawn, using the distribution of words in the documents. In this project, each of the talks is considered to be a document in a large corpus of documents (corpus of talks), where each document is comprised of multiple topics. 

The aim of this project is thus to adress the following questions: 
1. Is it possible to find distinct topics across the transcript of TedTalks (documents)? 
2. Are there any temoral developments over the years the topics that the talks are adressing? 
3. Is is possible to generate "recommendations" of talks for each of the topics? 


## Methods

### Data and Preprocessing
The data, which was used for this project is a dataset from [Kaggle](https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset), which contains information and transcripts of 4005 English Ted Talks from 2016 to 2020. For this project, I only used talks from 2016-2020, to reduce processing time, which led to a total of 2167 talks. 
 
The transcripts were preprocessed using the following steps: 
1. Remove all "non-content" aspects of the transcripts, e.g. *(Laugther)*, by removing everything that is in "()" or "[]".
2. Lowercase and tokenize all texts using gensim's `simple_process` function.
3. Remove stopwords using the english stopword list from nltk.
4. Create bigrams and trigrams, using a minimum count of 3 and a threshold of 100. 
5. Extract only tokens which have the POS tag `NOUN`, to only focus on the *meaniningful* words, and lemmatise these tokens. 
6. Remove tokens using a self-defined list of tokens (based on investigating the data and seeing which words occur often, but contribute relalively little to inferring something about a "topic". This list contained the following tokens: "people", "world", "talk", "time", "other", "hundred", "one", "life", "thousand", "number", "way", "year", "thing", "story", "day", "lot", "question", "idea", "word".

### LDA Topic Modelling
For topic modelling, an LDA Multicore model was trained on the derived dictionary and corpus of tokens. The model was run for 15 topics and for 20 topics. Based on a higher coherence score for 15 topics (0.45) compared to 20 topics (0.43), the focus in the following will be on the model for 15 topics. However, output is provided for both models in `out/`. Optimally, an exploration of more coherence values for different number of topics should have been conducted, but was omitted due to time and processing reasons. Using the derived topics and distributions of topics for each of the texts, the following outputs are generated for questions (1)-(3) outlined above: 

1. Metrics: coherece and perplexity scores
2. Keywords: the weight and total occurance of the top 10 keywords for each topic are visualisated
2. Dominant topics: to the original dataframe, the dominant topic for each document and it's percentual contibution is appended
3. Representatives: for each topic, the names of 5 talks with the highest percentage of contibution are printed, as "recommendations" for a given topic
4. Topics over time: using the topic distributions for each document and the publish date of the documents, the development of topics is plotted with a rolling mean of 200


## Repository Structure 
```
|-- data/
    |-- ted_talks_en.csv                # Raw data of ted talks
    
|-- out/                                # Directory for output, corresponding to scripts
    |-- LDA_metrics.txt                 # Coherence and Perplexity scores
    |-- LDA_keywords.png                # Visualisation of weight and count of keywords for each topic
    |-- LDA_dominant_topics.csv         # Original .csv with appended columns of dominant_topic and topic_perc_contrib
    |-- LDA_representatives.txt         # 3 talks with highest topic_perc_contrib for each topic
    |-- LDA_topics_over_time.png        # Visualisation of topic development over time

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
   
- `-n, --n_topics`: *int, optional, default:* `20`\
   Number of topics to extract.
   
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
Output of the scripts can be found in the corresponding directory in `out/`. The model had a perplexity score of -8.6 and a coherence measure of 0.45. Below the results are illustrated and discussed in relation to the three questions posed above: 

__1. Is it possible to find distinct topics across the transcript of Ted Talks?__

To answer this question, one can inspect the keywords each of the topics, to see if they relate to the same topic and thus compose a distinct topic. Below are displayed the keywords of the 15 derived topics: 

![](https://github.com/nicole-dwenger/cdslanguage-LDA/blob/master/out/LDA_15_topics/LDA_keywords.png)

Based on these keywords, I could imagine the following fitting topic names: 

- Topic 0: Education
- Topic 1: Climate
- Topic 2: Technology 
- Topic 3: Human Body & Animals
- Topic 4: Psychology & Neuroscience
- Topic 5: Medicine
- Topic 6: Gender
- Topic 7: Arts & Literature
- Topic 8: Health
- Topic 9: Economy
- Topic 10: Family
- Topic 11: Nutrition & Environment
- Topic 12: Society & Justice
- Topic 13: Planet Earth & Space
- Topic 14: Love 

In general it seems like for some e.g. Psychology & Neuroscience (4) or Health Care (8) it was quite easy to see how the keywords fit into a broader category. For others, e.g. Economy (9) or Human Body & Animals (3) it was more difficult, since words seem to come from different topics.

__2. Are there any temoral developments over the years the topics that the talks are adressing?__

Below, the temporal development of topics from 2016-2020 are displayed, see `file` for larger size of image: 

![](https://github.com/nicole-dwenger/cdslanguage-LDA/blob/master/out/LDA_15_topics/LDA_topics_over_time.png)

Looking at this development, there seems a few topics which are fairly dominant across time, which are topic Family (10), Technology (2) and Climate (1). The reason for this might be, that the words of these topic may not be a distinct and specific category, but that many talks make use of these more general words (e.g. women, children, today, change, work). For the topic of Planet Eart and Space (13) there seems to be quite an increase around 2019, which might simply be due to an increased awareness of the ennvironment. Lastly, the topic of Love (12) seems to not be as dominant, which might also be related to the fact that the words seemed to come from a range of topics, and it was not very clear how they fit together.

__3. Is is possible to generate "recommendations" of talks for each of the topics?__ 

For the full list of the 3 talks which had the highest percentage of contribution for each topic, see the [file](https://github.com/nicole-dwenger/cdslanguage-LDA/blob/master/out/LDA_15_topics/LDA_representatives.txt) in the `out/` directory, only a few examples will be discussed here. 

For Planet Earth & Space (13) the following titles were extracted: '3 moons and a planet that could have alien life', 'Is space trying to kill us?', 'A needle in countless haystacks: Finding habitable worlds'. Based on only the titles, these talks seem to fit quite nicely into the broad topic of Planet Earth and Space. 

For Medicine (8) the following titles were extracted: 'What are stem cells?', 'The power of the placebo effect', 'The dangers of mixing drugs'. Similarly, based on the title these also seem to fit quite well and all realte to the human body and treatments. 

For Family (10), which seemed to be the dominant topic in the temoral development, displayed above, the following titles were exctracted: 'Why we should take laughter more seriously', 'Love others to love yourself', 'Three ideas. Three contradictions. Or not.'. Looking at these titles, it seems like they rather all relate to something like lifesyle or life advice. They seem to cover more broad topics, rather than specific ones, such as Medicine, which might be the reason why the topic was dominant across time. 

## Contact
If you have any questions, feel free to contact me at 201805351@post.au.dk.

