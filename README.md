***
# OK?
 
Group members: Alexander Weyhe, Kascha Kruschwitz, Ludmila Bajuk

## Introduction (Background)

<div align="justify">
In spoken language, the word `ok` is uttered frequently and can appear in different contexts. Multiple studies have investigated in which ways the word is used in English and other languages. According to Burchifield (1982 in Adegbija & Bello, 2019), the word `okay` can appear as a response or phrase to show acceptance or agreement. Additionally, it can be classified as an adjective to mean “correct”, good”, or “satisfactory. Condon (1986 in Lee, 2008) introduces `okey` as a marker of transitions in discourse. Finally, in another study by Adegbija & Bello (2001), the usage of `O.K.` was investigated in Nigerian English and compared to (“Standard”) English. In Nigerian English, it can be used as a gap filler, surprise, or discourse termination.
</div>

<div align="justify">
Here, the question comes to mind if one uses the word in the same context in written language. Consequently, the focus of this project is to investigate the different usage of “ok” in written language, in particular in online forums. The peculiarity in written language arises from the different spellings available for the word – as can be seen by the different spellings in the preceding sentences. Thus, this project investigates whether various spelling forms are used differently in written language and answers the research question: In how far do variations of the word `ok` differ in their usage in online forums? So, from the project we expect to see differences in the usage of the word "ok" and its different spellings across subreddits.
</div>



### Research Questions
  
1. ***Are the different spellings associated with different uses, meanings etc...?***
2. ***Are there differences across topics/user types?***


## Dataset

<div align="justify">
For this project, we used the “webis/tldr-17” dataset (Völske et al., 2017) from Hugging Face which can be found here: https://huggingface.co/datasets/webis/tldr-17. A special feature of this dataset is that it only contains Reddit posts with the abbreviation “tl;dr”. This stands for “too long; didn’t read” and has become a popular tool for authors from Reddit posts to give a short summary of their posts’ contents. The dataset consists of 3,848,330 posts across a total of 29651 subreddits. Each data point contains the following data fields: author, body, normalizedBody, subreddit, subreddit_id, id, content summary. We used the "content" part only for the subreddits (not including the tldr) to train the model. For our work, the different subreddits are of particular interest because we want to compare our research questions across them. Additionally, since we used posts from the four biggest subreddits, it was necessary that for each post, it's subreddit is specified. 
</div>



## Methods

<div align="justify">
Our project was coded on Google Colab in order to work collaboratively. Therefore, relevant sections need to be adapted to run the code, mainly loading the files, for example. Even though we have included a requirements.txt file, we pip-installed all the requirements in the Colab Notebook, so they would have to be removed from the top of the two documents. We uploaded one model into the code folder for demonstration purposes.
 
</div>

<div align="justify">
The code consist of two documents: 
</div>

1. One for extracting and preprocessing the data, and running the skipgram model. Such model is saved after the last step, so that it can be used locally as input to the second file.

2. The second file consist of the analysis of the data using fours comparison metrics and plotted the results obtained. 
<br>
### Prepocessing

<div align="justify">
From the dataset we extracted the four largest subreddits from the dataset and put it in ad pandas dataframe to facilitate its access during the analysis. Also, we saved it if as a csv file in order to prevent having to load it every time when we want to access it. We used the following four datasets for our project: AskReddit (because of its size, we only used the first [:200000000] characters), League of Legends, Relationships, and TIFU. We then defined a custom tokenizer with the NLTK library and created regular expressions to capture exceptions, i.e. to remove from punctuation from within a word. The tokenizer removes all the punctuation for every tokens, except for "O.K", and also lowers all tokens, expecpt for "Ok" and "O.K". 
 
</div>

```bash
def custom_tokenizer(raw_list):
  # split into sentences and words: list of lists for word2vec
  tokenized = sent_tokenize(raw_list)
  tokenized = [word_tokenize(i) for i in tokenized]
  # remove all tokens that are just punctuation
  tokenized = [[i for i in sent if i not in string.punctuation] for sent in tokenized ]
  # for all tokens that are not a version of OK, lowercase and remove puntuation
  tokenized = [[i if i in initial_ok_list else i.lower() for i in sent] for sent in tokenized ]
  tokenized = [[i if i in initial_ok_list else re.sub(r'[^\w\s]', '', i)  for i in sent] for sent in tokenized ]
  return tokenized
```

<div align="justify">
 
We got all the OKs in each subreddits and got a list of all the oks that appeared in every single subreddit (more than 5 times to generate mostly good embeddings).

</div>

                                         'okay', 'ok', 'Ok', 'O.K', 'okey', 'k'

### Model Setup 

<div align="justify">
We built a skipgram Word2Vec model using Gensim, trained a model for each subreddit and save it to be easily loaded whenever we needed it. The model was trained on the entire content of each subreddit for words appearing more than 5 times. Furthermore, we specified for the model a vector size of 100, because of our small traning sample size, as well as the skip-gram function, do that we could generate similar words and make semantic comparisons.

</div>
      

```bash
def skipgram_w2v_model(tokenized_content, model_name):
  skipgram_model = Word2Vec(vector_size=100, # size of 2D vector
 			    window=5, # distance between the current and predicted word in each direction
 			    sg=1,  # Skip-Gram model
 			    min_count=5)  # Ignores all words with a total frequency lower than five 			
  skipgram_model.build_vocab(tokenized_content) # store all words in a vocabulary
  skipgram_model.train(tokenized_content, total_examples=skipgram_model.corpus_count, epochs=10)
```




## Comparison Metrics and Results

We analysed the different version of OK with several comparison metrics.

### Nearest Neighbors

<div align="justify">
In a first exploratory step, we generated the Nearest Neighbors of each word for each subreddit, the first five of each given in the table below. This revealed that while `okay`, `ok` and `Ok` had generated embeddings that would be expected from versions of Ok, the other three spellings did not generate the most reliable vectors. This can be seen by the most common words being rather unique, likely due to the relatively small sample size. However, we still took them into account in the successive analysis, to see if we could discover some pattern. 
 
</div>

| Subreddit  | Words in Common from 10 most similar      |
|----------------|-----------------------------------------|
| **Askreddit**  | okay: ['ok', 'alright', 'fine', 'allright', 'mhm']  |
|              | ok: ['okay', 'alright', 'fine', 'allright', 'mhm'] |
|              | Ok: ['yeah', 'uh', 'yea', 'mhm', 'ummmm'] |
|              | O.K: ['poopie', 'yyes', 'mhmm', 'oughta', 'ummmmmm'] |
|              | okey: ['thehealeroftri', 'dany', 'batcave', 'yeaaa', 'spotlighted'] |
|              | k: ['j', 'c', 'g', 'b', 'p'] |  
|----------------|-----------------------------------------|
| **League of Legends**  | okay: ['ok', 'alright', 'yeah', 'fine', 'Ok']  |
|                | ok:  ['okay', 'alright', 'yeah', 'fine', 'yea']  | 
|                | Ok: ['yeah', 'hey', 'yea', 'okay', 'welp']  | 
|                | O.K:   ['autoloss', 'dammed', 'cartman', 'neato', 'apealing']  | 
|                |okey:   ['yeah', 'idek', 'yeh', 'ahh', 'yea']     | 
|                | k:  ['t1', 'sgw', 'lgim', 'telecom', 'starhorn']      | 
|----------------|-----------------------------------------|
| **TIFU**       | okay: ['ok', 'alright', 'fine', 'overreacting', 'yeah']  |
|                | ok:  ['okay', 'alright', 'fine', 'yeah', 'O.K']    | 
|                | Ok:   ['yeah', 'eh', 'okay', 'yea', 'yrrah321'] | 
|                | O.K: ['poogadi', 'bluffing', 'excommunicated', 'rue', 'honoured'] | 
|                |okey: ['nooooope', 'fuuuuuuuuuuuuuck', 'neato', 'uhhuh', 'fuckfuckfuckfuckfuck']  | 
|                | k:  ['h', 'j', 'kelly', 'linette', 'l'] | 
|----------------|-----------------------------------------|
| **Relationships**  | okay:['ok', 'alright', 'fine', 'okey', 'allright']|
|                | ok:['okay', 'fine', 'alright', 'allright', 'okey'] | 
|                | Ok:['yeah', 'thanksokay', 'hey', 'rr_a', 'alright'] | 
|                | O.K:['concur', 'mutuals', 'tabled', 'sidekick', 'bullcrap']                   | 
|                |okey: ['ahah', 'rehearsed', 'ehhhh', 'wha', 'noncommital']                 | 
|                | k: ['john', 'j', 'jane', 'h', 'sarah']               | 



### Word Senses 

<div align="justify">
The idea was to find out the different word senses for the different spellings of OK and then see whether the subreddits yield different results. In a first step, we used the nltk library and the built-in wordnet function to see all possible definitions for the different versions of “ok”. This gave us the following results: 
</div>


WordNet definitions for "ok":

definition 0: a state in south central United States

definition 1: an endorsement

definition 2: being satisfactory or in satisfactory condition

definition 3: an expression of agreement normally occurring at the beginning of a sentence

***

WordNet definitions for "okay":

definition 0: an endorsement

definition 1: give sanction to

definition 2: being satisfactory or in satisfactory condition

definition 3: in a satisfactory or adequate manner; ; ; (`alright` is a nonstandard variant of `all right`)

***

WordNet definitions for "Ok":

definition 0: a state in south central United States

definition 1: an endorsement

definition 2: being satisfactory or in satisfactory condition

definition 3: an expression of agreement normally occurring at the beginning of a sentence

***

WordNet definitions for "O.K.":

definition 0: an endorsement

definition 1: give sanction to

definition 2: being satisfactory or in satisfactory condition

definition 3: in a satisfactory or adequate manner; ; ; (`alright` is a nonstandard variant of `all right`)

***

WordNet definitions for "k":

definition 0: the basic unit of thermodynamic temperature adopted under the Systeme International d'Unites

definition 1: a light soft silver-white metallic element of the alkali metal group; oxidizes rapidly in air and reacts violently with water; is abundant in nature in combined forms occurring in sea water and in carnallite and kainite and sylvite

definition 2: the cardinal number that is the product of 10 and 100

definition 3: a unit of information equal to 1000 bytes

definition 4: a unit of information equal to 1024 bytes

definition 5: the 11th letter of the Roman alphabet

definition 6: street names for ketamine

definition 7: denoting a quantity consisting of 1,000 items or units

***

WordNet definitions for "okey":

definition 0: an endorsement

***

<div align="justify">
 
In the next step, we employed the Lesk algorithm to find out the definition for each variation of OK based on its most similar words across subreddits. To briefly lay out what the Lesk algorithm does, it takes an input word and a sentence in which the given word appears in and then finds the overlapping words from the sentence in each gloss of the target word. The gloss with the most matches will be output as the definition. Unfortunately, in our case it did not yield any results and the definition for each variation of “OK” was the same in each subreddit. We then tried to see how many of the most similar words for each `OK` overlapped with the glosses in all subreddits. We found out that only for the word `okay` there was one match each for all subreddits. All other variations of “ok” across the subreddits had zero matches.

</div>

### Cosine Similarity 

<div align="justify">
The cosine similarity analisys was conducted in order to, firstly, observe the similarity or dissimilarity of the different versions of OK in eachsubreddit. Therefore, within each subreddit we calculated the cosine similarity for each two Oks using Word2Vec's inbuilt metric. For example, `ok` and `O.K` similarity scores differ significantly between the TIFU and relationship subreddits. 
 
</div>

| tifu               | ok - O.K : 0.7133671   |
| askreddit          | ok - O.K : 0.7002533   |
| league of legends  | ok - O.K : 0.562705    | 
| relationships      | ok - O.K : 0.3847269   |


<div align="justify">
After obtaining the scores, we plotted the similarity using  "t-distributed Stochastic Neighbor Embedding" to visualize the distribution of each OK. The results exhibit that 'ok' and 'okay' have the most simmilar word embeddings within all subreddits. More so, we can conclude, as visualized in the graphs that Askreddit and TIFU have the most similar OK distribution. 
 
</div>

| ![Graph of Cosine Similarity of OKs in Askreddit](/figures/ok_askred.png)  | ![Graph of Cosine Similarity of OKs in League of Legends](/figures/ok_lol.png) |
| ![Graph of Cosine Similarity of OKs in Relationships](/figures/ok_relations.png)  | ![Graph of Cosine Similarity of OKs in TIFU](/figures/ok_tifu.png) |



### Overlapping Nearest Neighbors

<div align="justify">
A pairwise semantic similarity analysis was carried out to compare the overlapping most similar words of each OK and their corresponding similarity scores. For this analysis, we extracted the similar words of all OKs in two subreddits and compares only the overlapping words. 
 
</div>

| Version of OK  | Words in Common from 10 most similar      |
|----------------|-----------------------------------------|
| 'okay'         | ['ok', 'alright', 'fine', 'yeah', 'yea']  |                                                       
| 'ok'           | ['okay', 'fine', 'alright']               |
| 'Ok'           |['yeah', 'hey']                            |
| 'O.K'          | []                                        |
| 'okey'         | []                                        | 
| 'k'            | []                                        |


<div align="justify">
We paired all the subreddits with each other obtaining 6 different pairs. In this case, the only OKs that had overlapping similar words were `ok`, `okay`, and `Ok`. Although `k` only had similar overlapping words or letter with in a few pairs, they were not reliable since we had an only one letter embedding. 
 
</div>

<div align="justify">
For each pair we got the mean semantic similarity for each similar and overlapping word for each 'OK', so that we got now one score for each word in the list of overlaping words. Once every word had only one score, we averaged those scores (in the case of pair1, for example: 
 
</div>

      pair1 = get_most_similar_words(tifu_sim_ok, askreddit_sim_ok)
      {'okay': ['ok', 'alright', 'fine', 'allright', 'yeah', 'O.K', 'yea']}

 <div align="justify">
we get the mean value of the scores we got from the first 6 words found in the `okay` overlapping words list). After getting the mean score for each 'OK' in each subreddit, we calculated the final mean score of each OK across subreddits obtaining, in that way, the semantic similarity of the same 'OK' in different contexts.
  
  </div>

<div align="justify">
The results of the pairwise comparison grouped the OKs into the ones that actually had overlapping similar words, and the ones that did not. On the one hand, the variations `ok`, `okay` and `Ok` had an approximate of 80% of semantic similarity accross subreddits. More so, for the word `k` a percentage of around 80 is shown, however, these results are not accurate since the similar common words for `k` were mainly aisolated letters or proper people's names. This may account for the fact that 'k' is also an sigle letter, and therefore, quiet unrelieable. The matching to the proper names could be the consequence of simplifying names into nicknames of only one letter, for instances, `Kim` = `k`. On the other hand, the words `O.K` and `okey` presented no comparable overlapping word neighbours, resulting into an dissimilar semantic relation.
 
</div>

![Word Overlapping Semantic Similarity](https://github.com/kakrusch/docana-project-luka/assets/162272922/5a572432-a872-4175-96e5-f51b15b836ae)


### Sentiment analysis 

<div align="justify">
Finally, sentiment analysis was conducted based on the 50 nearest neighbors for each OK in each subreddit, using a lexicon provided by the VADER sentiment analyzer (Hutto & Gilbert, 2014). This lexicon included human-rated sentiments for many words specific to social media discourse and was thus deemed well-suited for the task of calculating the sentiments on Reddit. 50 nearest neighbors were used to match at least one similar word to a word in the provided lexicon but not generate words that are too dissimilar. This resulted in all words having at least one match, except for `k` in the League of Legends subreddit. The average number of matched words was 4.76, with disproportionately many matches for `okay` and `ok`. We then plotted the results.
 
</div>

<div align="justify">
First, the sentiment across subreddits was compared. `okay` and `ok` most consistently reflect a very positive sentiment, whereas all other words either are on the border too or are themselves neutral. Further, the version `O.K` and the uses of ok in the League of Legends subreddit seemed to express a more neutral or negative sentiment.
 
</div>

| ![Graph of the Sentiments of okays across subreddits](/figures/Sentiment of different okays across subreddits.png)  | ![Graph of the Sentiments of all subreddits expressed by ok](/figures/sent-all-okays.png) |

<div align="justify">
Within subreddits, the patterns observed above are confirmed. TIFU expresses the most positive sentiment because all versions of OK express a positive sentiment. In Askreddit, Relationships, and TIFU, `okay`, `Ok`, `ok`, and `k` all express positive sentiments. Of these, only `okay` is positive in League of Legends, which otherwise displays a markedly different pattern than the other subreddits, with `okay` being neutral and `Ok` very negative. `okey` and `O.K` display similar patterns, in that their sentiment is different in one out of the four subreddits. `okey` is overall positive, except in AskReddit, where it is the most negative version. `O.K` shows the opposite pattern, being consistently negative except in TIFU, where, like all other spellings, it has a positive sentiment.
 
</div>

| ![Graph of the Sentiments of okays in AskReddit](/figures/Sentiment of different okays in AskReddit.png)  | ![Graph of the Sentiments of okays in League of Legends](/figures/Sentiment of different okays in League of Legends.png) |
| ![Graph of the Sentiments of okays in Relationships](/figures/Sentiment of different okays in Relationships.png)  | ![Graph of the Sentiments of okays in TIFU](/figures/Sentiment of different okays in TIFU.png) |


## Discussion and Conclusion

- Is the meaning of OK and its variations always the same?
- Are the different spellings associated with different uses, meanings etc...?
- Are there differences across topics/user types?

<div align="justify">
Analyzing the results provided by every metric employed, we can deduce that the standard spellings `okay`, `ok` and `Ok` seem to be mostly associated with the standard uses idetntified for spoken langugage, such as adjectival or response particle. Apart from being consistent across subreddits, the nearest neighbors of these three versions of OK displayed a high percentage of similar words, these mainly being words like `alright`, `fine`, and `yea`. One comparable feature from these words is, that they denote a positive sentiment associated with meanings like 'good' in the majority of the subreddits, except for the League of Legends subreddit. It thus seems any variation of Ok have a negative connotation in the gaming area. 
 
</div>

<div align="justify">
Regarding the more unique versions of OK, such as `okey`, `O.K` and `k` the realiability of the embeddings varies signficantly, especially for `k`. Thus, our analzsis here must be taken with a grain of salt. The single character version `k` often fails to convey a consistent meaning, due to its brevity and potential overlap with other uses of the letter `k`, such as in nicknames or as abbreviations. Therefore, `k` leads to ambiguous interpretations which makes sny semantic analysis not dependable. The terms `okey` and `O.K` cluster somewhat, in that their nearest neighbors seem to relate to words of uncertainty or swear words. The term `okey` shows variability in the cosine similarity analysis and has a low cross-subredits similarity, which indicates that it has several diverse uses. Mainly in the AskReddit subreddit, `okey` is associated with a negative connotation while in other subreddit it denotes a positive one. Finally, `O.K` consistenly shows a mostly negative sentiment accross the majority of the subreddits while having different nearest neighbors in the different subreddits. This means that, even though the usage in the TIFU and Askreddit are slightly similar, they differ. However, the sentiment is usually not a positive one in any of the cases. 
 
</div>

<div align="justify">
In terms of differences across subreddits, the cosine distribution between TIFU and AskReddit seems the most similar, which is expected considering they both have the most broad topic. Further, League of Legends is the most different in almost all metrics, which again is expected since it is specific to gaming, whereas the other subreddits deal with more banal real-life topics. It thus does seem to be the case that the uses of different spellings of OK differs across topics, i.e. across subreddits.
 
</div>

<div align="justify">
Overall, we have found differences both within and across subreddits in the use of different orthographoc realizations of the words `[``okay,` `ok,` `Ok,` `O.K,` `okey,` `k``]`. Users of Reddit may make use of these forms. similar to how speakers may use different intonation to change the meaning of a single word.
 
</div>



***


## Individual Contributions

| Team Member           | Contributions                                             |
|-----------------------|-----------------------------------------------------------|
| Alexander Weyhe       | Lesk Algorithm, Graph Design                              |                                                       
| Kascha Kruschwitz     | Data Preprocessing, Sentiment Analysis                    |
| Ludmila Bajuk         | Word2Vec Model, Pairwise Similarity                       |

## References

Smetanin, S. (2018, November 16). Google News and Leo Tolstoy: Visualizing word2vec word embeddings with T-Sne. Medium. https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-embeddings-with-t-sne-11558d8bd4d 

Word embeddings in NLP. GeeksforGeeks. (2024, January 5). https://www.geeksforgeeks.org/word-embeddings-in-nlp/ 

Beach, Wayne A.(1993). Transitional regularities for ‘casual’ "Okay" usages. Journal of Pragmatics, vol. 19, no. 4, pp. 325-352. https://doi.org/10.1016/0378-2166(93)90092-4

Condon, Sherri L. (1986). The discourse functions of OK. Semiotica, vol. 60, no. 1-2, pp. 73-102. https://doi.org/10.1515/semi.1986.60.1-2.73

Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

Adegbija, E., & Bello, J. (2001). The semantics of ‘okay’(OK) in Nigerian English. World Englishes, 20(1), 89-98.

Lee, J. M. (2017). The multifunctional use of a discourse marker okay by Korean EFL teachers. 외국어교육연구 (Foreign Language Education Research), 21, 41-65.

Völske, M., Potthast, M., Syed, S., & Stein, B. (2017, September). Tl; dr: Mining reddit to learn automatic summarization. In Proceedings of the Workshop on New Frontiers in Summarization (pp. 59-63).













