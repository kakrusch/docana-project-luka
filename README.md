
## Setup The Report Template

1.  Add all your code, such as Python scripts and Jupyter notebooks, to the `code` folder. Use markdown files for your project report. [Here](https://docs.gitlab.com/ee/user/markdown.html) you can read about how to format Markdown documents. 

2. **Configure GitHub Pages:** Navigate to `Settings` -> `Pages` in your newly forked repository. Under the `Branch` section, change from `None` to `master` and then click `Save`.

3. **Customize Configuration:** Modify the `_config.yml` file within your repository to personalize your site. Update the `title:` to reflect the title of your project and adjust the `description:` to provide a brief summary.

4. **Start Writing:** Start writing your report by modifying the `README.md`. You can also add new Markdown files for additional pages by modifying the `_config.yml` file. Use the standard [GitHub Markdown syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) for formatting. 

5. **Access Your Site:** Return to `Settings` -> `Pages` in your repository to find the URL to your live site. It typically takes a few minutes for GitHub Pages to build and publish your site after updates. The URL to access your live site follows this schema: `https://<<username>>.github.io/<<repository_name>>/`

***

# Project Title

Group members: Alex Weyhe, Kascha Kruschwitz, Ludmila Bajuk

## Introduction (Background)
Start off by setting the stage for your project. Give a brief overview of relevant studies or work that have tackled similar issues. Then, clearly describe the main question or problem your project is designed to solve.

                                       FEEL FREE TO CHANGE EVERYTHING YOU WANT

In spoken language the word “ok” is uttered frequently and can appear in different contexts. Multiple studies have investigated in what different ways the word is used in English and other languages. According to Burchifield (1982 in Adegbija & Bello, 2019), the word “okay” can appear as a response or phrase to show acceptance or agreement. Additionally, it can be classified as an adjective to mean “correct”, good”, or “satisfactory. Condon (1986 in Lee, 2008) introduces “okey” as a marker of transitions in discourse. Finally, in another study by Adegbija & Bello (2001), the usage of “O.K.” was investigated in Nigerian English and compared to (“Standard”) English. In Nigerian English it can be used as a gap filler, surprise, or discourse termination. [better transition needed]Here, the question comes to mind if one uses the word in the same contexts in written language. Consequently, the focus of this project is to investigate the different usage of “ok” in written language, in particular in online forums. The peculiarity in written language arises from the different spelling variations of the word – as it can be seen by the different spellings in the preceding sentences. Thus, this project investigates whether various spelling forms are used differently in written language and answers the research question: In how far do variations of the word “ok” differ in their usage in online forums. 

### Research Questions
- Is the meaning of OK and its variations always the same?
   - Are the different spellings associated with different uses, meanings etc...?
   - Are there differences across topics/user types?
  
Hypothesis:
- we expect differences 

literature 



- Condon (1986 in Lee, 2008): verbal okay has 3 pragmatic uses in decision-making interaction
   - the beginning of the entire discourse.
   - the transition from a non-decision sequence frame to a decision discourse
   - the transition from one decision sequence to the next
   - Marker of transition
- Beach (1993 in Lee, 2008):
   - free-standing receipt marker
   - Again marker of discourse transitions


- Burchfield (1982 in Adegbija & Bello, 2019) - american english (? - could be british)
     1. Adjective uses: “correct,' `all right,' `satisfactory,'`good,' `well,' `everything is in order,' or `in good health
     2.  Phrase: “acceptable to me”
     3.  Response particle like “yes” or “certainly”
     4.  “signal social or cultural acceptance, being fashionable, having or showing prestige, belonging to a high class”
     5.  Transitive Verbs
- Nigerian uses (Adegbija and Bello, 2019):
     1. OK as a gap filler
     2. Expressing a sense of surprise
     3. As a marker of discourse termination “stop it!”
 


## Dataset
The dataset used is 

https://aclanthology.org/W17-4508.pdf

Provide a short description of the dataset used in your project. Focus on highlighting the aspects that are particularly relevant to your work.
what parts of the datas set we used and why 
 - count post number to pick corpora

For this project, we used the “webis/tldr-17” dataset from Hugging Face. A special feature of this dataset is that it only contains Reddit posts with the abbreviation “tl;dr”. This stands for “too long; didn’t read” and has become a popular tool for authors from Reddit posts to give a short summary of their posts’ contents. The dataset consists of 3,848,330 posts across a total of 29651 subreddits. Each data point contains the following data fields: author, body, normalizedBody, subreddit, subreddit_id, id, content summary. For our work the different subreddits are of particular interest because we want to compare our research question across them. Additionally, since we used posts from the four biggest subreddits, it was necessary that for each post its subreddit is specified. 

- we used the "content" part only (not including the tldr) to train the model

  
## METHODS

1. got the 4 largest subreddits and put it in a dataframe to access it better and save it in a csv file so that we didnt have to load the data every time
    - AskReddit, League of Legends, Relationships, TIFU
2. then, we custom tokenize the subreddits. We removed the punctuation for all tokens except for the O.K and made every lower except for Ok and O.K
        - for the verisions of ok case and punctuation matters: Ok vs ok vs O.K., but for other words it introduced too much noise
3. we got all the oks in each subreddits and got a list of all the oks that appeared in every single subreddit (more than 5 times to generate mostly good embeddings).
    - 'okay', 'ok', 'Ok', 'O.K', 'okey', 'k'
4. We built a skipgram word2vec model and trained a model for each subredit and save it to be easily loaded whenever we needed it.
    -   trained on entire content of each subreddit for words appearing more than 5 times
    -   vector size of 100
5. With the model we check the 100 most similar words for each ok in each subreddit, the most frequent oks in each and the vector representation of each ok
    - (include list for each sub and each ok?)
6. we analysis the different version of ok with 4 comparison metrics.
7. 

| Subreddit  || Words in Common from 10 most similar      |
|----------------||-----------------------------------------|
| 'okay'         || ['ok', 'alright', 'fine', 'yeah', 'yea']  |                                                       
| 'ok'           || ['okay', 'fine', 'alright']               |
| 'Ok'           ||['yeah', 'hey']                            |
| 'O.K'          || []                                        |
| 'okey'         || []                                        | 
| 'k'            || []                                        |



Askreddit: {'okay': ['alright', 'fine', 'allright', 'mhm'], 'ok': ['alright', 'fine', 'allright', 'mhm'], 'Ok': ['yeah', 'uh', 'yea', 'mhm', 'ummmm'], 'O.K': ['poopie', 'yyes', 'mhmm', 'oughta', 'ummmmmm'], 'okey': ['thehealeroftri', 'dany', 'batcave', 'yeaaa', 'spotlighted'], 'k': ['j', 'c', 'g', 'b', 'p']}
League of Legends {'okay': ['alright', 'yeah', 'fine'], 'ok': ['alright', 'yeah', 'fine', 'yea'], 'Ok': ['yeah', 'hey', 'yea', 'welp'], 'O.K': ['autoloss', 'dammed', 'cartman', 'neato', 'apealing'], 'okey': ['yeah', 'idek', 'yeh', 'ahh', 'yea'], 'k': ['t1', 'sgw', 'lgim', 'telecom', 'starhorn']}
TIFU {'okay': ['alright', 'fine', 'overreacting', 'yeah'], 'ok': ['alright', 'fine', 'yeah'], 'Ok': ['yeah', 'eh', 'yea', 'yrrah321'], 'O.K': ['poogadi', 'bluffing', 'excommunicated', 'rue', 'honoured'], 'okey': ['nooooope', 'fuuuuuuuuuuuuuck', 'neato', 'uhhuh', 'fuckfuckfuckfuckfuck'], 'k': ['h', 'j', 'kelly', 'linette', 'l']}
Relationships {'okay': ['alright', 'fine', 'allright'], 'ok': ['fine', 'alright', 'allright'], 'Ok': ['yeah', 'thanksokay', 'hey', 'rr_a', 'alright'], 'O.K': ['concur', 'mutuals', 'tabled', 'sidekick', 'bullcrap'], 'okey': ['ahah', 'rehearsed', 'ehhhh', 'wha', 'noncommital'], 'k': ['john', 'j', 'jane', 'h', 'sarah']}




### Prepocessing



### Model Setup 

Outline the tools, software, and hardware environment, along with configurations used for conducting your experiments. Be sure to document the Python version and other dependencies clearly. Provide step-by-step instructions on how to recreate your environment, ensuring anyone can replicate your setup with ease:

```bash
conda create --name myenv python=<version>
conda activate myenv
```

Include a `requirements.txt` file in your project repository. This file should list all the Python libraries and their versions needed to run the project. Provide instructions on how to install these dependencies using pip, for example:

```bash
pip install -r requirements.txt
```


## Comparison Metrics and Results

Report how you conducted the experiments. We suggest including detailed explanations of the preprocessing steps and model training in your project. For the preprocessing, describe  data cleaning, normalization, or transformation steps you applied to prepare the dataset, along with the reasons for choosing these methods. In the section on model training, explain the methodologies and algorithms you used, detail the parameter settings and training protocols, and describe any measures taken to ensure the validity of the models.

### words senses 

### cosine similarity 
- within each subreddit we calculated the cosine similarity for each two Oks using Word2Vec's inbuilt metric
      - see how similar each two OKs are
- plot the similarity using  "t-distributed Stochastic Neighbor Embedding" to visualize the distribution
      - shown below
- Results:
      - within all subreddits "okay - ok" have the most similar embeddings
      - Askreddit and Tifu have most similar distribution

| ![Graph of Cosine Similarity of OKs in Askreddit](/figures/ok_askred.png)  | ![Graph of Cosine Similarity of OKs in League of Legends](/figures/ok_lol.png) |
| ![Graph of Cosine Similarity of OKs in Relationships](/figures/ok_relations.png)  | ![Graph of Cosine Similarity of OKs in TIFU](/figures/ok_tifu.png) |



### overlapping words

A pairwise semantic similarity analysis was carried out to compare the overlapping most similar words of each OK and their corresponding similarity scores. For this analysis, we extracted the similar words of all OKs in two subreddits and compares only the overlapping words. We paired all the subreddit with each other obtaining 6 different pairs. In this case, the only OKs that had overlapping similar were 'ok', 'okay', and 'Ok'; 'k' only had similar overlapping words or letter with in a few pairs, however, they were not reliable since it was a one letter embedding.

For each pair we got the mean semantic similarity for each similar and overlapping word for each OK, so that we got now one score for each word in the list of overlaping words. Once every word had only one score, we averaged those scores (in the case if pair1: pair1 = 
get_most_similar_words(tifu_sim_ok, askreddit_sim_ok)
 {'okay': ['ok', 'alright', 'fine', 'allright', 'yeah', 'O.K', 'yea']
we get the mean value of the scores we got from the first 6 words found in the 'okay' overlapping words list). 

After getting the mean score for each okay in each subreddit, we calculated the final mean score of each OK across subreddits obtaining in that way the semantic similarity of the same OK in different contexts.
![Word Overlapping Semantic Similarity](https://github.com/kakrusch/docana-project-luka/assets/162272922/5a572432-a872-4175-96e5-f51b15b836ae)



| Version of OK  | Words in Common from 10 most similar      |
|----------------|-----------------------------------------|
| 'okay'         | ['ok', 'alright', 'fine', 'yeah', 'yea']  |                                                       
| 'ok'           | ['okay', 'fine', 'alright']               |
| 'Ok'           |['yeah', 'hey']                            |
| 'O.K'          | []                                        |
| 'okey'         | []                                        | 
| 'k'            | []                                        |


### Sentiment analysis 
Finally, sentiment analysis was conducted based on the 50 most similar words for each OK in each subreddit, using a lexicon provided in the VADER sentiment analyzer (Hutto & Gilbert, 2014). This lexicon included human-rated sentiments for many words specific to social media discourse and was thus deemed well-suited for the task of calculating the sentiments on Reddit. 

Across subreddits: “okay” and “ok” overall positive, others tending towards neutral


| ![Graph of the Sentiments of okays across subreddits](/figures/Sentiment of different okays across subreddits.png)  | ![Graph of the Sentiments of all subreddits expressed by ok](/figures/sent-all-okays.png) |


Within subreddits: tifu all positive, in other subreddits O.K is generally negative


| ![Graph of the Sentiments of okays in AskReddit](/figures/Sentiment of different okays in AskReddit.png)  | ![Graph of the Sentiments of okays in League of Legends](/figures/Sentiment of different okays in League of Legends.png) |
| ![Graph of the Sentiments of okays in Relationships](/figures/Sentiment of different okays in Relationships.png)  | ![Graph of the Sentiments of okays in TIFU](/figures/Sentiment of different okays in TIFU.png) |




## Discussion

- Is the meaning of OK and its variations always the same?
   - Are the different spellings associated with different uses, meanings etc...?
   - Are there differences across topics/user types?
 
  
   -  the standard spellings "okay", "ok" and "Ok" seem to be mostly associated with the standard uses
         -   consistent across subreddits
         -  most similar to words like "alright", "fine", "yeah" and posititve sentiment associated with meanings like good)
         -  like adjectival/response particle uses from spoken langugage uses
   - "okey", "O.K" and "k" more unique
         - "k" - not reliable embeddings
         - "okey":
               - different cosine similarity in each subreddit
               - overall very low across subreddit similarity
               - very negative in Askreddit but positive in all others
         - O.K.:
             - consistently most negative (except in TIFU)   
    - cosine similarity distributions in TIFU and AskReddit most similar - consistent with having the most similar uses
    - league of legends uses ok-variations differently to others (eg. more negative sentiment overall)- consistent with having the most different topic

  
- wordsense - lesk
- sentiment - 
- pairwise similarity of similar words within embedding: 3 very similar, 2 very different ('O.K', 'okey'), 'k' unreliable
- cosine similarity of OKs within subbreddit - 



Present the findings from your experiments, supported by visual or statistical evidence. Discuss how these results address your main research question.

## Conclusion

Summarize the major outcomes of your project, reflect on the research findings, and clearly state the conclusions you've drawn from the study.

## Contributions

| Team Member           | Contributions                                             |
|-----------------------|-----------------------------------------------------------|
| Alex Weyhe            | Data collection, preprocessing, model training, evaluation|                                                       
| Kascha Kruschwitz     | ... hello                                                 |
| Ludmila Bajuk         | ...  :)                                                   |

## References

Include a list of academic and professional sources you cited in your report, using an appropriate citation format to ensure clarity and proper attribution.

Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

Adegbija, E., & Bello, J. (2001). The semantics of ‘okay’(OK) in Nigerian English. World Englishes, 20(1), 89-98.

Lee, J. M. (2017). The multifunctional use of a discourse marker okay by Korean EFL teachers. 외국어교육연구 (Foreign Language Education Research), 21, 41-65.

Völske, M., Potthast, M., Syed, S., & Stein, B. (2017, September). Tl; dr: Mining reddit to learn automatic summarization. In Proceedings of the Workshop on New Frontiers in Summarization (pp. 59-63).









