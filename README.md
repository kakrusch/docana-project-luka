
## Setup The Report Template

-  Add all your code, such as Python scripts and Jupyter notebooks, to the `code` folder. Use markdown files for your project report. [Here](https://docs.gitlab.com/ee/user/markdown.html) you can read about how to format Markdown documents. 

1
2. **Configure GitHub Pages:** Navigate to `Settings` -> `Pages` in your newly forked repository. Under the `Branch` section, change from `None` to `master` and then click `Save`.

3. **Customize Configuration:** Modify the `_config.yml` file within your repository to personalize your site. Update the `title:` to reflect the title of your project and adjust the `description:` to provide a brief summary.

4. **Start Writing:** Start writing your report by modifying the `README.md`. You can also add new Markdown files for additional pages by modifying the `_config.yml` file. Use the standard [GitHub Markdown syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) for formatting. 

5. **Access Your Site:** Return to `Settings` -> `Pages` in your repository to find the URL to your live site. It typically takes a few minutes for GitHub Pages to build and publish your site after updates. The URL to access your live site follows this schema: `https://<<username>>.github.io/<<repository_name>>/`

***

# Project Title

Group members: Alex Weyhe, Kascha Kruschwitz, Ludmila Bajuk

## Introduction (Background)
Start off by setting the stage for your project. Give a brief overview of relevant studies or work that have tackled similar issues. Then, clearly describe the main question or problem your project is designed to solve.

### Research Questions
- Is the meaning of OK and its variations always the same?
   - Are the different spellings associated with different uses, meanings etc...?
   - Are there differences across topics/user types?
  
Hypothesis:
- we expect differences 

literature 

Jung, 2008

 - Condon (1986): verbal okay has 3 pragmatic uses in decision-making interaction
      - the beginning of the entire discourse.
      - the transition from a non-decision sequence frame to a decision discourse
      - the transition from one decision sequence to the next
      - Marker of transition
   - Beach (1993):
      - free-standing receipt marker
      - Again marker of discourse transitions

Adegbija and Bello, 2019
   - Burchfield (1982) - american english (? - could be british)
        1. Adjective uses: “correct,' `all right,' `satisfactory,'`good,' `well,' `everything is in order,' or `in good health
        2.  Phrase: “acceptable to me”
        3.  Response particle like “yes” or “certainly”
        4.  “signal social or cultural acceptance, being fashionable, having or showing prestige, belonging to a high class”
        5.  Transitive Verbs
   - Nigerian uses:
        1. OK as a gap filler
        2. Expressing a sense of surprise
        3. As a marker of discourse termination “stop it!”
 


#### What we did


## Dataset
The dataset used is 

https://aclanthology.org/W17-4508.pdf

Provide a short description of the dataset used in your project. Focus on highlighting the aspects that are particularly relevant to your work.
what parts of the datas set we used and why 
 - count post number to pick corpora


## METHODS

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


### comparison metrics

Report how you conducted the experiments. We suggest including detailed explanations of the preprocessing steps and model training in your project. For the preprocessing, describe  data cleaning, normalization, or transformation steps you applied to prepare the dataset, along with the reasons for choosing these methods. In the section on model training, explain the methodologies and algorithms you used, detail the parameter settings and training protocols, and describe any measures taken to ensure the validity of the models.

#### words senses 

#### cosine similarity 

| ![Graph of Cosine Similarity of OKs in Askreddit](/figures/ok_askred.png)  | ![Graph of Cosine Similarity of OKs in League of Legends](/figures/ok_lol.png) |
| ![Graph of Cosine Similarity of OKs in Relationships](/figures/ok_relations.png)  | ![Graph of Cosine Similarity of OKs in TIFU](/figures/ok_tifu.png) |

#### overlapping words


#### Sentiment analysis 
Finally, sentiment analysis was conducted based on the 50 most similar words for each OK in each subreddit, using a lexicon provided in the VADER sentiment analyzer (Hutto & Gilbert, 2014). This lexicon included human-rated sentiments for many words specific to social media discourse and was thus deemed well-suited for the task of calculating the sentiments on Reddit. 

Across subreddits: “okay” and “ok” overall positive, others tending towards neutral


| ![Graph of the Sentiments of okays across subreddits](/figures/Sentiment of different okays across subreddits.png)  | ![Graph of the Sentiments of all subreddits expressed by ok](/figures/sent-all-okays.png) |


Within subreddits: tifu all positive, in other subreddits O.K is generally negative


| ![Graph of the Sentiments of okays in AskReddit](/figures/Sentiment of different okays in AskReddit.png)  | ![Graph of the Sentiments of okays in League of Legends](/figures/Sentiment of different okays in League of Legends.png) |
| ![Graph of the Sentiments of okays in Relationships](/figures/Sentiment of different okays in Relationships.png)  | ![Graph of the Sentiments of okays in TIFU](/figures/Sentiment of different okays in TIFU.png) |




## Results and Discussion
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

Völske, M., Potthast, M., Syed, S., & Stein, B. (2017, September). Tl; dr: Mining reddit to learn automatic summarization. In Proceedings of the Workshop on New Frontiers in Summarization (pp. 59-63).
