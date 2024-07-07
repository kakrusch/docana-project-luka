
## Setup The Report Template

-  Add all your code, such as Python scripts and Jupyter notebooks, to the `code` folder. Use markdown files for your project report. [Here](https://docs.gitlab.com/ee/user/markdown.html) you can read about how to format Markdown documents. 

1
2. **Configure GitHub Pages:** Navigate to `Settings` -> `Pages` in your newly forked repository. Under the `Branch` section, change from `None` to `master` and then click `Save`.

3. **Customize Configuration:** Modify the `_config.yml` file within your repository to personalize your site. Update the `title:` to reflect the title of your project and adjust the `description:` to provide a brief summary.

4. **Start Writing:** Start writing your report by modifying the `README.md`. You can also add new Markdown files for additional pages by modifying the `_config.yml` file. Use the standard [GitHub Markdown syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) for formatting. 

5. **Access Your Site:** Return to `Settings` -> `Pages` in your repository to find the URL to your live site. It typically takes a few minutes for GitHub Pages to build and publish your site after updates. The URL to access your live site follows this schema: `https://<<username>>.github.io/<<repository_name>>/`

***

# Project Title

_Group members: Alex Weyhe, Kascha Kruschwitz, Ludmila Bajuk

## Introduction

Start off by setting the stage for your project. Give a brief overview of relevant studies or work that have tackled similar issues. Then, clearly describe the main question or problem your project is designed to solve.

### Research Questions
- Is the meaning of OK and its variations always the same?
   - Are the different spellings associated with different uses, meanings etc...?
   - Are there differences across topics/user types?
  
Hypothesis:
- we expect differences 

## Background 
- literature on different uses of ok
- different word senses


## Dataset

Provide a short description of the dataset used in your project. Focus on highlighting the aspects that are particularly relevant to your work.

## Methods

### Setup 

Outline the tools, software, and hardware environment, along with configurations used for conducting your experiments. Be sure to document the Python version and other dependencies clearly. Provide step-by-step instructions on how to recreate your environment, ensuring anyone can replicate your setup with ease:

```bash
conda create --name myenv python=<version>
conda activate myenv
```

Include a `requirements.txt` file in your project repository. This file should list all the Python libraries and their versions needed to run the project. Provide instructions on how to install these dependencies using pip, for example:

```bash
pip install -r requirements.txt
```

```
print("hellos woerld")
```

#### What we did
 - count post number to pick corpora
 - extract explatives using POS tagger: adjust research question to OK

### Experiments

Report how you conducted the experiments. We suggest including detailed explanations of the preprocessing steps and model training in your project. For the preprocessing, describe  data cleaning, normalization, or transformation steps you applied to prepare the dataset, along with the reasons for choosing these methods. In the section on model training, explain the methodologies and algorithms you used, detail the parameter settings and training protocols, and describe any measures taken to ensure the validity of the models.

## Results and Discussion
- wordsenses - lesk
- sentiment - 
- pairwise similarity of similar words within embedding: 3 very similar, 2 very different ('O.K', 'okey'), 'k' unreliable
- cosine similarity of OKs within subbreddit - 



Present the findings from your experiments, supported by visual or statistical evidence. Discuss how these results address your main research question.

## Conclusion

Summarize the major outcomes of your project, reflect on the research findings, and clearly state the conclusions you've drawn from the study.

## Contributions

| Team Member  | Contributions                                             |
|--------------|-----------------------------------------------------------|
| Alex Weyhe            | Data collection, preprocessing, model training, evaluation|                                                       |
| Kascha Kruschwitz     | ...                                                       |
| Ludmila Bajuk         | ...                                                       |

## References

Include a list of academic and professional sources you cited in your report, using an appropriate citation format to ensure clarity and proper attribution.

