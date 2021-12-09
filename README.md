![header](./data/header_banner.jpeg "Header")

# Machine Learning vs. March Madness
Authors: [Matthew Reinhart](https://www.linkedin.com/in/matthew-reinhart-1bb372173/), [Mendy Nemanow](https://www.linkedin.com/in/mendy-nemanow-2594ab225/), [Paul Lindquist](https://www.linkedin.com/in/paul-lindquist/), [TJ Bray](https://www.linkedin.com/in/thomas-tj-bray-24499354/)

## Overview
Sports gambling is one of the fastest growing industries in the country, with states continuing to pass betting-friendly legislation and companies like DraftKings, FanDuel and BetMGM experiencing increased, year-over-year revenue. One of the premier sports betting events of the year is the NCAA college basketball postseason tournament, commonly known as *March Madness*.

We target college basketball because of our domain knowledge and the talent disparity within the sport. In professional sports, the talent gap between the best and worst teams is very small. It's unusual to see NBA spreads greater than 10 points. In college basketball, this happens regularly. Teams are 25-point underdogs on a given night and that creates betting opportunity. There also tends to be more regular competition in college basketball. With only around 30 games in a season, teams don't have the luxury to take nights off like we see in an 82-game NBA season. And with constant effort comes more predictable outcomes.

## Business Objective
This project posits that we run a sports gambling company. We offer our customers advisory services during the busiest time of the sports year on events with the greatest amount of betting action.

To maximize returns, we run a series of machine learning algorithms to model predictions for single games in the NCAA tournament. We pay particular focus to accurately predicting underdog team wins, as doing so yields higher payouts. Accuracy, and more specifically *predictive* accuracy, is paramount in selecting our models, as we strive to minimize risk for our customers.

## Data
This project uses datasets from Kaggle's *[March Machine Learning Mania 2021](https://www.kaggle.com/c/ncaam-march-mania-2021/data)* competition.

## Methodology
We set the win/loss outcome for the favored team as the binary target variable, with 1 equaling a win for the favored team and 0 equaling a win for the underdog. Rankings are assigned using the reputable KenPom ratings.

We then use an iterative approach to build 6 predictive, classification models: Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Bagging classifier and XGBoost. We utilize hyperparameter tuning, cross-validation and scoring to select the highest performing, predictive models. This approach is applied to regular season, postseason and cumulative postseason data.

## Results
After comparing accuracy scores across all 6 of our models, the top 3 performers are Logistic Regression, XGBoost and Random Forest. While XGBoost and Logistic Regression yield the same accuracy score, we opt to go with latter because its standard deviation is half of that of XGBoost's.

![img1](https://i.ibb.co/G2kV1bW/a.png)

Our model consistently outperforms the baseline (i.e. only betting on favored teams) in postseason tournaments.

![img2](https://i.ibb.co/3kbq72V/e.png)

There's inherent value in correctly betting on underdogs, as that yields higher payouts. We give specific focus to those predictions to assess our model's performance.

![img4](https://i.ibb.co/NSLBHBp/d.png)

The results are very strong.

![img3](https://i.ibb.co/p37vTPr/c.png)

## Conclusions
TBD

## For More Information
Please review our full analysis in our Jupyter Notebook or presentation deck.

For additional questions, please contact [Matthew](https://www.linkedin.com/in/matthew-reinhart-1bb372173/), [Mendy](https://www.linkedin.com/in/mendy-nemanow-2594ab225/), [Paul](https://www.linkedin.com/in/paul-lindquist/) or [TJ](https://www.linkedin.com/in/thomas-tj-bray-24499354/).

## Respository Structure
```
├── README.md                           <- The top-level README for reviewers of this project
├── final_notebook.ipynb                <- Narrative documentation of analysis in Jupyter notebook
├── project_presentation.pdf            <- PDF version of project presentation
├── data                                <- Both sourced externally and generated from code
└── images                              <- Both sourced externally and generated from code
```
