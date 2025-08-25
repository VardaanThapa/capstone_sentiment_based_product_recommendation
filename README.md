# Sentiment based Product Recommendation System Capstone
> 

## Table of Contents
* [General Information](#general-information)
* [Approach](#approach)
* [Deployment](#deployment)
* [Technologies Used](#technologies-used)
* [Contact](#contact)
* [License](#license)

## General Information
'Ebuss' has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products. With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

#### Objective
As a Senior ML Engineer, you are tasked with building a model that will improve the recommendations given to the users given their past reviews and ratings.

## Approach

Following is the step-by-step approach followed to solve the given problem statement:
1. Data Understanding
2. Data Cleaning
3. Text Preprocessing
4. Exploratory Data Analysis
5. Feature Extraction
6. Sentiment Model Building
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Hypertuned XGBoost  
7. Sentiment Model Inference
8. Recommendation System Building
  - User-based recommendation
  - Item-based recommendation
9. Recommendation System Inference
10. User-based fine-tuned Recommendations
11. Deployment

## Deployment

- Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli#install-the-heroku-cli)
- Confirm Heroku has been installed correctly by running `heroku --version`
- Login into Heroku CLI with `heroku login`
- Create a new app with `heroku create capstone-sentiment-recommendation`
- Deploy to Heroku using the main branch `git push heroku main`
- Restrict app scaling by running `heroku ps:scale web=1`
- Open the app on browser by running `heroku open`
- Check the logs to see activity by running `heroku logs --tail`

Deployed app : https://capstone-recommendation-engine-092a3b5549a1.herokuapp.com/

Github link : https://github.com/VardaanThapa/capstone_sentiment_based_product_recommendation

## Technologies Used
- python (ver 3.12.7)
- numpy (ver 1.26.4)
- pandas (ver 2.2.2)
- matplotlib (ver 3.9.2)
- seaborn (ver 0.13.2)
- wordcloud (ver 1.9.4)
- nltk (ver 3.9.1)
- sklearn (ver 1.5.1)
- xgb (ver 2.1.3)
- swifter (ver 1.4.0)
- pickle (ver 4.0)
- spacy (ver 1.4.0)

## Contact
Created by [@VardaanThapa](https://github.com/VardaanThapa) - feel free to contact me!

## License
This project is open source and available under the [MIT License](LICENSE.md)