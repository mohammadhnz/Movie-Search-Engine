import re
import sys

from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import time
import pandas as pd
import urllib
from transformers import pipeline
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IMDBCrawler:
    def __init__(self, csv_path, sentiment_analysis_model_name="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                 start_index=28901, end_index=35000, review_limit=50, save_period=25, result_path="reviews-sen.csv"):
        self.sentiment_analysis = pipeline("sentiment-analysis", model=sentiment_analysis_model_name,
                                      tokenizer=sentiment_analysis_model_name, max_length=512,
                                      truncation=True, device=device)
        self.csv_path = csv_path
        self.start_index = start_index
        self.end_index = end_index
        self.review_limit = review_limit
        self.save_period = save_period
        self.result_path = result_path

    @staticmethod
    def search_movie_in_searchbar(driver, movie_title):
        # driver.get("https://imdb.com")
        # search_bar = driver.find_element(By.ID, "suggestion-search")
        # search_bar.send_keys(movie_title)
        # search_button = driver.find_element(By.ID, "suggestion-search-button")
        # search_button.click()
        query = urllib.parse.urlencode({"q": movie_title})
        driver.get(f"https://imdb.com/find/?{query}")

    @staticmethod
    def open_movie_page_from_search_results(driver, movie_title):
        try:
            see_more_element = driver.find_element(By.CSS_SELECTOR, "span.ipc-see-more > button")
            see_more_element.click()
            time.sleep(1)
        except:
            pass
        movie_elements = driver.find_elements(By.XPATH, "//section[@data-testid='find-results-section-title']//div[last()]//ul//li")
        for movie_element in movie_elements:
            try:
                title = movie_element.find_element(By.XPATH, "div[last()]//div//a").text
                if title.lower() == movie_title.lower():
                    movie_element.click()
                    return title
            except:
                pass
        movie_element = movie_elements[0]
        title = movie_element.find_element(By.XPATH, "div[last()]//div//a").text
        movie_element.click()
        return title

    @staticmethod
    def open_movie_reviews_page(driver):
        try:
            movie_rating = driver.find_element(By.XPATH, "//div[@data-testid='hero-rating-bar__aggregate-rating__score']//span").get_attribute("innerHTML")
        except:
            movie_rating = None
        try:
            total_rating_num = driver.find_element(By.XPATH, "//div[@data-testid='hero-rating-bar__aggregate-rating']//a//span//div//div[last()]//div[last()]").get_attribute("innerHTML")
        except:
            total_rating_num = None

        movie_reviews_url = driver.find_element(By.XPATH, "//div[@data-testid='reviews-header']//div//a").get_attribute("href")
        driver.get(movie_reviews_url)
        return movie_rating, total_rating_num

    @staticmethod
    def extract_rating_from_review_element(element):
        try:
            rates = element.find_elements(By.CSS_SELECTOR, "div.ipl-ratings-bar > span > span")
            rate = re.search('.*(\d+).*', rates[0].text).group(1)
            max_rate = re.search('\D*(\d+).*', rates[1].text).group(1)
            return rate, max_rate
        except:
            return None, None

    @staticmethod
    def extract_title_from_review_element(element):
        try:
            title = element.find_element(By.CSS_SELECTOR, "a.title").text
            return title
        except:
            return None

    @staticmethod
    def extract_date_from_review_element(element):
        try:
            date = element.find_element(By.CSS_SELECTOR, "div.display-name-date > span.review-date").text
            return date
        except:
            return None

    @staticmethod
    def extract_content_from_review_element(element):
        try:
            content = element.find_element(By.CSS_SELECTOR, "div.content > div.text").get_attribute('innerHTML')
            return content
        except:
            return None

    @staticmethod
    def extract_review_from_element(element):
        rating = IMDBCrawler.extract_rating_from_review_element(element)
        title = IMDBCrawler.extract_title_from_review_element(element)
        date = IMDBCrawler.extract_date_from_review_element(element)
        content = IMDBCrawler.extract_content_from_review_element(element)
        return {
            "rating": rating,
            "title": title,
            "date": date,
            "content": content
        }

    @staticmethod
    def extract_reviews_in_page(driver, review_limit):
        try:
            all_review_num = driver.find_element(By.XPATH, "//div[@class='lister']//div[@class='header']//div//span").text
            all_review_num = re.search('(\d+).*', all_review_num).group(1)
        except:
            all_review_num = 0
        review_elements = []
        while True:
            try:
                review_elements = driver.find_elements(By.CSS_SELECTOR, "div.lister-item-content")
            except:
                break
            try:
                if len(review_elements) >= review_limit:
                    break
                more_button = driver.find_element(By.ID, "load-more-trigger")
                more_button.click()
            except:
                break

        reviews = [IMDBCrawler.extract_review_from_element(x) for x in review_elements]
        return reviews, all_review_num

    @staticmethod
    def search_movie_in_google(driver, movie_title):
        query = urllib.parse.urlencode({"q": movie_title})
        driver.get(f"https://google.com/search/?{query}")

    @staticmethod
    def open_movie_page_from_google_search_results(driver, movie_title):
        sys.exit()
        return None, None

    @staticmethod
    def get_movie_reviews(driver, movie_title, review_limit, search_in_imdb=True):
        if search_in_imdb:
            try:
                IMDBCrawler.search_movie_in_searchbar(driver, movie_title)
                time.sleep(1)
            except:
                return None, None, None, None, None, f"error in search_movie_in_searchbar of movie {movie_title}"
            try:
                opened_movie_title = IMDBCrawler.open_movie_page_from_search_results(driver, movie_title)
                time.sleep(1)
                movie_main_page_url = driver.current_url
            except:
                return None, None, None, None, None, f"error in open_movie_page_from_search_results of movie {movie_title}"
        else:
            try:
                IMDBCrawler.search_movie_in_google(driver, movie_title)
                time.sleep(1)
            except:
                return None, None, None, None, None, f"error in search_movie_in_searchbar of movie {movie_title}"
            try:
                opened_movie_title = IMDBCrawler.open_movie_page_from_google_search_results(driver, movie_title)
                time.sleep(1)
                movie_main_page_url = driver.current_url
            except:
                return None, None, None, None, None, f"error in open_movie_page_from_search_results of movie {movie_title}"
        try:
            movie_rating, total_ratings_num = IMDBCrawler.open_movie_reviews_page(driver)
            time.sleep(1)
        except:
            return opened_movie_title, movie_main_page_url, None, None, \
                None, f"error in open_movie_reviews_page of movie {movie_title}"
        reviews, all_reviews_num = IMDBCrawler.extract_reviews_in_page(driver, review_limit)
        return opened_movie_title, movie_main_page_url, movie_rating, total_ratings_num, all_reviews_num, reviews

    @staticmethod
    def add_review_to_df(df, movie_title, opened_movie_title, movie_main_page_url, movie_rating,
                         total_ratings_num, all_reviews_num, reviews):
        for review in reviews:
            rate = None if review["rating"][0] is None or review["rating"][1] is None else "/".join(review["rating"])
            df.loc[len(df.index)] = [movie_title, opened_movie_title, movie_main_page_url, movie_rating, total_ratings_num,
                                     all_reviews_num, rate, review["title"], review["date"], review["content"]]

    def sentiment_analise_df(self, df):
        batch_size = 128
        for i in range(len(df) // batch_size + 1):
            start_index = i * batch_size
            end_index = min(len(df), (i + 1) * batch_size)
            df.loc[start_index:end_index, "Content"] = list(
                map(lambda x: x["label"], self.sentiment_analysis(list(df.loc[start_index:end_index, "Content"].apply(str)))))

    @staticmethod
    def save_df(df, cols, output_path):
        mode = "a" if os.path.exists(output_path) else "w"
        header = True if mode == "w" else False
        df.loc[:, cols].to_csv(output_path, mode=mode, index=False, header=header)

    def get_movies(self, df: pd.DataFrame, start_index, end_index, review_limit, save_period, result_path):
        # options = Options()
        # options.add_argument("--headless")
        # options.add_argument("--no-sandbox")
        # options.add_argument(
        #     "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36")
        # driver = webdriver.Chrome(options=options)

        driver = webdriver.Chrome()
        cols = ['Search Title', 'Found Title', 'Found Movie URL', 'Movie Rating', 'Movie Total Rating', 'Total Reviews', "Comment Rating", "Title", "Date", "Content"]
        cols_to_save = ['Search Title', 'Found Title', 'Movie Rating', 'Movie Total Rating', 'Total Reviews', "Comment Rating", "Content"]
        res = pd.DataFrame(columns=cols)
        for i in range(start_index, min(end_index, len(df))):
            movie_title = df.loc[i, "Title"]
            print(f"trying to get movie number {i}: {movie_title}")

            try:
                opened_movie_title, movie_main_page_url, movie_rating, \
                    total_ratings_num, all_reviews_num, reviews = IMDBCrawler.get_movie_reviews(driver, movie_title, review_limit)
                if not isinstance(reviews, list):
                    res.loc[len(res)] = [movie_title, opened_movie_title, movie_main_page_url, movie_rating, total_ratings_num,
                                         all_reviews_num, None, None, None, None]
                    print(reviews)
                else:
                    IMDBCrawler.add_review_to_df(res, movie_title, opened_movie_title, movie_main_page_url, movie_rating,
                                     total_ratings_num, all_reviews_num, reviews)
            except Exception as e:
                print("error in getting and saving reviews\n", e)

            time.sleep(1)
            try:
                if i % save_period == 0:
                    self.sentiment_analise_df(res)
                    IMDBCrawler.save_df(res, cols_to_save, result_path)
                    res = pd.DataFrame(columns=cols)

            except Exception as e:
                print(e)
                print("error in saving\n", e)

        self.sentiment_analise_df(res)
        IMDBCrawler.save_df(res, cols_to_save, result_path)

    def start_crawling(self):
        wiki_movies = pd.read_csv(self.csv_path)
        self.get_movies(wiki_movies, start_index=self.start_index, end_index=self.end_index,
                        review_limit=self.review_limit, save_period=self.save_period, result_path=self.result_path)


# r = IMDBCrawler("Persian movie dataset.csv", sentiment_analysis_model_name="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
#                 start_index=0, end_index=float('inf'), review_limit=50, save_period=25, result_path="reviews-sen.csv")
# r.start_crawling()
