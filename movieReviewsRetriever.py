import re
import time
import urllib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

def search_movie_in_searchbar(driver, movie_title):
    query = urllib.parse.urlencode({"q": movie_title})
    driver.get(f"https://imdb.com/find/?{query}")


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


def extract_rating_from_review_element(element):
    try:
        rates = element.find_elements(By.CSS_SELECTOR, "div.ipl-ratings-bar > span > span")
        rate = re.search('.*(\d+).*', rates[0].text).group(1)
        max_rate = re.search('\D*(\d+).*', rates[1].text).group(1)
        return rate, max_rate
    except:
        return None, None


def extract_title_from_review_element(element):
    try:
        title = element.find_element(By.CSS_SELECTOR, "a.title").text
        return title
    except:
        return None


def extract_date_from_review_element(element):
    try:
        date = element.find_element(By.CSS_SELECTOR, "div.display-name-date > span.review-date").text
        return date
    except:
        return None


def extract_content_from_review_element(element):
    try:
        content = element.find_element(By.CSS_SELECTOR, "div.content > div.text").get_attribute('innerHTML')
        return content
    except:
        return None


def extract_review_from_element(element):
    rating = extract_rating_from_review_element(element)
    title = extract_title_from_review_element(element)
    date = extract_date_from_review_element(element)
    content = extract_content_from_review_element(element)
    return {
        "rating": rating,
        "title": title,
        "date": date,
        "content": content
    }


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

    reviews = [extract_review_from_element(x) for x in review_elements]
    return reviews, all_review_num


def get_movie_reviews(driver, movie_title, review_limit):
    try:
        search_movie_in_searchbar(driver, movie_title)
        # time.sleep(1)
    except:
        return None, None, None, None, None, f"error in search_movie_in_searchbar of movie {movie_title}"
    try:
        opened_movie_title = open_movie_page_from_search_results(driver, movie_title)
        time.sleep(1)
        movie_main_page_url = driver.current_url
    except:
        return None, None, None, None, None, f"error in open_movie_page_from_search_results of movie {movie_title}"
    try:
        movie_rating, total_ratings_num = open_movie_reviews_page(driver)
        # time.sleep(1)
    except:
        return opened_movie_title, movie_main_page_url, None, None, \
            None, f"error in open_movie_reviews_page of movie {movie_title}"
    reviews, all_reviews_num = extract_reviews_in_page(driver, review_limit)
    res = {
        "opened movie title": opened_movie_title,
        "movie main page url": movie_main_page_url,
        "movie rating": movie_rating,
        "total ratings num": total_ratings_num,
        "all reviews num": all_reviews_num,
        "reviews": reviews
    }
    return res


def get_movies_reviews(movie_titles, review_limit=5, hidden=True):

    if hidden:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36")
        driver = webdriver.Chrome(options=options)
    else:
        driver = webdriver.Chrome()
    res = {}
    for movie_title in movie_titles:
        res[movie_title] = get_movie_reviews(driver, movie_title, review_limit)
    return res
