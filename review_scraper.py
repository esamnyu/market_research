import pandas as pd
from google_play_scraper import Sort, reviews_all
import sys
import logging
import argparse
import time

def setup_logging():
    """
    Sets up the logging configuration.
    Logs are saved to 'scrape_reviews.log' and also output to the console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("scrape_reviews.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def scrape_google_play_reviews(app_id, output_file='sweepy_cleaning_reviews.xlsx', lang='en', country='us'):
    """
    Scrapes all reviews from a Google Play Store app and saves them to an Excel file.

    Parameters:
    - app_id (str): The unique application ID on Google Play Store.
    - output_file (str): The name of the output Excel file.
    - lang (str): Language code for reviews.
    - country (str): Country code for reviews.
    """
    try:
        logging.info(f"Starting to scrape reviews for app ID: {app_id}")
        logging.info(f"Fetching reviews with lang='{lang}' and country='{country}'")
        
        # Fetch all reviews
        reviews = reviews_all(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.RATING,  # You can change the sort order if needed
            filter_score_with=None  # To get all ratings
        )
        
        logging.info(f"Total reviews fetched: {len(reviews)}")
        
        if not reviews:
            logging.warning("No reviews found. Please check the app ID or try different language/country parameters.")
            return
        
        # Convert the list of reviews to a Pandas DataFrame
        df = pd.DataFrame(reviews)
        
        # Ensure required columns exist
        required_columns = ['userName', 'score', 'content', 'at']
        for col in required_columns:
            if col not in df.columns:
                logging.warning(f"Expected column '{col}' not found in reviews data.")
                df[col] = None  # Assign None to missing columns
        
        # Select and rename relevant columns
        df_selected = df[required_columns].copy()
        df_selected.rename(columns={
            'userName': 'Author',
            'score': 'Rating',
            'content': 'Review Text',
            'at': 'Review Date'
        }, inplace=True)
        
        # Export the DataFrame to an Excel file
        df_selected.to_excel(output_file, index=False)
        logging.info(f"Successfully saved {len(df_selected)} reviews to '{output_file}'.")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

def parse_arguments():
    """
    Parses command-line arguments for the script.

    Returns:
    - args: The parsed arguments containing app_id, output_file, lang, and country.
    """
    parser = argparse.ArgumentParser(description="Scrape Google Play Store app reviews.")
    parser.add_argument('--app_id', type=str, required=False, default='app.sweepy.sweepy',
                        help='Application ID (package name) of the Google Play Store app.')
    parser.add_argument('--output', type=str, required=False, default='sweepy_cleaning_reviews.xlsx',
                        help='Output Excel file name.')
    parser.add_argument('--lang', type=str, required=False, default='en',
                        help='Language code for reviews (e.g., en, es, fr).')
    parser.add_argument('--country', type=str, required=False, default='us',
                        help='Country code for reviews (e.g., us, gb, de).')
    return parser.parse_args()

def main():
    """
    The main function that orchestrates the scraping process.
    """
    setup_logging()
    args = parse_arguments()
    scrape_google_play_reviews(args.app_id, args.output, args.lang, args.country)

if __name__ == "__main__":
    main()
