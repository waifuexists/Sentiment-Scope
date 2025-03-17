import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import re
from collections import Counter
import nltk
from nltk.corpus import brown

# Function to download required NLTK data
def download_nltk_data():
    try:
        # Check if 'brown' corpus is available
        brown.words()
    except LookupError:
        print("Downloading required NLTK data (brown corpus)...")
        nltk.download('brown', quiet=True)
        nltk.download('punkt', quiet=True)  # For tokenization
        nltk.download('averaged_perceptron_tagger', quiet=True)  # For POS tagging
        print("NLTK data downloaded successfully.")

# List of Indian news websites with search URL templates
NEWS_SITES = {
    "NDTV": "https://www.ndtv.com/search?searchtext={}",
    "Times of India": "https://timesofindia.indiatimes.com/topic/{}",
    "India Today": "https://www.indiatoday.in/search/{}"
}

# Headers to mimic a browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def clean_text(text):
    """Clean text by removing extra whitespace and special characters."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,]', '', text)
    return text

def scrape_news_site(url, query):
    """Scrape content from a news site based on the query."""
    try:
        # Replace {} in the URL with the query
        search_url = url.format(query.replace(" ", "+"))
        response = requests.get(search_url, headers=HEADERS, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text based on common HTML tags (customize per site if needed)
        content = []
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'article']):
            text = tag.get_text(strip=True)
            if text and query.lower() in text.lower():  # Filter relevant content
                content.append(clean_text(text))

        return content if content else ["No relevant content found."]
    except requests.RequestException as e:
        return [f"Error scraping {url}: {str(e)}"]

def analyze_sentiment(text_list):
    """Perform sentiment analysis on a list of text snippets."""
    positive, negative, neutral = 0, 0, 0
    total = len(text_list)
    key_phrases = []

    for text in text_list:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # Ranges from -1 (negative) to 1 (positive)

        if polarity > 0:
            positive += 1
        elif polarity < 0:
            negative += 1
        else:
            neutral += 1

        # Extract key phrases (noun phrases)
        try:
            key_phrases.extend(blob.noun_phrases)
        except Exception as e:
            print(f"Warning: Could not extract noun phrases due to {str(e)}. Skipping...")

    # Calculate percentages
    if total > 0:
        pos_percent = (positive / total) * 100
        neg_percent = (negative / total) * 100
        neu_percent = (neutral / total) * 100
    else:
        pos_percent = neg_percent = neu_percent = 0

    # Get most common key phrases (if extracted)
    phrase_counts = Counter(key_phrases).most_common(3)
    top_phrases = [phrase for phrase, count in phrase_counts] if key_phrases else ["N/A"]

    return {
        "positive": round(pos_percent, 2),
        "negative": round(neg_percent, 2),
        "neutral": round(neu_percent, 2),
        "key_phrases": top_phrases,
        "total_articles": total
    }

def main():
    # Download NLTK data if not present
    download_nltk_data()

    # Get user input
    query = input("Enter a product, brand, or topic to analyze (e.g., iPhone 14): ").strip()
    if not query:
        print("Please enter a valid query.")
        return

    print(f"\nScraping data for '{query}' from Indian news channels...\n")

    # Scrape data from each news site
    all_content = []
    for site_name, url in NEWS_SITES.items():
        print(f"Fetching from {site_name}...")
        content = scrape_news_site(url, query)
        all_content.extend(content)
        for line in content[:2]:  # Show first 2 snippets for brevity
            print(f"  - {line[:100]}...")

    # Analyze sentiment
    print("\nAnalyzing sentiment...")
    result = analyze_sentiment(all_content)

    # Display results
    print("\nSentiment Analysis Results:")
    print(f"Total Articles/Sections Found: {result['total_articles']}")
    print(f"Positive: {result['positive']}%")
    print(f"Negative: {result['negative']}%")
    print(f"Neutral: {result['neutral']}%")
    print("Key Phrases:", ", ".join(result['key_phrases']))

if __name__ == "__main__":
    main()