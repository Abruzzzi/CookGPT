# CookGPT

CookGPT is an AI-powered recipe recommendation system that generates personalized meal suggestions using natural language input.

## What It Does

- Accepts user ingredients, allergies, and meal preferences
- Classifies meal type using GPT
- Detects Big-8 food allergens
- Filters unsafe recipes
- Ranks recipes using a weighted scoring system
- Extracts descriptive keywords from reviews
- Generates intelligent shopping lists

The system combines LLM-based reasoning with deterministic filtering and multi-factor scoring.

## Tech Stack

- Python
- Pandas
- OpenAI GPT-4o API
- ThreadPoolExecutor (parallel processing)
- Data cleaning & feature engineering

## How to Run

1. Set your OpenAI API key:

   export OPENAI_API_KEY="your_api_key"

2. Install dependencies:

   pip install pandas openai

3. Run:

   python recipe_multiple.py

## Author
Rukai Gao
Erxing (Jason) Yang  
Washington University in St. Louis
