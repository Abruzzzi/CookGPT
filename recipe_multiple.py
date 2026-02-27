import ast
import openai
import pandas as pd

# ========== CONFIGURATION ==========
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# ========== LOAD + CLEAN DATA ==========
print("📦 Loading and cleaning data...")

recipes_df = pd.read_csv("RAW_recipes.csv")
interactions_df = pd.read_csv("RAW_interactions.csv")
meal_type_df = pd.read_csv("RAW_recipes_with_meal_type.csv")[["id", "meal_type"]]

recipes_df = recipes_df.merge(meal_type_df, on="id", how="left")
recipes_df["parsed_ingredients"] = recipes_df["ingredients"].apply(ast.literal_eval).apply(
    lambda lst: [i.lower() for i in lst])
recipes_df["parsed_nutrition"] = recipes_df["nutrition"].apply(ast.literal_eval)
recipes_df["parsed_steps"] = recipes_df["steps"].apply(ast.literal_eval)

rating_map = interactions_df.groupby("recipe_id")["rating"].mean().apply(lambda x: x / 5.0)
recipes_df["rating_score"] = recipes_df["id"].map(rating_map).fillna(0)

all_reviews = interactions_df.groupby("recipe_id")["review"].apply(list)
recipes_df["all_reviews"] = recipes_df["id"].map(all_reviews).apply(lambda x: x if isinstance(x, list) else [])

# ========== LOAD FOOD ALLERGEN DATA ==========
food_allergens_df = pd.read_csv('food_allergens.csv')

# ========== USER INPUT ==========
user_ingredients = input("🥕 What ingredients do you have? (comma-separated)\n> ").strip().lower().split(',')
user_ingredients = [i.strip() for i in user_ingredients]
user_goal = input(
    "🍽️ What kind of meal are you looking to prepare? (You can specify multiple, e.g., 'dinner with beef and dessert with banana, low sugar')\n> ").strip()
user_allergy_input = input(
    "⚠️ Any food allergies I should keep in mind? (comma-separated list, e.g., milk, eggs, peanuts)\n> ").strip().lower().split(
    ',')
user_allergy_input = [i.strip() for i in user_allergy_input]


# ========== GPT ALLERGY CLASSIFICATION ==========
def extract_allergy_classes(user_input):
    prompt = f"""
You are an allergy classifier. The user will list food allergies (possibly in natural language). Your job is to extract all relevant allergen classes from the following list:

[Milk, Eggs, Peanuts, Soy, Wheat, Tree Nuts, Fish, Shellfish]

Return a Python list of all applicable allergen classes (using only these names, and only if there is a clear match). Ignore anything that isn't clearly one of these eight. Do not include any other explanation or text. Example: if the user input mentions "almonds", include "Tree Nuts". If they say "seafood", include both "Fish" and "Shellfish".

User input: "{user_input}"

Your answer must be a valid Python list of strings, e.g. ["Milk", "Eggs", "Peanuts"]
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response['choices'][0]['message']['content'].strip()
    # Sanitize and parse as Python list
    if content.startswith("```"):
        content = content.strip("`").strip("python").strip()
    return ast.literal_eval(content)


allergy_classes = extract_allergy_classes(", ".join(user_allergy_input))
print(f"✅ Detected allergies: {allergy_classes}")


# ========== FILTER RECIPES BASED ON MULTIPLE ALLERGENS FROM CSV ==========
def filter_recipes_by_allergens(recipes_df, allergy_classes, food_allergens_df):
    allergens_to_check = []

    for allergy_class in allergy_classes:
        allergens_to_check.extend(
            [a.strip().lower() for a in food_allergens_df[allergy_class].dropna().tolist()]
        )

    allergens_to_check = list(set(a for a in allergens_to_check if a and a != "ph"))

    def check_ingredients(ingredients):
        for allergen in allergens_to_check:
            for ing in ingredients:
                if allergen in ing:
                    return False
        return True

    filtered_recipes = recipes_df[recipes_df["parsed_ingredients"].apply(check_ingredients)]
    return filtered_recipes


recipes_df = filter_recipes_by_allergens(recipes_df, allergy_classes, food_allergens_df)


# ========== GPT PREFERENCE EXTRACTION (MULTI-MEAL-TYPE) ==========
def extract_multi_meal_preferences(user_input):
    prompt = f"""
You are a meal preference extractor. Interpret natural language and convert it into a structured Python list of dictionaries, with **one dictionary per requested meal type**.

For each meal type the user requests, create a dictionary with:
- "meal_type": one of ["breakfast", "dinner", "dessert", "drink", "other"]
- "diet": list of dietary goals, choose from: ["vegetarian", "gluten-free", "dairy-free", "low fat", "low sugar", "low sodium", "low calorie", "high protein"]
- "time": "quick" if user wants something fast, else "null"
- "keywords": other descriptive terms (e.g. easy, spicy, creamy)
- "ingredients": a list of specific foods, ingredients, or dish names the user wants to appear in the recipe name (e.g. ["banana", "strawberry", "chocolate"]). If none, use an empty list [].

If a dietary preference or keyword applies only to a specific meal, attach it to the appropriate dictionary. If it applies to all, repeat as needed.

Examples:
User: "I want a healthy dinner with beef and a dessert with banana, low sugar"
Output:
[
  {{ "meal_type": "dinner", "diet": ["healthy"], "time": "null", "keywords": [], "ingredients": ["beef"] }},
  {{ "meal_type": "dessert", "diet": ["low sugar"], "time": "null", "keywords": [], "ingredients": ["banana"] }}
]

User: "breakfast with eggs, easy and dinner with pork"
Output:
[
  {{ "meal_type": "breakfast", "diet": [], "time": "null", "keywords": ["easy"], "ingredients": ["eggs"] }},
  {{ "meal_type": "dinner", "diet": [], "time": "null", "keywords": [], "ingredients": ["pork"] }}
]

User: "vegetarian lunch and a low sugar dessert"
Output:
[
  {{ "meal_type": "lunch", "diet": ["vegetarian"], "time": "null", "keywords": [], "ingredients": [] }},
  {{ "meal_type": "dessert", "diet": ["low sugar"], "time": "null", "keywords": [], "ingredients": [] }}
]

User: "dessert with banana and chocolate, quick and easy"
Output:
[
  {{ "meal_type": "dessert", "diet": [], "time": "quick", "keywords": ["easy"], "ingredients": ["banana", "chocolate"] }}
]

User: "dinner"
Output:
[
  {{ "meal_type": "dinner", "diet": [], "time": "null", "keywords": [], "ingredients": [] }}
]

User said: "{user_input}"

Your response must be a valid Python list of dictionaries, with one dictionary per meal type as in the examples above.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response['choices'][0]['message']['content'].strip()
    if content.startswith("```"):
        content = content.strip("`").strip("python").strip()
    return ast.literal_eval(content)

# Extract meal preferences for each meal type requested
multi_preferences = extract_multi_meal_preferences(user_goal)
print("\n✅ Extracted preferences:")
for i, pref in enumerate(multi_preferences):
    print(f"  [{i + 1}] {pref}")


# ========== SCORING FUNCTIONS ==========
def compute_ingredient_score(user_ingredients, recipe_ingredients):
    return sum(any(user_ing in recipe_ing for recipe_ing in recipe_ingredients) for user_ing in user_ingredients) / len(
        recipe_ingredients) if recipe_ingredients else 0


def compute_nutrition_score(nutrition_list, diet_prefs):
    if not isinstance(nutrition_list, list) or len(nutrition_list) != 7:
        return 0
    match_count = 0
    rules = {
        "low fat": nutrition_list[1] <= 20,
        "low sugar": nutrition_list[2] <= 10,
        "low sodium": nutrition_list[3] <= 15,
        "high protein": nutrition_list[4] >= 15,
        "low calorie": nutrition_list[0] <= 300,
        "low carb": nutrition_list[6] <= 20
    }
    for pref in diet_prefs:
        if pref in rules and rules[pref]:
            match_count += 1
    return match_count / max(1, len(diet_prefs)) if diet_prefs else 0


def compute_time_score(minutes, pref_time):
    return 1.0 if pref_time == "null" or (pref_time == "quick" and minutes <= 30) else 0.0


def compute_easy_score(n_steps, keywords):
    return 1.0 if "easy" in keywords and n_steps <= 10 else 0.0

# ========== GPT KEYWORD EXTRACTION ==========
def extract_keywords_from_reviews(row):
    joined_reviews = "\n".join(row["reviews"]) if isinstance(row["reviews"], list) else ""
    prompt = f"""
Analyze the following reviews and extract the 3 most meaningful recurring words or short phrases that describe the recipe’s qualities, experience, or style. Focus especially on adjectives, adverbs, and descriptive phrases (such as "easy," "quick," "delicious," "family style," "kid friendly," "healthy," "comfort food," etc.). Avoid generic nouns (like "recipe," "dish," "meal") and exclude filler words.

Reviews:
{joined_reviews}

Only output a comma-separated list of the 3 most relevant descriptive words or phrases.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"⚠️ GPT error on review analysis: {e}")
        return "unknown, unknown, unknown"


# ========== SHOPPING LIST FUNCTION ==========
def gpt_shopping_list(user_ingredients, recipe_ingredients, recipe_name):
    prompt = f"""
You are a helpful cooking assistant. The user has these ingredients in their fridge:
{user_ingredients}

The recipe '{recipe_name}' requires these ingredients:
{recipe_ingredients}

For each required ingredient, if it can be reasonably substituted or covered by something the user already has, do NOT include it in the shopping list (for example, 'bananas' is covered by 'banana', and 'brown sugar' is covered by 'sugar'). Only output the ingredients the user definitely still needs to buy.

Return the shopping list as a simple Python list of ingredient names (do not include any explanation or formatting).

Shopping list:
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response['choices'][0]['message']['content'].strip()
    if content.startswith("```"):
        content = content.strip("`").strip("python").strip()
    try:
        shopping_list = ast.literal_eval(content)
        if not isinstance(shopping_list, list):
            return []
        return [str(i) for i in shopping_list]
    except Exception:
        return []


# ========== MAIN LOOP OVER MEAL TYPES ==========
for meal_pref in multi_preferences:
    meal_type = meal_pref.get("meal_type", "null")
    meal_keywords = meal_pref.get("ingredients", [])
    if isinstance(meal_keywords, str):
        meal_keywords = [k.strip().lower() for k in meal_keywords.split() if k.strip()]
    else:
        meal_keywords = [k.strip().lower() for k in meal_keywords if k.strip()]
    diet_prefs = meal_pref.get("diet", [])
    pref_time = meal_pref.get("time", "null")
    pref_keywords = meal_pref.get("keywords", [])

    # Filter recipes for this meal type
    filtered_df = recipes_df
    if meal_keywords:
        filtered_df = filtered_df[filtered_df["name"].apply(
            lambda name: isinstance(name, str) and any(kw in name.lower() for kw in meal_keywords)
        )]
    if meal_type != "null":
        filtered_df = filtered_df[filtered_df["meal_type"] == meal_type]
    if len(filtered_df) == 0:
        print(f"\n😢 Sorry, we couldn't find any recipes matching your request for '{meal_type}'.")
        continue

    # Score and rank
    filtered_scores = []
    for _, row in filtered_df.iterrows():
        ing_score = compute_ingredient_score(user_ingredients, row["parsed_ingredients"])
        nutri_score = compute_nutrition_score(row["parsed_nutrition"], diet_prefs)
        time_score = compute_time_score(row["minutes"], pref_time)
        rating_score = row["rating_score"]
        easy_score = compute_easy_score(row["n_steps"], pref_keywords)

        final_score = 0.3 * ing_score + 0.2 * nutri_score + 0.2 * time_score + 0.2 * rating_score + 0.1 * easy_score

        if ing_score > 0:
            filtered_scores.append({
                "id": row["id"],
                "name": row["name"],
                "minutes": row["minutes"],
                "ingredients": row["parsed_ingredients"],
                "steps": row["parsed_steps"],
                "nutrition": row["parsed_nutrition"],
                "reviews": row["all_reviews"],
                "final_score": final_score
            })
    if not filtered_scores:
        print(f"\n😢 No recipes found for meal type '{meal_type}' after scoring/filtering.")
        continue

    top_20 = pd.DataFrame(sorted(filtered_scores, key=lambda x: x["final_score"], reverse=True)[:20])
    # Extract keywords for top 20
    print(f"\n📊 Top Recipe Recommendations for '{meal_type.capitalize()}':")
    top_20["keywords"] = top_20.apply(extract_keywords_from_reviews, axis=1)
    for i, row in top_20.iterrows():
        print(f"\n[{i}] 🍽️ {row['name']}")
        print(f"⏱️  Time: {row['minutes']} min")
        print(f"🥣 Ingredients: {', '.join(row['ingredients'][:5])}...")
        print(f"🔑 Keywords from reviews: {row['keywords']}")

    # Recipe selection for this meal type
    choice = input(
        f"\n👉 Which recipe(s) for '{meal_type}' are you most interested in? (comma-separated indices)\n> ").strip()
    indices = [int(i.strip()) for i in choice.split(",") if i.strip().isdigit()]

    for i in indices:
        row = top_20.iloc[i]
        print(f"\n🍽️ {row['name']}")
        print(f"⏱️  Time: {row['minutes']} min")

        # Nutrition with labels
        nutrition_labels = [
            "calories (#)",
            "total fat (PDV)",
            "sugar (PDV)",
            "sodium (PDV)",
            "protein (PDV)",
            "saturated fat (PDV)",
            "total carbohydrate (PDV)"
        ]
        nutrition_values = row['nutrition']
        if isinstance(nutrition_values, list) and len(nutrition_values) == 7:
            nutrition_string = ", ".join(
                f"{nutrition_labels[i]}: {nutrition_values[i]}{'%' if i != 0 else ''}"
                for i in range(7)
            )
        else:
            nutrition_string = str(nutrition_values)
        print(f"🥗 Nutrition: {nutrition_string}")

        print(f"🥣 Ingredients: {', '.join(row['ingredients'])}")
        print("🪜 Steps:")
        for step in row["steps"]:
            print(f"  - {step}")

    # Shopping list for this meal type
    want_list = input(
        f"\n🛒 Would you like a shopping list for your selected recipes? (yes/no)\n> ").strip().lower()
    if want_list in ['yes', 'y']:
        for i in indices:
            row = top_20.iloc[i]
            shopping_list = gpt_shopping_list(
                user_ingredients, row['ingredients'], row['name']
            )
            print(f"\n🛍️ Shopping List for '{row['name']}':")
            if shopping_list:
                for item in shopping_list:
                    print(f"- {item}")
            else:
                print("🎉 You already have all the ingredients for this recipe!")
