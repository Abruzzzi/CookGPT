import pandas as pd
# Create the data dictionary as per the user's request
data = {
    "Milk": [
        "Butter", "Cheddar", "Mozzarella", "Yogurt", "Cream", "Ice cream",
        "Whey", "Casein", "Ghee", "Custard", "Milk", "Pudding",
        "Buttermilk", "Lassi", "Ricotta cheese"
    ],
    "Eggs": [
        "Egg", "Eggs", "Mayonnaise", "Hollandaise sauce", "Meringue", "Quiche", "Omelette", "Cake", "Pancakes", "Soufflés"
    ],
    "Peanuts": [
        "Peanut", "Peanuts"
    ],
    "Soy": [
        "Soy", "Tofu", "Edamame", "Tempeh", "Miso","Soybean"
    ],
    "Wheat": [
        "Bread", "Pasta", "Cake", "Pizza dough", "Crackers", "Pastries", "Flour",
        "Cookies", "Muffins", "Donuts", "Cereal", "Tortillas", "Biscuits", "Pancakes", "Toast"
    ],
    "Tree Nuts": [
        "Nut", "Nuts", "Almonds", "Walnuts", "Pecans", "Cashews", "Hazelnuts", "Pistachios",
        "Almond milk"
    ],
    "Fish": [
        "Fish", "Salmon", "Tuna", "Trout", "Cod", "Sardines", "Mackerel", "Anchovies", "Herring", "Halibut",
        "Caviar", "Sushi"
    ],
    "Shellfish": [
        "Shellfish", "Shrimp", "Lobster", "Crab", "Mussels", "Clams", "Oysters", "Scallops", "Squid", "Calamari",
        "Prawns", "Crawfish"
    ]
}

# Padding each list to ensure they have 15 elements using "PH" as placeholder
max_length = 15
for allergen, ingredients in data.items():
    while len(ingredients) < max_length:
        ingredients.append("PH")

# Convert the data dictionary into a DataFrame
df_allergens = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_file_path = 'food_allergens.csv'
df_allergens.to_csv(csv_file_path, index=False)