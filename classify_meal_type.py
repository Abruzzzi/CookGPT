import pandas as pd
import openai
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================
# CONFIGURATION
# =============================
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_FILE = "RAW_recipes.csv"
OUTPUT_FILE = "RAW_recipes_with_meal_type.csv"

MODEL = "gpt-4o"
BATCH_SIZE = 10
MAX_WORKERS = 15
CHUNK_SIZE = 500
WAIT_BETWEEN_CHUNKS = 60
ADDITIONAL_WAIT_ON_ERROR = 10

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(INPUT_FILE)
df = df[["id", "name"]].dropna().reset_index(drop=True)
all_recipes = df.to_dict(orient="records")  # List of {"id": ..., "name": ...}

start_index = 0
if os.path.exists(OUTPUT_FILE):
    existing = pd.read_csv(OUTPUT_FILE)
    start_index = len(existing)
    print(f"📌 Resuming from index {start_index}...")

# =============================
# GPT CALL FUNCTION
# =============================
def classify_batch(index_batch_pair):
    batch_index, name_batch = index_batch_pair

    prompt = f"""
You are a meal-type classifier.

Classify each recipe title below into **exactly one** of the following categories:
- breakfast (e.g., eggs, muffins, pancakes, toast)
- dinner (a full/main course meal — includes protein/starch/veg)
- dessert (sweet dishes typically eaten after a meal)
- drink (smoothies, shakes, juices, cocktails, teas, etc.)
- other (side dishes, sauces, dips, dressings, toppings, or things not eaten alone)

⚠️ IMPORTANT:
- Only use "dinner" for full meals or clear main courses
- If it's unclear or sounds like a side dish, sauce, or light snack → use "other"
- Do NOT default to "dinner" unless it's clearly a main meal

Format:
- One label per line
- Use this format: 1. dinner, 2. other, etc.
- Keep the same order and numbering as the titles list

Examples:
- Creamed Spinach → other
- Buttermilk Baked Tomatoes → other
- Roast Chicken with Potatoes → dinner
- Blueberry Pancakes → breakfast
- Strawberry Milkshake → drink
- Chocolate Chip Cookies → dessert

Titles:
""" + "\n".join([f"{i + 1}. {title}" for i, title in enumerate(name_batch)])

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response["choices"][0]["message"]["content"].strip().lower()

        labels = []
        for i in range(1, len(name_batch) + 1):
            match = next((line for line in content.splitlines() if line.strip().startswith(f"{i}.")), None)
            if match:
                label = match.split(".", 1)[1].strip()
                labels.append(label)
            else:
                labels.append("unknown")

        return batch_index, labels

    except Exception as e:
        print(f"❌ Error on batch {batch_index[0]}–{batch_index[-1]}: {e}")
        raise e

# =============================
# MAIN LOOP
# =============================
while start_index < len(all_recipes):
    end_index = min(start_index + CHUNK_SIZE, len(all_recipes))
    print(f"\n🔁 Processing chunk {start_index} to {end_index - 1}...")

    chunk = all_recipes[start_index:end_index]
    batches = [chunk[i:i + BATCH_SIZE] for i in range(0, len(chunk), BATCH_SIZE)]
    batch_indices = [list(range(i + start_index, min(i + BATCH_SIZE + start_index, end_index))) for i in range(0, len(chunk), BATCH_SIZE)]

    results = [""] * len(chunk)

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i in range(len(batches)):
                index_batch_pair = (batch_indices[i], batches[i])
                futures.append(executor.submit(classify_batch, index_batch_pair))

            for count, future in enumerate(as_completed(futures), 1):
                batch_ids, labels = future.result()
                for idx, label in zip(batch_ids, labels):
                    results[idx - start_index] = label

        # Prepare DataFrame
        chunk_ids = [item["id"] for item in chunk]
        chunk_names = [item["name"] for item in chunk]
        df_out = pd.DataFrame({"id": chunk_ids, "name": chunk_names, "meal_type": results})

        # Append to CSV
        df_out.to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE) or start_index == 0, index=False)
        print(f"✅ Saved recipes {start_index}–{end_index - 1} to {OUTPUT_FILE}")

        start_index = end_index
        print(f"🕒 Waiting {WAIT_BETWEEN_CHUNKS}s before next chunk...")
        time.sleep(WAIT_BETWEEN_CHUNKS)

    except Exception:
        print(f"⚠️ Error occurred, waiting extra {ADDITIONAL_WAIT_ON_ERROR}s and retrying same chunk...")
        time.sleep(ADDITIONAL_WAIT_ON_ERROR)






