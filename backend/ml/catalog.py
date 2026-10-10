"""Dropdown choices for the meal builder. Foods and portions are the most common entries in the
dataset's dietary logs (plus sweets for what-if checks); medication text matches the dataset's
format so normalize.py parses names and doses exactly as it did in training."""

FOODS = [
    ("Rice", "g", [50, 75, 80, 100, 150, 200]),
    ("Coarse grain", "g", [50, 100, 133, 150]),
    ("Coarse grain steamed bread", "g", [50, 100, 140]),
    ("Steamed bun", "g", [50, 60, 100, 150, 200]),
    ("Steamed pork bun", "g", [60, 100, 120]),
    ("Steamed vegetable bun", "g", [40, 60, 100, 150]),
    ("Bread", "g", [25, 40, 50, 100]),
    ("Buckwheat bread", "g", [15, 20, 50, 80]),
    ("Noodles", "g", [50, 60, 100, 150, 200]),
    ("Buckwheat noodles", "g", [50, 100, 150, 200]),
    ("Wonton", "g", [50, 100, 120, 300]),
    ("Cereal", "g", [20, 40, 50, 100]),
    ("Corn", "g", [50, 100, 150, 200]),
    ("Sweet potato", "g", [30, 60, 100, 150]),
    ("Taro", "g", [40, 50, 75, 100]),
    ("Egg", "g", [50, 55, 60, 100]),
    ("Boiled egg", "g", [40, 50, 60]),
    ("Fish", "g", [40, 50, 80, 100]),
    ("Shrimp", "g", [30, 40, 50, 100]),
    ("Beef", "g", [15, 20, 50, 100]),
    ("Pork", "g", [30, 50, 70, 100]),
    ("Chicken", "g", [20, 40, 50, 100]),
    ("Meatball", "g", [30, 40, 50]),
    ("Vegetable", "g", [50, 100, 150, 200]),
    ("Boiled vegetable", "g", [50, 100, 150]),
    ("Cucumber", "g", [50, 80, 100, 200]),
    ("Cabbage", "g", [50, 100, 150]),
    ("Spinach", "g", [50, 100, 150]),
    ("Lettuce", "g", [50, 60, 100]),
    ("Tomato", "g", [50, 100, 200, 250]),
    ("Apple", "g", [50, 125, 150, 200]),
    ("Orange", "g", [50, 100, 150, 200]),
    ("Banana", "g", [50, 100, 150]),
    ("Milk", "ml", [100, 200, 250]),
    ("Soybean milk", "ml", [250, 290, 390]),
    ("Yogurt", "g", [100, 135, 200]),
    ("Coffee", "ml", [70, 100, 200]),
    ("Beer", "ml", [330, 500, 650]),
    ("Biscuit", "g", [10, 15, 20, 40]),
    ("Cake", "g", [50, 100, 150]),
    ("Cookie", "g", [20, 40, 60]),
    ("Candy", "g", [10, 20, 40]),
    ("Sugar", "g", [5, 10, 20, 50]),
    ("Honey", "g", [10, 20, 30]),
    ("Orange juice", "ml", [200, 250, 330, 500]),
]

INSULINS = ["Insulin aspart", "Insulin aspart 30/70", "Insulin degludec", "Insulin detemir", "Insulin glargine",
            "Insulin glulisine", "Novolin R", "Novolin 30R", "Humulin 70/30", "Gansulin R"]
INSULIN_DOSES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24]

TABLETS = [("Metformin", [250, 500, 850, 1000]), ("Acarbose", [25, 50, 100]),
           ("Voglibose", [0.2, 0.3]), ("Sitagliptin", [50, 100]), ("Gliclazide", [30, 60, 80]),
           ("Repaglinide", [0.5, 1, 2]), ("Glimepiride", [1, 2, 4]), ("Pioglitazone", [15, 30]),
           ("Dapagliflozin", [5, 10]), ("Empagliflozin", [10, 25])]


def catalog():
    return {"foods": [{"name": n, "unit": u, "portions": p} for n, u, p in FOODS],
            "insulins": [{"name": n, "doses_iu": INSULIN_DOSES} for n in INSULINS],
            "tablets": [{"name": n, "doses_mg": d} for n, d in TABLETS]}


def insulin_text(name, dose):
    return f"{name}, {dose:g} IU"


def tablet_text(name, dose):
    return f"{name.lower()} {dose:g} mg"
