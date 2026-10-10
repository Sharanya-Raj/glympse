"""Photo/typed food -> USDA FoodData Central nutrients and standard portions.

Self-contained feature: a local image classifier (nateraw/food) names the food,
USDA FDC supplies per-100 g nutrients and household portions, and the meal is handed
to the existing glucose forecaster as the same "<food> <grams> g" text it already reads.
No LLM/generative calls anywhere in this package.
"""
