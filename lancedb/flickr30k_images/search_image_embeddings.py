import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry
import torch
import sys

# Check if search term was provided
if len(sys.argv) < 2:
    print("Usage: python search_image_embeddings.py 'your search term'")
    sys.exit(1)

# Get search term from command line argument
search_term = sys.argv[1]

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
registry = EmbeddingFunctionRegistry.get_instance()
clip = registry.get("open-clip").create(device=device)

class Images(LanceModel):
    vector: Vector(clip.ndims()) = clip.VectorField()
    image_uri: str = clip.SourceField()

# Connect and search
db = lancedb.connect("./images.lancedb")
table = db["image_embeddings"]

# Perform search and print results
results = table.search(search_term).limit(10).to_pandas()
print(f"\nTop 10 results for '{search_term}':\n")
for i, row in enumerate(results['image_uri'], 1):
    print(f"{i}. {row}")
