import lancedb
from pathlib import Path
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Connect to LanceDB
db = lancedb.connect("./images.lancedb")

# Get the CLIP embedding function
registry = EmbeddingFunctionRegistry.get_instance()
clip = registry.get("open-clip").create(
    device=device,
    batch_size=512  # or even try 1024
)

# Define the data model
class Images(LanceModel):
    vector: Vector(clip.ndims()) = clip.VectorField()
    image_uri: str = clip.SourceField()

# Create or get the table
table_name = "image_embeddings"
if table_name in db:
    table = db[table_name]
else:
    table = db.create_table(table_name, schema=Images)
    
    # Get all image paths (assuming jpg format - adjust as needed)
    image_dir = Path("/home/syran/sandbox/ai/lancedb/flickr30k_images/flickr30k_images")  # Replace with your image directory
    image_paths = [str(f) for f in image_dir.glob("*.jpg")]
    
    # Add images to the table - embeddings will be automatically generated
    table.add([{"image_uri": path} for path in image_paths]) 

