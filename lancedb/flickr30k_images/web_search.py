from flask import Flask, render_template, request, send_file
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry
import torch
from pathlib import Path
import os

app = Flask(__name__)

# Setup LanceDB and CLIP (same as before)
device = "cuda" if torch.cuda.is_available() else "cpu"
registry = EmbeddingFunctionRegistry.get_instance()
clip = registry.get("open-clip").create(device=device)

class Images(LanceModel):
    vector: Vector(clip.ndims()) = clip.VectorField()
    image_uri: str = clip.SourceField()

db = lancedb.connect("./images.lancedb")
table = db["image_embeddings"]

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        search_term = request.form['search']
        results = table.search(search_term).limit(20).to_pandas()
        # Convert full paths to relative paths for display
        results = [{'full_path': path, 
                   'display_path': os.path.basename(path)} 
                  for path in results['image_uri'].tolist()]
    return render_template('index.html', results=results)

@app.route('/image/<path:filename>')
def serve_image(filename):
    # Construct the full path to your images directory
    image_dir = "/home/syran/sandbox/ai/lancedb/flickr30k_images/flickr30k_images"
    return send_file(os.path.join(image_dir, filename))

# Update the template HTML
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Image Search</title>
    <style>
        body { max-width: 1200px; margin: 0 auto; padding: 20px; font-family: Arial, sans-serif; }
        .search-box { margin: 20px 0; text-align: center; }
        input[type="text"] { width: 300px; padding: 10px; }
        input[type="submit"] { padding: 10px 20px; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; }
        .image-container { position: relative; }
        img { width: 100%; height: 200px; object-fit: cover; }
        .image-path { font-size: 0.8em; word-break: break-all; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="search-box">
        <form method="POST">
            <input type="text" name="search" placeholder="Enter search term...">
            <input type="submit" value="Search">
        </form>
    </div>
    <div class="image-grid">
    {% for image in results %}
        <div class="image-container">
            <img src="{{ url_for('serve_image', filename=image.display_path) }}" alt="Result">
            <div class="image-path">{{ image.full_path }}</div>
        </div>
    {% endfor %}
    </div>
</body>
</html>
    ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
