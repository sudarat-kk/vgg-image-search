import sys
import json
from embedding_model import extract_features

if __name__ == "__main__":
    image_path = sys.argv[1]
    embedding = extract_features(image_path)

    # ส่งออกเป็น JSON
    print(json.dumps(embedding.tolist()))
