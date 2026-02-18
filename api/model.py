from py.embedding_model import extract_features

def get_embedding(img_path: str):
    vector = extract_features(img_path)
    return vector.tolist()  #ส่่งกลับมานี่ แล้วก็ออกไปนุ่นapp.py