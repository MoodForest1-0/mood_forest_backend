from deepface import DeepFace

image_path = "dataset/happy/IMG_0668.JPG"   # Change this to one of your images

embedding = DeepFace.represent(
    img_path=image_path,
    model_name="Facenet512",
    enforce_detection=False
)

print("Embedding length:", len(embedding[0]["embedding"]))
print("First 10 values:")
print(embedding[0]["embedding"][:10])