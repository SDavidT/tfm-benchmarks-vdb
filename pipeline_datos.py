import numpy as np
import h5py
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Hiperparámetros de la tesis
MODELO = 'all-MiniLM-L6-v2'
DIM = 384
NUM_TRAIN = 20000
NUM_TEST = 200
TOP_K = 100

def ejecutar_pipeline():
    print("Etapa 1: Ingesta de datos (Streaming ag_news)...")
    dataset = load_dataset("ag_news", split="train", streaming=True)
    
    textos = []
    for row in dataset:
        textos.append(row['text'])
        if len(textos) == (NUM_TRAIN + NUM_TEST):
            break
            
    textos_train = textos[:NUM_TRAIN]
    textos_test = textos[NUM_TRAIN:]

    print("Etapa 2: Vectorización (Batch = 128)...")
    modelo = SentenceTransformer(MODELO)
    emb_train_crudos = modelo.encode(textos_train, batch_size=128, show_progress_bar=True)
    emb_test_crudos = modelo.encode(textos_test, batch_size=128, show_progress_bar=True)

    print("Aplicando Normalización L2...")
    emb_train = emb_train_crudos / np.linalg.norm(emb_train_crudos, axis=1, keepdims=True)
    emb_test = emb_test_crudos / np.linalg.norm(emb_test_crudos, axis=1, keepdims=True)

    print("Etapa 3: Cálculo del Ground Truth (Fuerza Bruta Exacta)...")
    # Al estar normalizados L2, el producto escalar equivale a la similitud de coseno
    matriz_similitud = np.dot(emb_test, emb_train.T)
    # Obtenemos los 100 índices más cercanos reales
    ground_truth = np.argsort(matriz_similitud, axis=1)[:, -TOP_K:][:, ::-1]

    print("Etapa 4: Persistencia en formato jerárquico HDF5...")
    with h5py.File("dataset_tfm_real.hdf5", "w") as f:
        f.create_dataset("train", data=emb_train)
        f.create_dataset("test", data=emb_test)
        f.create_dataset("ground_truth", data=ground_truth)
        
    print("Pipeline completado exitosamente.")

if __name__ == "__main__":
    ejecutar_pipeline()