import h5py
import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, SearchParams

def calcular_recall(resultados_ann, ground_truth_exacto, k=10):
    hits = 0
    for ann, exact in zip(resultados_ann, ground_truth_exacto):
        if ann in exact[:k]:
            hits += 1
    return (hits / len(ground_truth_exacto)) * 100

print("--- Benchmarking Técnico: Qdrant (Motor en Rust) ---")

# 1. Cargar datos del HDF5
with h5py.File("dataset_tfm_real.hdf5", "r") as f:
    train_data = f["train"][:]
    test_data = f["test"][:]
    ground_truth = f["ground_truth"][:]

# 2. Conectar e indexar
client = QdrantClient("localhost", port=6333)
nombre_coleccion = "tfm_qdrant"

print("\n[1/3] Creando colección con configuración indexación HNSW...")
client.recreate_collection(
    collection_name=nombre_coleccion,
    vectors_config=VectorParams(size=384, distance=Distance.DOT),
    hnsw_config=HnswConfigDiff(m=16, ef_construct=100)
)

print("[2/3] Iniciando ingesta masiva de 20,000 vectores...")
start_time = time.perf_counter()
client.upload_collection(
    collection_name=nombre_coleccion,
    vectors=train_data,
    payload=[{"id": i} for i in range(len(train_data))],
    batch_size=1000
)
index_time = time.perf_counter() - start_time
print(f"[OK] Ingesta finalizada. Tiempo de indexación: {index_time:.2f} segundos.")

# 3. Pruebas de Estrés
print("\n[3/3] Ejecutando pruebas de estrés con vectores de consulta (Test)...")
latencias = []
resultados_ids = []

for vector in test_data:
    t0 = time.perf_counter()
    search_result = client.search(
        collection_name=nombre_coleccion,
        query_vector=vector.tolist(),
        limit=10,
        search_params=SearchParams(hnsw_ef=128) # Sweet spot
    )
    t1 = time.perf_counter()
    latencias.append((t1 - t0) * 1000)
    resultados_ids.append(search_result[0].id)

latencia_media = np.mean(latencias)
p95 = np.percentile(latencias, 95)
p99 = np.percentile(latencias, 99)
recall = calcular_recall(resultados_ids, ground_truth)

print("\n================ RESULTADOS FINALES QDRANT ================")
print(f"Tiempo total de construcción del índice: {index_time:.2f} segundos")
print(f"Latencia Media: {latencia_media:.2f} ms")
print(f"Latencia Crítica p95: {p95:.2f} ms")
print(f"Latencia Crítica p99: {p99:.2f} ms")
print(f"Precisión del sistema (Recall@10): {recall:.2f}%")
print("===========================================================")