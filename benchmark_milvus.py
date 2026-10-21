import h5py
import time
import numpy as np
import warnings
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# Ignorar los warnings de depreciación para limpiar la salida
warnings.filterwarnings("ignore")

def calcular_recall(resultados_ann, ground_truth_exacto, k=10):
    hits = 0
    for ann, exact in zip(resultados_ann, ground_truth_exacto):
        if ann in exact[:k]:
            hits += 1
    return (hits / len(ground_truth_exacto)) * 100

print("--- Benchmarking Técnico: Milvus Standalone (Motor en Go/C++) ---")

# 1. Cargar datos del HDF5
with h5py.File("dataset_tfm_real.hdf5", "r") as f:
    train_data = f["train"][:]
    test_data = f["test"][:]
    ground_truth = f["ground_truth"][:]

# 2. Conexión y Esquema
connections.connect("default", host="localhost", port="19530")
nombre_coleccion = "tfm_milvus"
if utility.has_collection(nombre_coleccion):
    utility.drop_collection(nombre_coleccion)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)
]
schema = CollectionSchema(fields, "TFM Benchmark Schema")
coleccion = Collection(name=nombre_coleccion, schema=schema)

# 3. Ingesta (Persistencia en MinIO)
print("\n[1/3] Iniciando ingesta masiva de 20,000 vectores...")
ids = [i for i in range(len(train_data))]
t_inicio_ingesta = time.perf_counter()
coleccion.insert([ids, train_data])
coleccion.flush()
tiempo_escritura = time.perf_counter() - t_inicio_ingesta
print(f"[OK] Ingesta finalizada. Tiempo de escritura en disco: {tiempo_escritura:.2f} segundos.")

# 4. Construcción HNSW en Knowhere
print("\n[2/3] Construyendo el índice HNSW en el servidor...")
config_indice = {"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 100}}
t_inicio_idx = time.perf_counter()
coleccion.create_index(field_name="vector", index_params=config_indice)
tiempo_idx = time.perf_counter() - t_inicio_idx
print(f"[OK] Índice HNSW construido en: {tiempo_idx:.2f} segundos.")

print("Cargando colección en memoria RAM de la VM...")
coleccion.load()

# 5. Pruebas de Estrés
print("\n[3/3] Ejecutando pruebas de estrés con vectores de consulta (Test)...")
search_params = {"metric_type": "IP", "params": {"ef": 64}} # Sweet spot param

latencias = []
resultados_ids = []

for vector in test_data:
    t0 = time.perf_counter()
    resultados = coleccion.search(
        data=[vector.tolist()], anns_field="vector", param=search_params, limit=10
    )
    t1 = time.perf_counter()
    latencias.append((t1 - t0) * 1000)
    resultados_ids.append(resultados[0][0].id)

latencia_media = np.mean(latencias)
p95 = np.percentile(latencias, 95)
p99 = np.percentile(latencias, 99)
recall = calcular_recall(resultados_ids, ground_truth)

print("\n================ RESULTADOS FINALES MILVUS ================")
print(f"Tiempo de Ingesta (Escritura): {tiempo_escritura:.2f} segundos")
print(f"Tiempo de Construcción del Índice: {tiempo_idx:.2f} segundos")
print(f"Latencia Media: {latencia_media:.2f} ms")
print(f"Latencia Crítica p95: {p95:.2f} ms")
print(f"Latencia Crítica p99: {p99:.2f} ms")
print(f"Precisión del sistema (Recall@10): {recall:.2f}%")
print("===========================================================")