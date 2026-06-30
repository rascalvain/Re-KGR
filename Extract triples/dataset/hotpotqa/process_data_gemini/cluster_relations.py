import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import pickle

def load_relations(file_path):
    """Load relations from relation2id.txt"""
    relations = []
    relation2id = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                rel_name = parts[0]
                rel_id = int(parts[1])
                relations.append(rel_name)
                relation2id[rel_name] = rel_id
    
    return relations, relation2id

def cluster_relations(relations, n_clusters=2000, model_path='../../../sentence-bert'):
    """
    Cluster relations using sentence-transformers and KMeans
    """
    print(f"Loading sentence-transformer model from {model_path}...")
    model = SentenceTransformer(model_path)
    
    print(f"Computing embeddings for {len(relations)} relations...")
    embeddings = model.encode(relations, show_progress_bar=True, batch_size=32)
    
    print(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # For each cluster, find the relation closest to the centroid
    cluster_representatives = {}
    relation_to_cluster = {}
    
    print("Finding cluster representatives...")
    for cluster_id in tqdm(range(n_clusters)):
        # Get all relations in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        # Get embeddings for this cluster
        cluster_embeddings = embeddings[cluster_mask]
        centroid = kmeans.cluster_centers_[cluster_id]
        
        # Find the relation closest to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        representative = relations[closest_idx]
        
        cluster_representatives[cluster_id] = representative
        
        # Map all relations in this cluster to the representative
        for idx in cluster_indices:
            relation_to_cluster[relations[idx]] = representative
    
    return relation_to_cluster, cluster_representatives

def create_mapping_output(relation_to_cluster, output_path):
    """
    Create JSON mapping output in the format {"rel": ["a", "b", "c"]}
    Where each key is a cluster representative and value is list of relations mapping to it
    """
    mapping = {}
    
    for relation, representative in relation_to_cluster.items():
        if representative not in mapping:
            mapping[representative] = []
        if relation not in mapping[representative]:
            mapping[representative].append(relation)
    
    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    print(f"Mapping saved to {output_path}")
    print(f"Number of cluster representatives: {len(mapping)}")
    print(f"Total relations mapped: {sum(len(v) for v in mapping.values())}")
    
    return mapping

def create_new_relation2id(mapping, output_path):
    """
    Create a new relation2id.txt with only the cluster representatives
    """
    representatives = list(mapping.keys())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, rel in enumerate(sorted(representatives)):
            f.write(f"{rel}\t{idx}\n")
    
    print(f"New relation2id.txt saved to {output_path}")
    print(f"Reduced from original to {len(representatives)} relations")


def create_relation_mapping_txt(relation_to_cluster, output_path):
    """
    Create a txt file with relation to cluster mapping
    Format: original_relation \t cluster_representative
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # 按关系词排序输出
        for relation in sorted(relation_to_cluster.keys()):
            cluster_rep = relation_to_cluster[relation]
            f.write(f"{relation}\t{cluster_rep}\n")
    
    print(f"Relation mapping txt saved to {output_path}")
    print(f"Total relations mapped: {len(relation_to_cluster)}")

def main():
    import os
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_file = os.path.join(script_dir, 'relation2id.txt')
    output_mapping_file = os.path.join(script_dir, 'relation_cluster_mapping.json')
    output_relation2id_file = os.path.join(script_dir, 'relation2id_clustered.txt')
    output_mapping_txt_file = os.path.join(script_dir, 'relation_to_cluster_mapping.txt')
    
    print("=" * 60)
    print("Relation Clustering with Sentence-Transformers")
    print("=" * 60)
    
    # Load relations
    print(f"\nStep 1: Loading relations from {input_file}")
    relations, relation2id = load_relations(input_file)
    print(f"Loaded {len(relations)} relations")
    
    # Cluster relations
    print(f"\nStep 2: Clustering relations to 2000 clusters")
    relation_to_cluster, cluster_representatives = cluster_relations(relations, n_clusters=2000)
    
    # Create mapping output (JSON format)
    print(f"\nStep 3: Creating mapping output (JSON)")
    mapping = create_mapping_output(relation_to_cluster, output_mapping_file)
    
    # Create relation to cluster mapping (TXT format)
    print(f"\nStep 4: Creating relation to cluster mapping (TXT)")
    create_relation_mapping_txt(relation_to_cluster, output_mapping_txt_file)
    
    # Create new relation2id file
    print(f"\nStep 5: Creating new relation2id file")
    create_new_relation2id(mapping, output_relation2id_file)
    
    # Print some statistics
    print("\n" + "=" * 60)
    print("Statistics:")
    print("=" * 60)
    print(f"Original number of relations: {len(relations)}")
    print(f"Number of clusters: {len(mapping)}")
    print(f"Average relations per cluster: {sum(len(v) for v in mapping.values()) / len(mapping):.2f}")
    
    # Show some example clusters
    print("\nExample clusters (showing first 5):")
    for i, (rep, rels) in enumerate(list(mapping.items())[:5]):
        print(f"\nCluster {i+1} - Representative: '{rep}'")
        print(f"  Contains {len(rels)} relations:")
        for rel in rels[:5]:
            print(f"    - {rel}")
        if len(rels) > 5:
            print(f"    ... and {len(rels) - 5} more")
    
    print("\n" + "=" * 60)
    print("Done! Output files:")
    print("=" * 60)
    print(f"1. JSON mapping (cluster -> relations): {output_mapping_file}")
    print(f"2. TXT mapping (relation -> cluster):   {output_mapping_txt_file}")
    print(f"3. New relation2id file:                {output_relation2id_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()

