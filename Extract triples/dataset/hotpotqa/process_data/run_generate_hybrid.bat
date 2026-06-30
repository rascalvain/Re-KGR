@echo off
REM 生成混合嵌入 (TransE + SentenceTransformer)

echo ========================================
echo 生成混合嵌入
echo ========================================
echo.

REM 基础运行（使用 triples.txt 中的所有实体和关系，使用本地sentence-bert模型）
python generate_hybrid_embeddings.py ^
    --kb_entity_file entity2id.txt ^
    --kb_relation_file relation2id.txt ^
    --transe_dir ./output ^
    --transe_format pkl ^
    --triple_file triples.txt ^
    --sentence_model ../../../sentence-bert ^
    --batch_size 32 ^
    --output_dir ./hybrid_embeddings

echo.
echo ========================================
echo 完成！
echo ========================================
echo.
echo 生成的文件：
echo   - ./hybrid_embeddings/entity_hybrid_embeddings.pkl
echo   - ./hybrid_embeddings/relation_hybrid_embeddings.pkl
echo   - ./hybrid_embeddings/entity_hybrid_embeddings.npy
echo   - ./hybrid_embeddings/relation_hybrid_embeddings.npy
echo   - ./hybrid_embeddings/entity2idx.txt
echo   - ./hybrid_embeddings/relation2idx.txt
echo.

pause

