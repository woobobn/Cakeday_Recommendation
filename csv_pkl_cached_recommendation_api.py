import os
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# 경로 설정
csv_path = "cakes.csv"
pickle_path = "cake_vectors.pkl"

# CSV에서 벡터 생성 및 피클 저장 (없을 경우에만)
if os.path.exists(pickle_path):
    with open(pickle_path, "rb") as f:
        df, mlb, tag_vectors, cosine_sim_matrix = pickle.load(f)
else:
    df = pd.read_csv(csv_path)
    df["tags"] = df[["sheet", "filling", "size", "type"]].values.tolist()
    mlb = MultiLabelBinarizer()
    tag_vectors = mlb.fit_transform(df["tags"])
    cosine_sim_matrix = cosine_similarity(tag_vectors)
    with open(pickle_path, "wb") as f:
        pickle.dump((df, mlb, tag_vectors, cosine_sim_matrix), f)

# 요청 모델 정의
class VariantRequest(BaseModel):
    selected_variant_ids: List[int]

@app.post("/recommend")
async def recommend_cakes(request: VariantRequest):
    selected_ids = request.selected_variant_ids
    if not selected_ids:
        return {"recommended_cakes": []}

    selected_indices = [df[df["variant_id"] == vid].index[0] for vid in selected_ids if vid in df["variant_id"].values]
    if not selected_indices:
        return {"recommended_cakes": []}

    avg_sim = cosine_sim_matrix[selected_indices].mean(axis=0)
    sim_scores = list(enumerate(avg_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # ✅ 변경: 전체 variant_id를 유사도 순으로 정렬하여 반환
    recommended_variant_ids = df.iloc[[i for i, _ in sim_scores]]["variant_id"].tolist()

    return {"recommended_cakes": recommended_variant_ids}
