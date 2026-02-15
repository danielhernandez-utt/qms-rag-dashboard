import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.metrics.pairwise import cosine_similarity


def explain_similarity(
    question,
    document,
    embedding_model,
    num_features=8,
    num_samples=300,          
    batch_size=32
):
    """
    Optimized LIME explanation for RAG similarity.
    Faster, batched, and CPU-friendly.
    """

    try:
        # ===== 1️⃣ Compute question embedding ONCE =====
        question_embedding = embedding_model.encode(
            [question],
            convert_to_numpy=True
        )

        # ===== 2️⃣ Prediction function (batched) =====
        def similarity_predict(texts):

            doc_embeddings = embedding_model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True
            )

            similarities = cosine_similarity(
                doc_embeddings,
                question_embedding
            )

            similarities = np.clip(similarities, 0, 1)

            probs = np.hstack([
                1 - similarities,
                similarities
            ])

            return probs

        # ===== 3️⃣ LIME explainer =====
        explainer = LimeTextExplainer(
            class_names=["Not Relevant", "Relevant"]
        )

        explanation = explainer.explain_instance(
            document,
            similarity_predict,
            num_features=num_features,
            num_samples=num_samples   # 🔥 Massive speed boost
        )

        return explanation

    except Exception as e:
        print("LIME ERROR:", e)
        return None
