from sentence_transformers import SentenceTransformer
from essential_generators import DocumentGenerator
import torch 

print(f"torch.cuda.is_available() --> {torch.cuda.is_available()}")
print(f"torch.cuda.device_count() --> {torch.cuda.device_count()}")

model_type = "all-distilroberta-v1"
model = SentenceTransformer(f'sentence-transformers/{model_type}')
n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
    sentence_model = model.module
else:
    sentence_model = model 

model.to(torch.device("cuda"))

gen = DocumentGenerator()
sentences = [gen.sentence() for _ in range(100000)]

print(sentences[:10])
predict_batch_size = 64


if n_gpu == 1:
    vectors = sentence_model.encode(
                sentences,
                batch_size=predict_batch_size,
                show_progress_bar=True,
                convert_to_tensor=False, # this will return a stacked tensor, we want default return (list of tensors)
                convert_to_numpy=False,
                normalize_embeddings=True)
    print(len(vectors))
    result = [vector.detach().cpu().numpy() for vector in vectors]
    print(len(result))
elif n_gpu >= 2:
    pool = sentence_model.start_multi_process_pool()
    vectors = sentence_model.encode_multi_process(
                sentences=sentences,
                pool=pool,
                batch_size=predict_batch_size,
                )

                # convert_to_tensor=False, # this will return a stacked tensor, we want default return (list of tensors)
                # convert_to_numpy=False,
                # normalize_embeddings=True)
    sentence_model.stop_multi_process_pool(pool)
    print(len(vectors))