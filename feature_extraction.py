from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import AutoTokenizer, AutoModel, logging
import torch
from tqdm import tqdm
import warnings

# Filter out specific warnings for cleaner output
logging.set_verbosity_error()  # Only show errors from transformers
warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length is longer than the specified maximum",
)


def extract_targets(df):
    """Extract avg/min/max response times from dataframe."""
    return df[["avg_resp_time", "min_resp_time", "max_resp_time"]].to_numpy()


def build_tfidf_features(train_samples, val_samples, test_samples):
    """
    Builds TF-IDF feature matrices and target arrays from dataset samples.

    Args:
        train_samples: Training data samples (DataFrame)
        val_samples: Validation data samples (DataFrame)
        test_samples: Test data samples (DataFrame)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, vectorizer
    """
    # Extract DSL text
    train_texts = train_samples["tpcm_text"].tolist()
    val_texts = val_samples["tpcm_text"].tolist()
    test_texts = test_samples["tpcm_text"].tolist()

    # Create vectorizer and transform text
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    y_train = extract_targets(train_samples)
    y_val = extract_targets(val_samples)
    y_test = extract_targets(test_samples)

    return X_train, y_train, X_val, y_val, X_test, y_test, vectorizer


def build_bert_features(
    train_samples,
    val_samples,
    test_samples,
    model_name="microsoft/codebert-base",
    device="cpu",
    batch_size=32,  # Increased for RTX 3090
):
    """
    Builds BERT embeddings and target arrays from dataset samples.
    Optimized for RTX 3090 GPU.

    Args:
        train_samples: Training data samples (DataFrame)
        val_samples: Validation data samples (DataFrame)
        test_samples: Test data samples (DataFrame)
        model_name: Name of the pretrained model to use
        device: Device to run inference on ('cpu' or 'cuda')
        batch_size: Batch size for processing (32 is optimal for RTX 3090)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model
    """
    # Set torch to use higher precision operations for better embedding quality
    if device == "cuda":
        # Try to optimize CUDA operations
        if torch.cuda.is_available():
            # Set optimal CUDA settings for RTX 3090
            torch.backends.cudnn.benchmark = True  # Speed up training
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
            torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cuDNN

            # Check for available memory and adjust batch size if needed
            free_mem = torch.cuda.get_device_properties(
                0
            ).total_memory - torch.cuda.memory_allocated(0)
            free_mem_gb = free_mem / (1024**3)
            print(f"Available CUDA memory: {free_mem_gb:.2f} GB")

            # If less than 8GB free, reduce batch size
            if free_mem_gb < 8 and batch_size > 16:
                print(
                    f"Reducing batch size from {batch_size} to 16 due to memory constraints"
                )
                batch_size = 16

            # Enable memory efficient attention if model supports it
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                print("Using memory-efficient attention mechanism")

    # Try to load the model
    try:
        # First try with the specified model
        print(f"Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        # If it fails, try alternatives
        print(f"Error loading {model_name}: {e}")
        alternative_models = [
            "bert-base-uncased",  # Simplest BERT model
            "google/bert-base-uncased",
        ]

        for alt_model in alternative_models:
            try:
                print(f"Trying alternative model: {alt_model}")
                tokenizer = AutoTokenizer.from_pretrained(alt_model)
                model = AutoModel.from_pretrained(alt_model).to(device)
                model.eval()
                print(f"Successfully loaded {alt_model}")
                break
            except Exception as alt_e:
                print(f"Error loading {alt_model}: {alt_e}")
        else:
            # If we get here, all alternatives failed
            raise RuntimeError(
                "Could not load any BERT model. Please check your transformers and torch installation."
            )

    # Determine max sequence length for this model
    max_length = 512  # Default for BERT-like models

    # For RTX 3090, we can use larger chunk sizes and process more chunks in parallel
    max_chunks = 12  # Increase from 8 to 12 for RTX 3090
    max_chars = 3000  # Increase from 2000 to 3000 characters per chunk

    def encode_with_chunks(df):
        """
        Encode text with a smart chunking approach optimized for RTX 3090.
        Uses batch processing and half-precision for better performance.
        """
        texts = df["tpcm_text"].tolist()
        embeddings = []

        # Use amp for faster processing if on GPU
        amp_enabled = (
            device == "cuda"
            and hasattr(torch.cuda, "amp")
            and torch.cuda.is_available()
        )
        amp_dtype = torch.float16 if amp_enabled else torch.float32

        if amp_enabled:
            print("Using automatic mixed precision for faster processing")

        # Process in batches for efficiency
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding DSL files"):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = []

            # Process each text in the batch
            for text in batch_texts:
                # Handle empty text case
                if not text or len(text.strip()) == 0:
                    # Create embedding for empty text
                    inputs = tokenizer(
                        " ",  # Use space instead of empty string to avoid errors
                        return_tensors="pt",
                        padding=True,
                    ).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        batch_embeddings.append(
                            outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                        )
                    continue

                # For long DSL files, divide into logical chunks
                text_len = len(text)

                if text_len > max_chars:
                    # Identify natural break points in the text if possible
                    # For DSL, try to break at } characters which often end blocks
                    chunks = []
                    current_pos = 0

                    # Try to find natural break points
                    while current_pos < text_len:
                        # Find next closing bracket after max_chars
                        next_pos = current_pos + max_chars
                        if next_pos >= text_len:
                            # End of text, take the rest
                            chunks.append(text[current_pos:])
                            break

                        # Look for natural break points in order of preference
                        # First try closing brace, then newline, then period
                        break_pos = text.find("}", next_pos)
                        if break_pos == -1 or break_pos > next_pos + 500:
                            # Try to find newline
                            break_pos = text.find("\n", next_pos)
                            if break_pos == -1 or break_pos > next_pos + 500:
                                # Try to find period
                                break_pos = text.find(".", next_pos)
                                if break_pos == -1 or break_pos > next_pos + 500:
                                    # Just use the arbitrary position
                                    chunks.append(text[current_pos:next_pos])
                                    current_pos = next_pos
                                    continue

                        # Use the natural break point (include the punctuation)
                        chunks.append(text[current_pos : break_pos + 1])
                        current_pos = break_pos + 1

                    # Limit chunks to prevent excessive computation
                    # RTX 3090 can handle more chunks
                    chunks = chunks[:max_chunks]
                else:
                    chunks = [text]  # Single chunk for shorter texts

                # Process chunks in a single batch to maximize GPU utilization
                all_inputs = tokenizer(
                    chunks,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length,
                ).to(device)

                with torch.no_grad():
                    if amp_enabled:
                        with torch.amp.autocast("cuda"):
                            outputs = model(**all_inputs)
                            chunk_embeddings = (
                                outputs.last_hidden_state[:, 0, :].cpu().numpy()
                            )
                    else:
                        outputs = model(**all_inputs)
                        chunk_embeddings = (
                            outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        )

                # Combine chunk embeddings with smart weighting
                if len(chunk_embeddings) > 1:
                    # Weight chunks based on their length (longer chunks have more influence)
                    chunk_lengths = np.array([len(c) for c in chunks])
                    weights = chunk_lengths / chunk_lengths.sum()

                    # Apply weighted average
                    weighted_emb = np.zeros_like(chunk_embeddings[0])
                    for j, emb in enumerate(chunk_embeddings):
                        weighted_emb += emb * weights[j]

                    batch_embeddings.append(weighted_emb)
                elif len(chunk_embeddings) == 1:
                    # Just use the single chunk
                    batch_embeddings.append(chunk_embeddings[0])
                else:
                    # Fallback for empty chunks (shouldn't happen)
                    inputs = tokenizer(
                        " ",  # Single space to avoid errors with empty string
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        batch_embeddings.append(
                            outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                        )

            # Add batch embeddings to overall result
            embeddings.extend(batch_embeddings)

            # Clear cache periodically to prevent memory buildup
            if device == "cuda" and (i + batch_size) % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

        # Final cleanup
        if device == "cuda":
            torch.cuda.empty_cache()

        # Stack all embeddings into a single array
        if not embeddings:
            # Handle case where no embeddings were created
            print("Warning: No embeddings were created, returning zero vector")
            dummy_embedding = (
                model(tokenizer(" ", return_tensors="pt").to(device))
                .last_hidden_state[:, 0, :]
                .cpu()
                .numpy()[0]
            )
            return np.zeros((1, dummy_embedding.shape[0]))

        return np.vstack(embeddings)

    # Process datasets with progress reporting
    print(f"Processing training set ({len(train_samples)} samples)...")
    X_train = encode_with_chunks(train_samples)
    y_train = extract_targets(train_samples)

    print(f"Processing validation set ({len(val_samples)} samples)...")
    X_val = encode_with_chunks(val_samples)
    y_val = extract_targets(val_samples)

    print(f"Processing test set ({len(test_samples)} samples)...")
    X_test = encode_with_chunks(test_samples)
    y_test = extract_targets(test_samples)

    print("Feature extraction complete!")
    print(f"Features shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"X_val={X_val.shape}, X_test={X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model
