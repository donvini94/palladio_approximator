from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm


def extract_targets(df):
    return df[["avg_resp_time", "min_resp_time", "max_resp_time"]].to_numpy()


def build_tfidf_features(train_samples, val_samples, test_samples):
    """
    Builds TF-IDF feature matrices and target arrays from dataset samples.

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
    batch_size=8,
    chunking_strategy="hierarchical",
):
    """
    Builds CodeBERT embeddings and target arrays from dataset samples with improved handling for long texts.

    Args:
        train_samples, val_samples, test_samples: Dataset samples
        model_name: Name of the pretrained model to use
        device: Device to run inference on ('cpu' or 'cuda')
        batch_size: Batch size for processing
        chunking_strategy: How to handle long texts that exceed model max tokens
                          - 'truncate': Simple truncation (original behavior)
                          - 'average': Split into chunks and average their embeddings
                          - 'hierarchical': Create embeddings for chunks and then aggregate them

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Determine max sequence length for this model
    max_length = 512  # Default for BERT-like models

    def encode_with_chunks(df, chunk_strategy=chunking_strategy):
        texts = df["tpcm_text"].tolist()
        embeddings = []

        for i in tqdm(
            range(0, len(texts), batch_size), desc=f"Encoding with {chunk_strategy}"
        ):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = []

            for text in batch_texts:
                if chunk_strategy == "truncate":
                    # Simple truncation (original behavior)
                    inputs = tokenizer(
                        text,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=max_length,
                    ).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs)
                        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                    batch_embeddings.append(cls_embedding[0])

                elif chunk_strategy == "average":
                    # Split text into chunks and average their embeddings
                    tokens = tokenizer.tokenize(text)
                    chunk_size = max_length - 2  # Account for [CLS] and [SEP]
                    chunks = [
                        tokens[j : j + chunk_size]
                        for j in range(0, len(tokens), chunk_size)
                    ]

                    if not chunks:  # Handle empty text case
                        chunks = [[]]

                    chunk_embeddings = []
                    for chunk in chunks[
                        :10
                    ]:  # Limit to first 10 chunks to prevent excessive computation
                        chunk_text = tokenizer.convert_tokens_to_string(chunk)
                        inputs = tokenizer(
                            chunk_text,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=max_length,
                        ).to(device)

                        with torch.no_grad():
                            outputs = model(**inputs)
                            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                            chunk_embeddings.append(emb[0])

                    # Average all chunk embeddings
                    if chunk_embeddings:
                        avg_embedding = np.mean(chunk_embeddings, axis=0)
                        batch_embeddings.append(avg_embedding)
                    else:
                        # Fallback for empty text
                        inputs = tokenizer(
                            "",
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=max_length,
                        ).to(device)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            batch_embeddings.append(
                                outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                            )

                elif chunk_strategy == "hierarchical":
                    # Hierarchical approach: create embeddings for chunks, then a second embedding pass for combined chunks
                    tokens = tokenizer.tokenize(text)
                    chunk_size = max_length - 2  # Account for [CLS] and [SEP]
                    chunks = [
                        tokens[j : j + chunk_size]
                        for j in range(0, len(tokens), chunk_size)
                    ]

                    if not chunks:  # Handle empty text case
                        chunks = [[]]

                    # First level: get embeddings for each chunk
                    chunk_embeddings = []
                    for chunk in chunks[
                        :8
                    ]:  # Limit chunks to avoid excessive computation
                        chunk_text = tokenizer.convert_tokens_to_string(chunk)
                        inputs = tokenizer(
                            chunk_text,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=max_length,
                        ).to(device)

                        with torch.no_grad():
                            outputs = model(**inputs)
                            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                            chunk_embeddings.append(emb[0])

                    # If we only have one chunk, use it directly
                    if len(chunk_embeddings) == 1:
                        batch_embeddings.append(chunk_embeddings[0])
                    elif len(chunk_embeddings) > 1:
                        # Second level: combine chunk embeddings with another embedding pass
                        # Just concatenate the first few and last few chunks to capture beginning and end of text
                        important_parts = []
                        if len(chunks) > 0:
                            important_parts.append(
                                tokenizer.convert_tokens_to_string(chunks[0])
                            )  # First chunk
                        if len(chunks) > 1:
                            important_parts.append(
                                tokenizer.convert_tokens_to_string(chunks[1])
                            )  # Second chunk
                        if len(chunks) > 2:
                            important_parts.append(
                                tokenizer.convert_tokens_to_string(chunks[-1])
                            )  # Last chunk

                        combined_text = " ".join(important_parts)
                        inputs = tokenizer(
                            combined_text,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=max_length,
                        ).to(device)

                        with torch.no_grad():
                            outputs = model(**inputs)
                            final_embedding = (
                                outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                            )
                            batch_embeddings.append(final_embedding)
                    else:
                        # Fallback for empty text
                        inputs = tokenizer(
                            "",
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=max_length,
                        ).to(device)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            batch_embeddings.append(
                                outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                            )

            embeddings.extend(batch_embeddings)

        return np.vstack(embeddings)

    X_train = encode_with_chunks(train_samples)
    y_train = extract_targets(train_samples)

    X_val = encode_with_chunks(val_samples)
    y_val = extract_targets(val_samples)

    X_test = encode_with_chunks(test_samples)
    y_test = extract_targets(test_samples)

    return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model


def build_longformer_features(
    train_samples,
    val_samples,
    test_samples,
    model_name="allenai/longformer-base-4096",
    device="cpu",
    batch_size=2,  # bigger inputs = smaller batch size
    use_global_attention=True,
):
    """
    Builds Longformer embeddings and target arrays from dataset samples.
    Takes advantage of Longformer's increased context window and global attention mechanism.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def encode(df):
        texts = df["tpcm_text"].tolist()
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding="longest",
                truncation=True,
                max_length=4096,  # Longformer can handle up to 4096
                return_tensors="pt",
            ).to(device)

            # Set global attention on special tokens and first tokens if requested
            if (
                use_global_attention
                and hasattr(model, "config")
                and "longformer" in model.config.architectures[0].lower()
            ):
                # Create global attention mask - give attention to [CLS] token
                global_attention_mask = torch.zeros_like(inputs["attention_mask"])
                # Set global attention on [CLS] token
                global_attention_mask[:, 0] = 1

                # Optionally set global attention on special syntax elements common in DSL
                for keyword in [
                    "{",
                    "}",
                    "repository",
                    "system",
                    "interface",
                    "component",
                ]:
                    keyword_ids = tokenizer.encode(keyword, add_special_tokens=False)
                    if len(keyword_ids) > 0:
                        # Find all occurrences of keyword in input_ids
                        for batch_idx in range(inputs["input_ids"].shape[0]):
                            for token_idx in range(
                                inputs["input_ids"].shape[1] - len(keyword_ids) + 1
                            ):
                                if torch.all(
                                    inputs["input_ids"][
                                        batch_idx,
                                        token_idx : token_idx + len(keyword_ids),
                                    ]
                                    == torch.tensor(keyword_ids).to(device)
                                ):
                                    global_attention_mask[
                                        batch_idx,
                                        token_idx : token_idx + len(keyword_ids),
                                    ] = 1

                # Use the global attention mask
                inputs["global_attention_mask"] = global_attention_mask

            with torch.no_grad():
                # Handle different Longformer implementations
                if (
                    use_global_attention
                    and "global_attention_mask" in inputs
                    and hasattr(model, "forward")
                    and "global_attention_mask" in model.forward.__code__.co_varnames
                ):
                    outputs = model(**inputs)
                else:
                    # Fall back to standard forward pass without global attention
                    if "global_attention_mask" in inputs:
                        del inputs["global_attention_mask"]
                    outputs = model(**inputs)

                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            embeddings.append(cls_embeddings)

        return np.vstack(embeddings)

    X_train = encode(train_samples)
    y_train = extract_targets(train_samples)

    X_val = encode(val_samples)
    y_val = extract_targets(val_samples)

    X_test = encode(test_samples)
    y_test = extract_targets(test_samples)

    return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model


def build_chunk_aware_features(
    train_samples,
    val_samples,
    test_samples,
    model_name="microsoft/codebert-base",
    device="cpu",
    batch_size=8,
):
    """
    Builds features that are aware of the DSL structure by extracting key sections
    and processing them separately before combining.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Set max token limits based on model
    if "longformer" in model_name:
        max_length = 4096
    else:
        max_length = 512

    def extract_key_sections(text):
        """Extract key sections from the DSL based on structure"""
        sections = {}

        # Extract repository section
        repo_match = text.split("repository ")
        if len(repo_match) > 1:
            sections["repository"] = "repository " + repo_match[1].split(" }")[0] + " }"

        # Extract system section
        system_match = text.split("system ")
        if len(system_match) > 1:
            sections["system"] = "system " + system_match[1].split(" }")[0] + " }"

        # Extract usage section
        usage_match = text.split("usage ")
        if len(usage_match) > 1:
            sections["usage"] = "usage " + usage_match[1].split(" }")[0] + " }"

        # Get interfaces and components
        interface_matches = [s for s in text.split("interface ")[1:]]
        if interface_matches:
            sections["interfaces"] = "interface " + " interface ".join(
                [m.split(" }")[0] + " }" for m in interface_matches[:3]]
            )

        component_matches = [s for s in text.split("component ")[1:]]
        if component_matches:
            sections["components"] = "component " + " component ".join(
                [m.split(" }")[0] + " }" for m in component_matches[:2]]
            )

        return sections

    def encode(df):
        texts = df["tpcm_text"].tolist()
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding DSL sections"):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = []

            for text in batch_texts:
                # Extract key sections
                sections = extract_key_sections(text)

                if not sections:
                    # If no sections extracted, use original text with truncation
                    inputs = tokenizer(
                        text,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=max_length,
                    ).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs)
                        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

                    batch_embeddings.append(embedding)
                    continue

                # Process each section and collect embeddings
                section_embeddings = []
                for section_name, section_text in sections.items():
                    inputs = tokenizer(
                        section_text,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=max_length,
                    ).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs)
                        section_emb = (
                            outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                        )
                        section_embeddings.append(section_emb)

                # Combine section embeddings
                if section_embeddings:
                    # Average all section embeddings
                    final_embedding = np.mean(section_embeddings, axis=0)
                    batch_embeddings.append(final_embedding)
                else:
                    # Fall back to processing full text with truncation
                    inputs = tokenizer(
                        text[
                            : max_length * 4
                        ],  # Use more tokens for first-level tokenization
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=max_length,
                    ).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs)
                        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

                    batch_embeddings.append(embedding)

            all_embeddings.extend(batch_embeddings)

        return np.vstack(all_embeddings)

    X_train = encode(train_samples)
    y_train = extract_targets(train_samples)

    X_val = encode(val_samples)
    y_val = extract_targets(val_samples)

    X_test = encode(test_samples)
    y_test = extract_targets(test_samples)

    return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model
