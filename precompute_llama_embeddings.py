#!/usr/bin/env python3
"""
Optimized script to pre-compute LLaMA embeddings for DSL models.
This version maximizes GPU utilization to speed up embedding generation.

Usage:
  python precompute_llama_embeddings.py --input_dir data/dsl_models --output_dir features/llama_embeddings
"""

import os
import time
import argparse
import gc
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
from concurrent.futures import ThreadPoolExecutor
import psutil
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llama_embedding_generation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Pre-compute LLaMA embeddings for DSL models"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/dsl_models",
        help="Directory containing DSL files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="features/llama_embeddings",
        help="Directory to save embeddings",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="codellama/CodeLlama-7b-hf",
        help="LLaMA model to use",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Maximum chunk size for processing long texts",
    )
    parser.add_argument(
        "--file_batch_size",
        type=int,
        default=8,
        help="Number of files to process in parallel",
    )
    parser.add_argument(
        "--chunk_batch_size",
        type=int,
        default=4,
        help="Number of chunks to process in parallel",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization for model",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        default=False,
        help="Use 8-bit quantization for model",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation of existing embeddings",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="embedding_metadata.json",
        help="File to store metadata about the embeddings",
    )
    parser.add_argument(
        "--max_ram_percentage",
        type=float,
        default=0.9,
        help="Maximum RAM percentage to use (0.1-0.95)",
    )
    parser.add_argument(
        "--max_vram_percentage",
        type=float,
        default=0.9,
        help="Maximum VRAM percentage to use (0.1-0.95)",
    )
    return parser.parse_args()


def estimate_optimal_batch_sizes(args):
    """
    Estimate optimal batch sizes based on available system resources and model size
    """
    # Get available RAM
    ram_available = psutil.virtual_memory().available
    total_ram = psutil.virtual_memory().total

    # Get available VRAM if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        total_vram = torch.cuda.get_device_properties(0).total_memory
        vram_allocated = torch.cuda.memory_allocated(0)
        vram_available = total_vram - vram_allocated

        logger.info(f"Total VRAM: {total_vram / 1e9:.2f} GB")
        logger.info(f"Available VRAM: {vram_available / 1e9:.2f} GB")
    else:
        vram_available = 0
        total_vram = 0

    logger.info(f"Total RAM: {total_ram / 1e9:.2f} GB")
    logger.info(f"Available RAM: {ram_available / 1e9:.2f} GB")

    # Estimate model size based on quantization
    if args.use_4bit:
        model_size_factor = 0.25  # 4-bit models use ~1/4 of the original memory
    elif args.use_8bit:
        model_size_factor = 0.5  # 8-bit models use ~1/2 of the original memory
    else:
        model_size_factor = 1.0  # Full precision models

    # Base model sizes in GB for different model sizes
    model_size_mapping = {
        "7b": 14,  # ~14GB for 7B model in full precision
        "13b": 26,  # ~26GB for 13B model in full precision
        "70b": 140,  # ~140GB for 70B model in full precision
    }

    # Try to determine model size from name
    model_size = "7b"  # Default
    for size in model_size_mapping.keys():
        if size in args.model_name.lower():
            model_size = size
            break

    estimated_model_size = model_size_mapping[model_size] * model_size_factor
    logger.info(
        f"Estimated model size: {estimated_model_size:.2f} GB ({model_size} model with {model_size_factor*100:.0f}% of original size)"
    )

    # Calculate optimal batch sizes
    if torch.cuda.is_available():
        # If using GPU, consider VRAM as the limiting factor
        available_vram_for_batches = (
            vram_available * args.max_vram_percentage - estimated_model_size * 1e9
        )

        if available_vram_for_batches <= 0:
            logger.warning(
                f"Not enough VRAM! Model needs approximately {estimated_model_size:.2f} GB but only {vram_available / 1e9:.2f} GB available."
            )
            # Default to conservative values
            file_batch_size = 1
            chunk_batch_size = 1
        else:
            # Each 1K token chunk needs ~8MB of VRAM for intermediate activations at 4-bit precision
            chunk_vram_usage = (
                8 * 1024 * 1024 * (args.chunk_size / 1024) * (1 / model_size_factor)
            )
            max_chunks = available_vram_for_batches / chunk_vram_usage

            # Optimize file and chunk batches
            chunk_batch_size = min(args.chunk_batch_size, max(1, int(max_chunks)))
            file_batch_size = min(
                args.file_batch_size, max(1, int(max_chunks / chunk_batch_size))
            )
    else:
        # If using CPU, use conservative values
        file_batch_size = 1
        chunk_batch_size = 1

    # Ensure file batch size is at least 1
    file_batch_size = max(1, file_batch_size)
    chunk_batch_size = max(1, chunk_batch_size)

    logger.info(
        f"Optimized batch sizes - Files: {file_batch_size}, Chunks: {chunk_batch_size}"
    )

    return file_batch_size, chunk_batch_size


def setup_model(model_name, use_4bit=True, use_8bit=False):
    """Set up the LLaMA model with memory optimizations"""
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Free up memory before loading model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    logger.info(f"Loading model {model_name}...")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure model loading parameters
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
            "device_map": "auto",  # Automatically place model parts on available devices
        }

        # Set up quantization
        if use_4bit:
            logger.info("Using 4-bit quantization for maximum memory efficiency")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quantization_config

        elif use_8bit:
            logger.info("Using 8-bit quantization for better memory efficiency")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = quantization_config

        # Load the model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        model.eval()  # Set to evaluation mode
        logger.info(f"Successfully loaded {model_name}")

        # Log memory usage
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(
                f"GPU memory after model load: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
            )

        return tokenizer, model, device

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def process_text_in_chunks(
    text, tokenizer, model, device, max_length=1024, batch_size=4
):
    """
    Process text in chunks with efficient batching for long documents

    Args:
        text: Text to embed
        tokenizer: HuggingFace tokenizer
        model: LLaMA model
        device: Device to use for computation
        max_length: Maximum sequence length
        batch_size: Number of chunks to process in parallel

    Returns:
        numpy.ndarray: Embedding vector
    """
    if not text or len(text.strip()) == 0:
        logger.warning("Empty text provided, returning zero embedding")
        embedding_dim = model.config.hidden_size
        return np.zeros(embedding_dim)

    # Tokenize the text to get token IDs
    tokenized_text = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
        return_tensors=None,
    )

    token_count = len(tokenized_text)

    try:
        if token_count > max_length:
            # For long documents, use a chunking approach
            logger.info(
                f"Long document with {token_count} tokens. Processing in chunks..."
            )

            # Determine chunk size and overlap
            effective_length = max_length - 100  # 100 tokens overlap

            # Create overlapping chunks of tokens
            chunks = []
            for i in range(0, token_count, effective_length):
                # Get token IDs for this chunk with overlap
                end_idx = min(i + max_length, token_count)
                chunk_tokens = tokenized_text[i:end_idx]
                chunk_text = tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)

            logger.info(f"Created {len(chunks)} chunks")

            # Process chunks in batches for better GPU utilization
            all_embeddings = []

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]

                # Tokenize all chunks in the batch
                batch_inputs = tokenizer(
                    batch_chunks,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)

                # Process batch
                with torch.no_grad():
                    with torch.amp.autocast("cuda"):
                        outputs = model(**batch_inputs, output_hidden_states=True)

                        # Get last hidden states
                        hidden_states = outputs.hidden_states[-1]

                        # Apply mean pooling for each chunk
                        masks = batch_inputs.attention_mask.unsqueeze(-1)
                        batch_embeddings = torch.sum(
                            hidden_states * masks, dim=1
                        ) / torch.sum(masks, dim=1)

                        # Move to CPU and convert to numpy
                        batch_embeddings = batch_embeddings.cpu().numpy()

                # Normalize each embedding
                for embedding in batch_embeddings:
                    normalized_embedding = embedding / np.linalg.norm(embedding)
                    all_embeddings.append(normalized_embedding)

                # Clear GPU memory
                if device == "cuda":
                    del batch_inputs, outputs, hidden_states
                    torch.cuda.empty_cache()

            # Combine chunk embeddings with position-based weighting
            if len(all_embeddings) > 1:
                # Position weights: emphasize start and end
                num_chunks = len(all_embeddings)
                pos_weights = np.ones(num_chunks)
                pos_weights[0] = 1.5  # First chunk (often contains imports/definitions)
                pos_weights[-1] = 1.3  # Last chunk (often contains main logic)

                # Normalize weights
                pos_weights = pos_weights / pos_weights.sum()

                # Apply weighted average
                final_embedding = np.zeros_like(all_embeddings[0])
                for i, emb in enumerate(all_embeddings):
                    final_embedding += emb * pos_weights[i]

                # Normalize final embedding
                final_embedding = final_embedding / np.linalg.norm(final_embedding)

                return final_embedding
            elif len(all_embeddings) == 1:
                return all_embeddings[0]
            else:
                raise ValueError("No chunks were processed")

        else:
            # For shorter texts, process the whole text at once
            inputs = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    outputs = model(**inputs, output_hidden_states=True)
                    # Use last hidden layer representation
                    hidden_states = outputs.hidden_states[-1]
                    # Mean pooling over sequence
                    mask = inputs.attention_mask.unsqueeze(-1)
                    embedding = torch.sum(hidden_states * mask, dim=1) / torch.sum(
                        mask, dim=1
                    )
                    embedding = embedding.cpu().numpy()[0]

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            return embedding

    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        embedding_dim = model.config.hidden_size
        return np.zeros(embedding_dim)


def process_file_batch(file_batch, tokenizer, model, device, output_dir, args):
    """
    Process a batch of files and save embeddings

    Args:
        file_batch: List of file paths to process
        tokenizer: HuggingFace tokenizer
        model: LLaMA model
        device: Device to use for computation
        output_dir: Directory to save embeddings
        args: Command line arguments

    Returns:
        List of metadata dictionaries for each file
    """
    metadata_entries = []

    for file_path in file_batch:
        output_path = Path(output_dir) / f"{file_path.stem}.npy"

        # Process this file
        logger.info(f"Processing {file_path}")
        try:
            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Generate embedding
            start_time = time.time()
            embedding = process_text_in_chunks(
                text,
                tokenizer,
                model,
                device,
                max_length=args.chunk_size,
                batch_size=args.chunk_batch_size,
            )
            elapsed_time = time.time() - start_time

            # Save the embedding
            np.save(output_path, embedding)

            # Create metadata entry
            metadata_entry = {
                "input_file": str(file_path),
                "output_file": str(output_path),
                "file_size_bytes": os.path.getsize(file_path),
                "token_count": len(tokenizer.encode(text)),
                "processing_time_seconds": elapsed_time,
                "embedding_shape": embedding.shape,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            logger.info(
                f"Processed {file_path} in {elapsed_time:.2f} seconds. Embedding shape: {embedding.shape}"
            )
            metadata_entries.append(metadata_entry)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            metadata_entries.append(
                {
                    "input_file": str(file_path),
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    return metadata_entries


def save_metadata(metadata, output_dir, metadata_file):
    """Save metadata to file"""
    import json

    metadata_path = Path(output_dir) / metadata_file
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def pre_compute_embeddings(args):
    """
    Pre-compute embeddings for all files in the input directory

    Args:
        args: Command line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get a list of all files to process
    input_files = list(Path(args.input_dir).glob("*.tpcm"))
    logger.info(f"Found {len(input_files)} files to process")

    # Filter out already processed files
    if not args.force_recompute:
        already_processed = set(f.stem for f in Path(args.output_dir).glob("*.npy"))
        input_files = [f for f in input_files if f.stem not in already_processed]
        logger.info(
            f"{len(input_files)} files left to process after filtering already computed embeddings"
        )

    if not input_files:
        logger.info("No files to process. Exiting.")
        return

    # Estimate optimal batch sizes
    file_batch_size, chunk_batch_size = estimate_optimal_batch_sizes(args)

    # Update args with estimated values
    args.file_batch_size = min(args.file_batch_size, file_batch_size)
    args.chunk_batch_size = min(args.chunk_batch_size, chunk_batch_size)

    # Set up model
    tokenizer, model, device = setup_model(
        args.model_name, use_4bit=args.use_4bit, use_8bit=args.use_8bit
    )

    # Process files
    metadata = {
        "model_name": args.model_name,
        "embedding_dim": model.config.hidden_size,
        "chunk_size": args.chunk_size,
        "file_batch_size": args.file_batch_size,
        "chunk_batch_size": args.chunk_batch_size,
        "processed_files": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        # Process files in batches
        for i in range(0, len(input_files), args.file_batch_size):
            batch_files = input_files[i : i + args.file_batch_size]

            logger.info(
                f"Processing batch {i//args.file_batch_size + 1}/{math.ceil(len(input_files)/args.file_batch_size)} ({len(batch_files)} files)"
            )

            # Process this batch
            batch_metadata = process_file_batch(
                batch_files, tokenizer, model, device, args.output_dir, args
            )

            # Update metadata
            for entry in batch_metadata:
                if "input_file" in entry:
                    file_stem = Path(entry["input_file"]).stem
                    metadata["processed_files"][file_stem] = entry

            # Save metadata after each batch
            save_metadata(metadata, args.output_dir, args.metadata_file)

            # Clear GPU memory
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

                # Log memory usage
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(
                    f"GPU memory after batch: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
                )

    finally:
        # Save final metadata
        save_metadata(metadata, args.output_dir, args.metadata_file)
        logger.info("Embedding pre-computation complete")


def main():
    """Main function"""
    args = parse_args()

    # Print CUDA info
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA not available. Using CPU only.")

    try:
        pre_compute_embeddings(args)
    except Exception as e:
        logger.error(f"Error during embedding pre-computation: {str(e)}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
