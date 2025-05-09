"""
Structured feature extraction for DSL models.
This module extracts architectural and performance-related features from DSL models.
"""

import re
import numpy as np
from collections import defaultdict


def extract_structured_features(dsl_text):
    """
    Extract structured features from DSL text that represent architectural properties.

    Args:
        dsl_text: Text content of a DSL model file

    Returns:
        Dictionary of extracted features
    """
    features = {}

    # Basic component counts
    features["component_count"] = len(re.findall(r"component\s+component_", dsl_text))
    features["interface_count"] = len(re.findall(r"interface\s+interface_", dsl_text))
    features["operation_count"] = len(re.findall(r"op\s+operation_", dsl_text))

    # Connection complexity
    features["assembly_count"] = len(re.findall(r"assembly\s+assembly_", dsl_text))
    features["provides_count"] = len(re.findall(r"provides", dsl_text))
    features["requires_count"] = len(re.findall(r"requires", dsl_text))

    # System boundary services
    features["system_provided_count"] = len(re.findall(r"system_provided_", dsl_text))

    # Resource demands
    cpu_processing = re.findall(
        r"cpu\s*\.\s*process\s*\(\s*«\s*([\d\.E]+)\s*»\s*\)", dsl_text
    )
    features["cpu_demand_count"] = len(cpu_processing)
    features["total_cpu_demand"] = (
        sum([float(x) for x in cpu_processing]) if cpu_processing else 0
    )
    features["avg_cpu_demand"] = features["total_cpu_demand"] / max(
        1, features["cpu_demand_count"]
    )
    features["max_cpu_demand"] = (
        max([float(x) for x in cpu_processing]) if cpu_processing else 0
    )

    # HDD operations
    hdd_operations = re.findall(
        r"hdd\s*\.\s*(read|write)\s*\(\s*«\s*([\d\.E]+)\s*»\s*\)", dsl_text
    )
    features["hdd_op_count"] = len(hdd_operations)

    # Usage scenario metrics
    populations = re.findall(r"population\s*\(\s*«\s*([\d\.]+)\s*»\s*\)", dsl_text)
    features["user_population"] = float(populations[0]) if populations else 0

    think_times = re.findall(r"thinkTime\s*\(\s*«\s*([\d\.]+)\s*»\s*\)", dsl_text)
    features["think_time"] = float(think_times[0]) if think_times else 0

    # Scenario complexity - length of usage scenario
    scenario_steps = len(re.findall(r"\.\s*operation_", dsl_text))
    features["scenario_steps"] = scenario_steps

    # Resource environment properties
    processing_rates = re.findall(r"processingRate\s*:\s*«\s*([\d\.E]+)\s*»", dsl_text)
    features["processing_rates_sum"] = (
        sum([float(x) for x in processing_rates]) if processing_rates else 0
    )
    features["processing_rates_count"] = len(processing_rates)
    features["avg_processing_rate"] = features["processing_rates_sum"] / max(
        1, features["processing_rates_count"]
    )

    # Network properties
    throughputs = re.findall(r"throughput\s*:\s*«\s*([\d\.E]+)\s*»", dsl_text)
    features["network_throughput"] = float(throughputs[0]) if throughputs else 0

    # Count resource types used
    features["cpu_resource_count"] = len(re.findall(r"CPU\s+CPUResource", dsl_text))
    features["hdd_resource_count"] = len(re.findall(r"HDDResource", dsl_text))
    features["link_count"] = len(re.findall(r"link\s+", dsl_text))

    # Scheduling policies
    features["process_sharing_count"] = len(re.findall(r"ProcessorSharing", dsl_text))
    features["fcfs_count"] = len(re.findall(r"FirstComeFirstServe", dsl_text))
    features["delay_count"] = len(re.findall(r"Delay", dsl_text))

    # Error handling
    features["failure_count"] = len(re.findall(r"failure", dsl_text))
    features["raises_count"] = len(re.findall(r"raises", dsl_text))

    # Parameter types and counts
    features["integer_param_count"] = len(re.findall(r"param\d+\s+Integer", dsl_text))
    features["double_param_count"] = len(re.findall(r"param\d+\s+Double", dsl_text))
    features["string_param_count"] = len(re.findall(r"param\d+\s+String", dsl_text))
    features["boolean_param_count"] = len(re.findall(r"param\d+\s+Boolean", dsl_text))

    # Count SEFFs (service effect specifications)
    features["seff_count"] = len(re.findall(r"seff\s+", dsl_text))

    # Resource pool sizes
    passive_resources = re.findall(r"capacity\s*:\s*«\s*([\d\.]+)\s*»", dsl_text)
    features["total_passive_capacity"] = (
        sum([float(x) for x in passive_resources]) if passive_resources else 0
    )

    # Calculate complexity metrics
    features["component_interface_ratio"] = features["interface_count"] / max(
        1, features["component_count"]
    )
    features["operations_per_interface"] = features["operation_count"] / max(
        1, features["interface_count"]
    )
    features["provides_requires_ratio"] = features["provides_count"] / max(
        1, features["requires_count"]
    )

    # Check for specific architectural patterns
    features["has_loadbalancing"] = 1 if "LoadBalancingResource" in dsl_text else 0
    features["has_caching"] = 1 if "Cache" in dsl_text else 0
    features["has_replication"] = 1 if "Replication" in dsl_text else 0

    return features


def combine_structured_and_embedded_features(structured_features, embedded_features):
    """
    Combine structured features with embedded features.

    Args:
        structured_features: Dictionary of structural features
        embedded_features: Embedded feature vector (e.g., TF-IDF, BERT)

    Returns:
        Combined feature vector
    """
    import numpy as np
    from scipy import sparse

    # Convert structured features to array
    struct_array = np.array(list(structured_features.values())).reshape(1, -1)

    # Check if embedded features are sparse
    if sparse.issparse(embedded_features):
        # Convert structured features to sparse and horizontally stack
        struct_sparse = sparse.csr_matrix(struct_array)
        return sparse.hstack([struct_sparse, embedded_features])
    else:
        # For dense arrays, use numpy concatenation
        return np.concatenate([struct_array, embedded_features], axis=1)


def process_batch_with_structured_features(texts, embedding_function):
    """
    Process a batch of DSL texts by extracting both structured and embedded features.

    Args:
        texts: List of DSL texts
        embedding_function: Function to generate embeddings

    Returns:
        Combined feature matrix
    """
    # Extract structured features
    structured_features_list = [extract_structured_features(text) for text in texts]

    # Get keys from the first feature dict
    if not structured_features_list:
        return None

    feature_keys = list(structured_features_list[0].keys())

    # Convert to array format
    structured_arrays = []
    for features in structured_features_list:
        # Ensure all features have the same keys in the same order
        feat_array = [features.get(key, 0) for key in feature_keys]
        structured_arrays.append(feat_array)

    structured_matrix = np.array(structured_arrays)

    # Generate embeddings
    embedded_features = embedding_function(texts)

    # Check if embedded features are sparse
    if hasattr(embedded_features, "toarray"):
        # Convert structured to sparse and combine
        struct_sparse = sparse.csr_matrix(structured_matrix)
        return sparse.hstack([struct_sparse, embedded_features])
    else:
        # For dense arrays, use numpy concatenation
        return np.concatenate([structured_matrix, embedded_features], axis=1)
