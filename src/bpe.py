"""
Byte Pair Encoding (BPE) implementation.

This module provides functions to perform Byte Pair Encoding on a given corpus.
It includes functionality for handling special characters, encoding words,
and saving/loading BPE models.
"""

import json
import logging
import re
from typing import List, Tuple, Dict, Iterator, Optional
from collections import Counter
from dataclasses import dataclass, field
import argparse
from pathlib import Path
from itertools import pairwise, chain
from tqdm import tqdm
import multiprocessing as mp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BPEConfig:
    """Configuration class for BPE parameters."""
    corpus_file: Path
    num_merges: int
    model_file: Path
    special_chars: str = r'[^\w\s-]'
    special_char_handling: str = 'replace'
    replacement_char: str = '_'
    log_level: str = 'INFO'
    batch_size: int = 10000
    num_cores: Optional[int] = None

    __slots__ = ('corpus_file', 'num_merges', 'model_file', 'special_chars',
                 'special_char_handling', 'replacement_char', 'log_level', 'batch_size', 'num_cores')

    def validate(self) -> None:
        """Validate the configuration parameters."""
        if not self.corpus_file.exists():
            raise FileNotFoundError(f"Corpus file {self.corpus_file} does not exist.")
        if not self.model_file.parent.exists():
            raise FileNotFoundError(f"Model save directory {self.model_file.parent} does not exist.")
        if self.num_merges <= 0:
            raise ValueError("Number of merges must be a positive integer")
        if self.batch_size <= 0:
            logger.warning("Invalid batch size, defaulting to 10000")
            self.batch_size = 10000


def load_config(config_file: Path) -> BPEConfig:
    """
    Load configuration from a JSON file.

    Args:
        config_file (Path): Path to the configuration file.

    Returns:
        BPEConfig: Configuration object.

    Raises:
        ValueError: If the configuration file cannot be loaded or parsed.
    """
    try:
        with config_file.open('r') as f:
            config_dict = json.load(f)
        config_dict['corpus_file'] = Path(config_dict['corpus_file'])
        config_dict['model_file'] = Path(config_dict['model_file'])
        
        config = BPEConfig(**config_dict)
        config.validate()
        
        # Set log level
        logger.setLevel(config.log_level)
        
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise ValueError(f"Failed to load configuration: {str(e)}") from e


def handle_special_chars(word: str, config: BPEConfig) -> str:
    """
    Handle special characters in a word based on the configuration.

    Args:
        word (str): The input word.
        config (BPEConfig): Configuration object.

    Returns:
        str: The word with special characters handled according to the configuration.
    """
    if config.special_char_handling == 'keep':
        return word
    elif config.special_char_handling == 'remove':
        return re.sub(config.special_chars, '', word)
    elif config.special_char_handling == 'replace':
        return re.sub(config.special_chars, config.replacement_char, word)
    else:
        logger.warning(f"Unknown special_char_handling option: {config.special_char_handling}. Defaulting to 'remove'.")
        return re.sub(config.special_chars, '', word)


def prepare_corpus(corpus: Iterator[str], config: BPEConfig) -> Iterator[List[str]]:
    """
    Prepare the corpus by handling special characters and adding end-of-word token.

    Args:
        corpus (Iterator[str]): Iterator of words in the corpus.
        config (BPEConfig): Configuration object.

    Yields:
        List[str]: Word represented as a list of characters with an added end-of-word token.
    """
    for word in corpus:
        word = handle_special_chars(word.lower(), config)
        yield list(word) + ['</w>']


def get_pair_counts(corpus: Iterator[List[str]]) -> Counter:
    """
    Count the frequency of symbol pairs in the corpus.

    Args:
        corpus (Iterator[List[str]]): The prepared corpus with end-of-word tokens.

    Returns:
        Counter: A counter mapping symbol pairs to their frequency in the corpus.
    """
    return Counter(chain.from_iterable(pairwise(word) for word in corpus))


def merge_pair(word: List[str], pair: Tuple[str, str], merged: str) -> List[str]:
    """
    Merge the given pair in a word.

    Args:
        word (List[str]): The word to process.
        pair (Tuple[str, str]): The pair to merge.
        merged (str): The merged representation of the pair.

    Returns:
        List[str]: Updated word with the pair merged.
    """
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            new_word.append(merged)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word


def perform_merge_step(corpus: List[List[str]], pair: Tuple[str, str], merged: str) -> List[List[str]]:
    """
    Perform one step of merging for the entire corpus.

    Args:
        corpus (List[List[str]]): The current state of the corpus.
        pair (Tuple[str, str]): The pair to merge.
        merged (str): The merged representation of the pair.

    Returns:
        List[List[str]]: Updated corpus after performing the merge.
    """
    return [merge_pair(word, pair, merged) for word in corpus]


def byte_pair_encoding(
    corpus: Iterator[str],
    config: BPEConfig
) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """
    Perform Byte Pair Encoding (BPE) on the given corpus.

    Args:
        corpus (Iterator[str]): An iterator of words in the corpus.
        config (BPEConfig): Configuration object.

    Returns:
        Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
            - Dictionary mapping original words to their BPE tokens.
            - List of merge operations performed.

    Raises:
        ValueError: If an error occurs during the BPE process.
    """
    try:
        vocab = {}
        merges = []
        corpus = list(prepare_corpus(corpus, config))

        logger.info("Starting BPE process...")
        for i in tqdm(range(config.num_merges), desc="Performing BPE merges"):
            pair_counts = get_pair_counts(corpus)
            if not pair_counts:
                logger.info(f"No more pairs to merge after {i} merges")
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            merges.append(best_pair)
            merged = ''.join(best_pair)
            corpus = perform_merge_step(corpus, best_pair, merged)
            logger.debug(f"Merge {i + 1}: {best_pair}")

        logger.info("BPE process completed")

        # Build the final vocabulary
        for original_word, bpe_tokens in zip(corpus, [' '.join(word) for word in corpus]):
            vocab[''.join(original_word)] = bpe_tokens.split()

        return vocab, merges
    except Exception as e:
        logger.error(f"Error during BPE: {str(e)}")
        raise ValueError("BPE process failed") from e


def build_bpe_vocab(merges: List[Tuple[str, str]]) -> Dict[str, int]:
    """
    Build a vocabulary from the learned BPE merge operations.

    Args:
        merges (List[Tuple[str, str]]): List of BPE merge operations.

    Returns:
        Dict[str, int]: A dictionary mapping merged tokens to unique indices.
    """
    vocab = {}
    for i, pair in enumerate(merges):
        merged_token = ''.join(pair)
        if merged_token not in vocab:
            vocab[merged_token] = i
        else:
            logger.warning(f"Duplicate merge token encountered: {merged_token}")
    return vocab


def load_corpus_from_file(file_path: Path) -> Iterator[str]:
    """
    Load corpus from a file.

    Args:
        file_path (Path): Path to the file containing the corpus.

    Yields:
        str: Words in the corpus.

    Raises:
        IOError: If the corpus file cannot be read.
    """
    try:
        with file_path.open('r', encoding='utf-8') as file:
            for line in file:
                yield from line.split()
    except IOError as e:
        logger.error(f"Error reading corpus file: {str(e)}")
        raise


def save_model(vocab: Dict[str, List[str]], merges: List[Tuple[str, str]], file_path: Path) -> None:
    """
    Save the BPE model (vocabulary and merges) to a file.

    Args:
        vocab (Dict[str, List[str]]): BPE vocabulary.
        merges (List[Tuple[str, str]]): List of BPE merge operations.
        file_path (Path): Path to save the model.

    Raises:
        IOError: If the model file cannot be written.
    """
    try:
        with file_path.open('w', encoding='utf-8') as file:
            json.dump({'vocab': vocab, 'merges': merges}, file)
        logger.info(f"Model saved to {file_path}")
    except IOError as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def load_model(file_path: Path) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """
    Load the BPE model (vocabulary and merges) from a file.

    Args:
        file_path (Path): Path to the saved model file.

    Returns:
        Tuple[Dict[str, List[str]], List[Tuple[str, str]]]: Loaded vocabulary and merges.

    Raises:
        IOError: If the model file cannot be read.
    """
    try:
        with file_path.open('r', encoding='utf-8') as file:
            data = json.load(file)
            vocab = data['vocab']
            merges = [tuple(merge) for merge in data['merges']]
        logger.info(f"Model loaded from {file_path}")
        return vocab, merges
    except IOError as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def apply_bpe_worker(args: Tuple[List[str], List[Tuple[str, str]], BPEConfig]) -> List[List[str]]:
    """
    Worker function for parallel BPE encoding.

    Args:
        args (Tuple[List[str], List[Tuple[str, str]], BPEConfig]): 
            Tuple containing word list, merges, and config.

    Returns:
        List[List[str]]: List of encoded words.
    """
    word_list, merges, config = args
    return [encode_word(word, merges, config) for word in word_list]


def apply_bpe(word_list: List[str], model_path: Path, config: BPEConfig) -> List[List[str]]:
    """
    Apply BPE encoding to a list of words using a pre-trained model.

    Args:
        word_list (List[str]): List of words to encode.
        model_path (Path): Path to the saved BPE model.
        config (BPEConfig): Configuration object.

    Returns:
        List[List[str]]: List of encoded words.
    """
    vocab, merges = load_model(model_path)
    
    # Use multiprocessing for parallel encoding
    num_cores = config.num_cores or mp.cpu_count() // 2
    with mp.Pool(num_cores) as pool:
        try:
            chunks = [word_list[i:i + config.batch_size] for i in range(0, len(word_list), config.batch_size)]
            results = pool.map(apply_bpe_worker, [(chunk, merges, config) for chunk in chunks])
        except Exception as e:
            logger.error(f"Error during parallel BPE encoding: {str(e)}")
            pool.terminate()
            raise
    
    return [item for sublist in results for item in sublist]


def main():
    parser = argparse.ArgumentParser(description="Byte Pair Encoding operations")
    parser.add_argument("config", type=Path, help="Path to the configuration file")
    parser.add_argument("--train", action="store_true", help="Train a new BPE model")
    parser.add_argument("--apply", type=Path, help="Apply BPE to a word list file")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level")
    parser.add_argument("--num-cores", type=int, help="Number of CPU cores to use for multiprocessing")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.log_level:
        config.log_level = args.log_level
        logger.setLevel(config.log_level)

    if args.num_cores:
        config.num_cores = args.num_cores

    if args.train:
        corpus = load_corpus_from_file(config.corpus_file)
        logger.info("Training BPE model...")

        vocab, merges = byte_pair_encoding(corpus, config)
        logger.info(f"BPE completed with {len(merges)} merges")

        save_model(vocab, merges, config.model_file)

    elif args.apply:
        if not args.apply.exists():
            raise FileNotFoundError(f"Word list file {args.apply} does not exist.")
        
        with args.apply.open('r', encoding='utf-8') as file:
            word_list = file.read().split()
        
        logger.info(f"Applying BPE to {len(word_list)} words...")
        encoded_words = apply_bpe(word_list, config.model_file, config)
        
        for word, encoded in zip(word_list, encoded_words):
            print(f"'{word}': {' '.join(encoded)}")

    else:
        logger.error("Please specify either --train or --apply")
        parser.print_help()


if __name__ == "__main__":
    main()