import re
import logging
import string

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 设置日志格式
)


# --- Start: Configuration ---

# A small set of common English words (expand as needed)
# You could load this from a file for a larger list
COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "person", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only",
    "come", "its", "over", "think", "also", "back", "after", "use",
    "two", "how", "our", "work", "first", "well", "way", "even", "new",
    "want", "because", "any", "these", "give", "day", "most", "us",
    "is", "are", "was", "were", "has", "had", "why", "where", "help"
}

consonants = "bcdfghjklmnpqrstvwxyz"


MIN_LENGTH_THRESHOLD = 3 # Treat inputs shorter than this as potentially junk unless a common word
MIN_WORD_LENGTH_FOR_COMMON_CHECK = 3 # Only check words of this length or more against COMMON_WORDS
VOWEL_RATIO_THRESHOLD = 0.1 # If vowel ratio is below this (and long enough), likely junk
CONSONANT_ONLY_THRESHOLD = 5 # Check strings longer than this for being all consonants
REPETITION_THRESHOLD = 5 # e.g., 'aaaaa' triggers this

# --- End: Configuration ---

def is_likely_junk_input(text: str) -> bool:
    """
    Checks if the input text is likely junk, random characters, or keyboard mashing.

    Args:
        text: The input string to check.

    Returns:
        True if the input is likely junk, False otherwise.
    """
    if not text:
        return True # Empty string is considered junk here

    # Normalize: lowercase and remove leading/trailing whitespace
    text = text.strip().lower()

    if not text:
        return True # String was only whitespace


    if re.fullmatch(r'^[^\w\s]+$', text):
        logging.info(f"Junk detected (punctuation/symbol only): '{text}'")
        return True

    # 3. Check for excessive character repetition
    # Looks for any character repeated REPETITION_THRESHOLD or more times
    if re.search(r'(.)\1{' + str(REPETITION_THRESHOLD - 1) + r',}', text):
        logging.info(f"Junk detected (repetition): '{text}'")
        return True

    # Prepare for word and character analysis
    words = re.findall(r'\b\w+\b', text) # Extract words (sequences of letters/numbers)
    alpha_chars = [char for char in text if char.isalpha()] # Get only alphabetic characters

    if not alpha_chars:
        # Contains no letters (might be just numbers or numbers+symbols)
        # Decide if you want to treat number-only input as junk
        if text.isdigit():
             logging.info(f"Input is numeric: '{text}' - Treating as non-junk (configurable)")
             return False # Or return True if you want to block numbers only
        else:
             logging.info(f"Junk detected (no alphabetic chars found): '{text}'")
             return True # Mix of numbers/symbols, likely junk

    # 4. Check for lack of common words
    found_common_word = False
    for word in words:
        # Only check words of reasonable length against the common list
        if len(word) >= MIN_WORD_LENGTH_FOR_COMMON_CHECK and word in COMMON_WORDS:
            found_common_word = True
            break
        # Handle very short input: if it's short AND a common word, it's ok
        if len(text) < MIN_LENGTH_THRESHOLD and word in COMMON_WORDS:
             found_common_word = True
             break

    # If the text is reasonably long OR short, but contains NO common words, flag it
    # Exception: Acronyms might fail this. Add more heuristics if needed.
    if not found_common_word and len(text) >= MIN_LENGTH_THRESHOLD :
         # Before flagging, do a vowel check as a fallback
         pass # Fall through to vowel check
    elif not found_common_word and len(text) < MIN_LENGTH_THRESHOLD:
        logging.info(f"Junk detected (short input, no common words): '{text}'")
        return True

    # 5. Check vowel ratio (lack of vowels)
    vowels = "aeiou"
    vowel_count = sum(1 for char in alpha_chars if char in vowels)
    total_alpha = len(alpha_chars)

    if total_alpha > 0:
        vowel_ratio = vowel_count / total_alpha
        # Check for very low vowel ratio in reasonably long strings
        if vowel_ratio < VOWEL_RATIO_THRESHOLD and total_alpha > CONSONANT_ONLY_THRESHOLD:
            logging.info(f"Junk detected (low vowel ratio: {vowel_ratio:.2f}): '{text}'")
            return True
        # Check for zero vowels in slightly shorter strings
        elif vowel_count == 0 and total_alpha >= MIN_LENGTH_THRESHOLD:
            # Double-check if it contained a common word - might be an acronym like 'rpg'
            if not found_common_word:
                logging.info(f"Junk detected (no vowels): '{text}'")
                return True

    # 6. Final check: If we passed all heuristics but found no common words (re-check)
    if not found_common_word and len(text) >= MIN_LENGTH_THRESHOLD:
        logging.info(f"Junk detected (no common words found after other checks): '{text}'")
        return True

    # If none of the junk conditions were met
    logging.info(f"Input seems valid: '{text}'")
    return False