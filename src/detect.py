import re
import logging
import string

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 设置日志格式
)
logger = logging.getLogger(__name__) # Use a specific logger


VALID_PATTERNS = [
    re.compile(r'^(?=.*[a-z])[a-z0-9]+$'), 
]


# A small set of common English words (expand as needed)
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

MIN_LENGTH_THRESHOLD = 3  
MIN_WORD_LENGTH_FOR_COMMON_CHECK = 3 
VOWEL_RATIO_THRESHOLD = 0.1 
CONSONANT_ONLY_THRESHOLD = 6 
REPETITION_THRESHOLD = 5 

# --- End: Configuration ---

def is_likely_junk_input(text: str, ents_list: list) -> bool:
    """
    Checks if the input text is likely junk, random characters, or keyboard mashing,
    while attempting to preserve specific terms and patterns (like gene names).

    Args:
        text: The input string to check.

    Returns:
        True if the input is likely junk, False otherwise.
    """
    if not text:
        logger.debug("Input is empty or None. Flagging as junk.")
        return True # Empty string is considered junk here

    original_text = text # Keep original for logging if needed
    # Normalize: lowercase and remove leading/trailing whitespace
    text = text.strip().lower()

    if not text:
        logger.debug(f"Input '{original_text}' reduced to empty string after stripping. Flagging as junk.")
        return True # String was only whitespace

    # --- Start: Exception Checks ---
    # Check if the exact text is in the allowlist
    if text in ents_list and len(text)>3:
        logger.info(f"Input '{original_text}' found in ents_list. Treating as non-junk.")
        return False

    # Check if the text matches any predefined valid patterns
    matched_a_pattern = False
    matching_pattern_str = None # Store the pattern string that matched

    # 1. Check if ANY pattern matches
    for pattern in VALID_PATTERNS:
        if pattern.fullmatch(text):
            matched_a_pattern = True
            matching_pattern_str = pattern.pattern
            # Optional: Log the specific match for debugging if needed
            # logger.debug(f"Input '{original_text}' initially matched pattern '{matching_pattern_str}'.")
            break # Found a match, no need to check other patterns

        # 2. If no pattern matched at all, it's definitely junk
        # if not matched_a_pattern:
        #     logger.info(f"Input '{original_text}' did not match any valid pattern. Treating as junk.")
        #     return True # IS junk

        if text.isalpha():
            vowels = "aeiou"
            contains_vowel = False
            for char in text.lower():
                if char in vowels:
                    contains_vowel = True
                    break
            if not contains_vowel:
                # It matched a pattern, but fails the vowel heuristic
                logger.info(f"Input '{original_text}' matched pattern '{matching_pattern_str}' but lacks vowels. Treating as junk.")
                return True # IS junk (due to heuristic)

            # 4. It matched a pattern AND passed all heuristics
            logger.info(f"Input '{original_text}' matched pattern '{matching_pattern_str}' and passed heuristics. Treating as non-junk.")
            return False # Is NOT junk
    
        if text.isalpha(): # Check if the string contains only letters
            vowels = "aeiou"
            contains_vowel = False
            for char in text.lower(): # Check in lowercase
                if char in vowels:
                    contains_vowel = True
                    break
            if not contains_vowel:
                print(f"Input '{text}' matches pattern but lacks vowels. Treating as junk.")
                return False


    if re.fullmatch(r'^[^\w\s]+$', text):
        logger.info(f"Junk detected (punctuation/symbol only): '{original_text}'")
        return True

    # 2. Check for excessive character repetition (after exceptions)
    # Looks for any character repeated REPETITION_THRESHOLD or more times
    if re.search(r'(.)\1{' + str(REPETITION_THRESHOLD - 1) + r',}', text):
        logger.info(f"Junk detected (repetition): '{original_text}'")
        return True

    # Prepare for word and character analysis
    # Using a simpler split might be okay if input is expected to be single 'words' or terms
    # words = text.split() # Simpler if multi-word common phrases aren't the main concern
    words = re.findall(r'\b\w+\b', text) # Keeps the original word extraction
    alpha_chars = [char for char in text if char.isalpha()] # Get only alphabetic characters

    if not alpha_chars:
        # Contains no letters (might be just numbers or numbers+symbols)
        if text.isdigit():
            # Decide if you want to treat number-only input as junk
            logger.info(f"Input '{original_text}' is numeric. Treating as non-junk (configurable).")
            return False # Default: Treat numbers as valid
        else:
            # Mix of numbers/symbols, no letters, didn't match allowlist/patterns
            logger.info(f"Junk detected (no alphabetic chars found and not allowlisted/pattern): '{original_text}'")
            return True

    # 3. Check for lack of common words (as one heuristic among others)
    found_common_word = False
    # Check the full text first if it's short and might be a common word itself
    if len(text) < MIN_WORD_LENGTH_FOR_COMMON_CHECK and text in COMMON_WORDS:
        found_common_word = True

    # Check individual words if not found above or text is longer
    if not found_common_word:
        for word in words:
            # Only check words of reasonable length against the common list
            if len(word) >= MIN_WORD_LENGTH_FOR_COMMON_CHECK and word in COMMON_WORDS:
                found_common_word = True
                break
            # Handle very short input: if it's short AND a common word, it's ok
            # This is somewhat redundant with the check before the loop but safe
            if len(word) < MIN_LENGTH_THRESHOLD and word in COMMON_WORDS:
                found_common_word = True
                break

    # Heuristic: If the text is short AND doesn't contain a common word
    # (and wasn't allowlisted/pattern-matched), it's likely junk.
    if not found_common_word and len(text) < MIN_LENGTH_THRESHOLD:
        logger.info(f"Junk detected (short input '{original_text}', no common words, not allowlisted/pattern).")
        return True

    # 4. Check vowel ratio (lack of vowels) - Less likely to trigger for allowlisted items
    vowels = "aeiou"
    vowel_count = sum(1 for char in alpha_chars if char in vowels)
    total_alpha = len(alpha_chars)

    if total_alpha > 0:
        vowel_ratio = vowel_count / total_alpha
        # Check for very low vowel ratio in reasonably long strings
        if vowel_ratio < VOWEL_RATIO_THRESHOLD and total_alpha > CONSONANT_ONLY_THRESHOLD:
            logger.info(f"Junk detected (low vowel ratio: {vowel_ratio:.2f} in '{original_text}', not allowlisted/pattern).")
            return True
        # Check for zero vowels in strings that have reached minimum length
        # Be careful here, acronyms/gene names might have zero vowels but passed the pattern check already
        elif vowel_count == 0 and total_alpha >= MIN_LENGTH_THRESHOLD:
            # This check might be less necessary now with allowlist/patterns,
            # but can catch things like 'rhythm' or 'crwth' if not common/allowlisted
            if not found_common_word: # Only flag if also not a common word
                logger.info(f"Junk detected (no vowels in '{original_text}', not common/allowlisted/pattern).")
                return True

    # 5. Final check: If it's reasonably long, passed other checks, but found NO common words
    # This acts as a fallback heuristic. Items that passed allowlist/pattern checks won't reach here.
    if not found_common_word and len(text) >= MIN_LENGTH_THRESHOLD:
        logger.info(f"Junk detected (input '{original_text}' has sufficient length but no common words, and wasn't allowlisted/patterned).")
        return True

    # If none of the junk conditions were met after considering exceptions
    logger.info(f"Input '{original_text}' seems valid.")
    return False