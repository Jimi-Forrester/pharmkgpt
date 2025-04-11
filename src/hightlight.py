from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict, Set, Tuple
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from typing import List
import time
import logging
import re

class HighlightError(Exception):
    pass

class HighlightDocuments(BaseModel):
    """Return the specific part of a document used for answering the question."""
    keyword: List[str] = Field(
        ..., description="List of key phrases from answers and question"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in 'generation' answer."""
    Faithfulness_score: int = Field(..., description="Faithfulness_score like 1, 2, 3, 4, 5")

    
def hightLight_context(
    input_data,
    llm_model,
    sleep_time=5,
    ):
    system = """Extract key **terms (words or short phrases)** from the provided question and answer that are essential to understanding their meaning.
    Used documents: User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>
    
    <format_instruction>
    {format_instructions}
    </format_instruction>
    """

    parser = PydanticOutputParser(pydantic_object=HighlightDocuments)

    prompt = PromptTemplate(
        template=system,
        input_variables=["documents", "question", "generation"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    

    chain = prompt | llm_model | parser
    
    try:
        return chain.invoke(input_data)
    except HighlightError as e:
        logging.info(f"Too much requests, we are sleeping! \n the error is {e}")
        time.sleep(sleep_time)
        return hightLight_context(input_data=input_data)

    except HighlightError:
        logging.info("Too much requests exceeding rate limit, we are sleeping!")
        time.sleep(sleep_time)
        return hightLight_context(input_data=input_data)
        
    except OutputParserException as e:
        logging.error("Original OutputParserException: %s", e)
        pass
    
def format_docs(docs):
    """Formats a list of documents into a single string."""
    formatted_docs = ""
    for pmid, context in docs.items():
        formatted_docs += f"<doc>:\nsource: {pmid}\nContent: {context['abstract']}\n</doc>\n"
    return formatted_docs


def detect_all_entity_name(pmid_list: List, kg_dict: Dict[str, Dict]) -> Dict:
    keyword_dict = {}
    for pmid in pmid_list:
        for en in kg_dict[pmid]['entities']:
            if en.label != "abstract":
                keyword_dict[en.name] = en.label
    return keyword_dict


def highlight_segments_prioritized(output_abstract, keyword_dict):
    """
    Highlights keywords in abstract texts, robustly prioritizing longer keywords
    over shorter ones when they overlap (e.g., 'kynurenine pathway' vs 'kynurenine').
    Avoids nested highlighting.

    Args:
        output_abstract (dict): A dictionary where keys are source identifiers
                                and values are dicts containing an 'abstract' string.
                                Example: {'source1': {'abstract': 'Text here...'}, ...}
        keyword_dict (dict): A dictionary mapping keywords to their group/class.
                             Example: {'kynurenine pathway': 'pathway', 'kynurenine': 'chemical'}

    Returns:
        dict: The modified output_abstract dictionary with keywords highlighted correctly.
    """
    # Process each source abstract individually
    for source, data in output_abstract.items():
        original_text = data['abstract']
        
        # 1. Find all potential matches with their positions and info for ALL keywords
        all_potential_matches = []
        logging.debug(f"--- Processing source: {source} ---")
        for keyword, group in keyword_dict.items():
            if len(keyword) <= 1:
                continue
            try:
                pattern = re.compile(rf'\b({re.escape(keyword)})\b', re.IGNORECASE)
                for match in pattern.finditer(original_text):
                    match_info = {
                        'start': match.start(1), # Start index of the matched keyword
                        'end': match.end(1),     # End index of the matched keyword
                        'keyword': match.group(1), # The actual matched string (original case)
                        'group': group,
                        'length': len(match.group(1)) # Length of the keyword
                    }
                    all_potential_matches.append(match_info)
                    logging.debug(f"Found potential match: {match_info}")
            except re.error as e:
                logging.error(f"Regex error for keyword '{keyword}' in source '{source}': {e}")
                continue

        if not all_potential_matches:
            logging.debug(f"No potential matches found in {source}")
            continue # No keywords found in this abstract

        sorted_matches = sorted(all_potential_matches, key=lambda x: (-x['length'], x['start']))
        logging.debug(f"\nSorted potential matches for {source}:")
        for m in sorted_matches: logging.debug(m)


        selected_matches = []
        # Keep track of character indices that are already covered by a selected (longer) match
        covered_indices = set()

        logging.debug(f"\nFiltering overlaps for {source}:")
        for match in sorted_matches:
            match_indices = set(range(match['start'], match['end']))
            if not match_indices.intersection(covered_indices):
                selected_matches.append(match)
                covered_indices.update(match_indices)
                logging.debug(f"  -> Selecting match: {match}")
                logging.debug(f"     Covered indices now: {covered_indices}")
            else:
                logging.debug(f"  -> Discarding match (overlaps with selected longer): {match}")

        selected_matches.sort(key=lambda x: x['start'])
        logging.debug(f"\nFinal selected matches for highlighting (sorted by start):")
        for m in selected_matches: logging.debug(m)


        highlighted_parts = []
        last_end = 0
        for match in selected_matches:
            # Append text before the current match
            highlighted_parts.append(original_text[last_end:match['start']])
            # Append the highlighted keyword
            highlighted_parts.append(f'<mark class="{match["group"]}">{match["keyword"]}</mark>')
            # Update the position after the current match
            last_end = match['end']

        # Append any remaining text after the last match
        highlighted_parts.append(original_text[last_end:])

        # Join the parts to form the final highlighted abstract
        final_highlighted_text = "".join(highlighted_parts)

        if final_highlighted_text != original_text:
             logging.info(f"在 '{source}' 中完成高亮处理 (长优先, 无重叠)")
             output_abstract[source]['abstract'] = final_highlighted_text
        else:
             logging.debug(f"No changes made to {source} after highlighting logic.")
        logging.debug(f"--- Finished processing source: {source} ---\n")


    return output_abstract


def highlight_segments(output_abstract, keyword_dict):
    """_summary_

    Args:
        output_abstract (_type_): _description_
        keyword_dict (_type_): {entity:group,entity:group}

    Returns:
        _type_: _description_
    """
    for keyword, group in keyword_dict.items():
        if len(keyword) <= 1:
            continue
        # 构造正则，忽略大小写，匹配整词（可选）
        pattern = re.compile(rf'(?<!<mark>)(?<!</mark>)(\b{re.escape(keyword)}\b)', re.IGNORECASE)

        for source in output_abstract:
            original_text = output_abstract[source]['abstract']
            def replace_func(match):
                # 获取第一个捕获组匹配到的文本 (即原始大小写的关键词)
                matched_keyword = match.group(1)
                # 使用 f-string 和外部变量 group 构建替换字符串
                return f'<mark class="{group}">{matched_keyword}</mark>'
            
            highlighted = pattern.sub(replace_func, original_text)
            if highlighted != original_text:
                logging.info(f"在 '{source}' 中找到并高亮 keyword: '{keyword}' with class '{group}'")
                output_abstract[source]['abstract'] = highlighted
            
            # if re.search(pattern, original_text):
            #     logging.info(f"找到 keyword: '{keyword}', 正在执行替换...")

            #     # 替换匹配内容，保留原始大小写
                
            #     highlighted = pattern.sub(r'f"<mark class="{group}">\1</mark>"', original_text)
            #     output_abstract[source]['abstract'] = highlighted

    return output_abstract





def hallucination_test(llm_model, input_data):
    sleep_time = 3
    # LLM with function call
    # Prompt
    system = """# Role
    You are a meticulous evaluator assessing whether an answer generated by a Large Language Model (LLM) is factually grounded in the provided background information (context).

    # Task
    Your task is to determine if the information presented in the 'Generated Answer' can be sufficiently supported by the 'Retrieved Context'. You must also consider the relevance of the answer to the 'Original Query'. Provide a hallucination score, where a higher score indicates the answer is more likely fabricated or unsupported by the context (hallucinated).

    # Input Information
    Please evaluate the 'Generated Answer' strictly based **only** on the provided 'Retrieved Context'. Do not use any external knowledge.

    ## Retrieved Context: 
    <docs>{documents}</docs>
    
    ## Generated Answer:
    <generation>{generation}</generation>

    # Evaluation Criteria & Scoring Guide
    Evaluate based on the following criteria and provide an integer score from 1 to 5:

    *   **Score 1 (Fully Grounded & Relevant):** All key information and logical connections presented in the answer are explicitly supported by the 'Retrieved Context'. This includes answers that accurately **synthesize multiple pieces of information explicitly present** in the context to directly address the query's core question. The answer is highly relevant to the 'Original Query'.
    *   **Score 2 (Mostly Grounded & Relevant):** The majority of key information is supported by the context. May involve minor, **reasonable paraphrasing or logical synthesis clearly based on information stated in the context**. No introduction of core facts or strong causal claims not directly supported by the building blocks within the context. Generally relevant to the query.
    *   **Score 3 (Partially Grounded / Mild Hallucination):** The answer contains a mix of information supported by the context and details or inferences not found there. Alternatively, the answer might be based on the context but has low relevance to the 'Original Query'.
    *   **Score 4 (Largely Hallucinated / Poorly Grounded):** Most of the key information in the answer cannot be verified from the 'Retrieved Context'. It appears largely fabricated by the model. This also applies if the context is irrelevant to the query, but the answer attempts to derive meaning from it anyway.
    *   **Score 5 (Completely Hallucinated / Fabricated / Irrelevant):** The information in the answer finds almost no support in the 'Retrieved Context'. This includes cases where the context is empty, but the answer provides specific information, or the answer is completely unrelated to both the context and the query.

    ***Specific Handling Notes:**
    *   If the 'Retrieved Context' is **empty** or **completely irrelevant**... **high score (4 or 5)**.
    *   If the 'Original Query' itself is **nonsensical**... **high score (4 or 5)**.
    *   **[REVISED RULE v4] Handling Answers Discussing Information Availability & Synthesis:**
        *   If the 'Generated Answer' explicitly states that the **core subject** or **essential information** needed to answer the 'Original Query' is **missing** or **not mentioned** in the 'Retrieved Context', AND the answer subsequently provides **no relevant, grounded information** about the query's topic from the context, assign **Score 5**.
        *   However, if the 'Generated Answer' accurately notes the absence of *specific details*, a *direct causal statement*, or uses cautious phrasing (e.g., "appears to be," "likely contributes," "suggests a role") reflecting the nuance or limitations of the context, **this is NOT grounds for a high score by itself.**
        *   Furthermore, if the answer **synthesizes** information by logically connecting **multiple facts that are *each individually supported* by the provided 'Retrieved Context'** to answer the query, this should be considered **grounded (Scores 1-2)**, provided the synthesis itself is logical and doesn't introduce unsupported claims. **Do NOT assign Score 5 simply because the synthesized conclusion wasn't stated verbatim as a single sentence in the context.** Evaluate the *accuracy and grounding of the synthesis* based on the context.
    Output format:
        <format_instruction>
        {format_instructions}
        <format_instruction>

    """
    parser = PydanticOutputParser(pydantic_object=GradeHallucinations)

    prompt = PromptTemplate(
        template=system,
        input_variables=["documents", "generation"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm_model | parser
    
    try:
        return chain.invoke(input_data)
    except HighlightError as e:
        logging.info(f"Too much requests, we are sleeping! \n the error is {e}")
        time.sleep(sleep_time)
        return hallucination_test(input_data=input_data)

    except HighlightError:
        logging.info("Too much requests exceeding rate limit, we are sleeping!")
        time.sleep(sleep_time)
        return hallucination_test(input_data=input_data)
        
    except OutputParserException as e:
        logging.error("Original OutputParserException: %s", e)
        pass
    
def format_scientific_text_with_colon(text):
    # List of keywords to check
    keywords = ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSION"]
    pattern = r"(" + "|".join(keywords) + r")(?=[A-Z])"

    def replace_func(m):
        return m.group(1) + ": "

    formatted_text = re.sub(pattern, replace_func, text)

    return formatted_text