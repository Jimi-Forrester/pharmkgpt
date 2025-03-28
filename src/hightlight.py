from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from typing import List, Optional, Dict
import time
import openai
import logging

class HighlightDocuments(BaseModel):
    """Return the specific part of a document used for answering the question."""
    keyword: List[str] = Field(
        ..., description="List of key phrases from answers and question"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in 'generation' answer."""
    binary_score: str = Field(..., description="'yes' or 'no'")

    
def HightLight_context(
    input_data,
    llm_model,
    sleep_time=5,
    ):
    system = """Extract key phrases, including both full names and abbreviations, from the provided question and answer. Ensure the extraction covers all relevant concepts, entities, and relationships in a comprehensive manner.
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
    except openai.BadRequestError as e:
        logging.info(f"Too much requests, we are sleeping! \n the error is {e}")
        time.sleep(sleep_time)
        return HightLight_context(input_data=input_data)

    except openai.RateLimitError:
        logging.info("Too much requests exceeding rate limit, we are sleeping!")
        time.sleep(sleep_time)
        return HightLight_context(input_data=input_data)
        
    except OutputParserException as e:
        logging.error("Original OutputParserException: %s", e)
        pass
    
def format_docs(docs):
    """Formats a list of documents into a single string."""
    formatted_docs = ""
    for pmid, context in docs.items():
        formatted_docs += f"<doc>:\nsource: {pmid}\nContent: {context['abstract']}\n</doc>\n"
    return formatted_docs


def highlight_segments(output_abtract, lookup_response):
    keyword_list = lookup_response.keyword
    for keyword in keyword_list:
        
        for source in output_abtract:
            highlighted_context = output_abtract[source]['abstract']
            
            if keyword in output_abtract[source]['abstract']:
                logging.info(f"找到 keyword: '{keyword}', 正在执行替换...")
                output_abtract[source]['abstract'] = highlighted_context.replace(keyword, f"<mark>{keyword}</mark>")
            
            elif keyword.lower() in output_abtract[source]['abstract']:
                output_abtract[source]['abstract'] = highlighted_context.replace(keyword.lower(), f"<mark>{keyword.lower()}</mark>")
            else:
                # logging.info(f"警告: 在 {source} context 中未找到 segment: '{keyword}'。未执行替换。")
                pass
    return output_abtract





def hallucination_test(llm_model, input_data):
    sleep_time = 3
    # LLM with function call
    # Prompt
    system = """Is the 'LLM Generation' fully supported solely by the 'Retrieved Facts'? Check if all factual claims in the generation are present in the facts. Output **only** 'yes' or 'no'.

    Facts: <facts>{documents}</facts>
    Generation: <generation>{generation}</generation>
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
    except openai.BadRequestError as e:
        logging.info(f"Too much requests, we are sleeping! \n the error is {e}")
        time.sleep(sleep_time)
        return hallucination_test(input_data=input_data)

    except openai.RateLimitError:
        logging.info("Too much requests exceeding rate limit, we are sleeping!")
        time.sleep(sleep_time)
        return hallucination_test(input_data=input_data)
        
    except OutputParserException as e:
        logging.error("Original OutputParserException: %s", e)
        pass
    
