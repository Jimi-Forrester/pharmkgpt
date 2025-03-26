from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from typing import List, Optional, Dict
import time
import openai
import logging


class Segment(BaseModel):
    segment: Optional[List[str]] = Field(None, description="List of direct segments from used documents that answers the question")


class HighlightDocuments(BaseModel):
    """Return the specific part of a document used for answering the question."""
    segment_list: List[Dict[str, List[str]]] = Field(
        ..., description="List of pmid"
    )



def HightLight_context(
    input_data,
    llm_model,
    sleep_time=5,
    ):


    system = """You are an advanced assistant for document search and retrieval. You are provided with the following:
    Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to 
    generate the given answer. The extracted segments must be verbatim snippets from the provided documents, ensuring a word-for-word match with the text 
    in the provided documents.

    For EACH document, provide the document's pmid, and the VERBATIM segments that support the answer.

    Ensure that:
    - (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
    - The relevance of each segment to the generated answer is clear and directly supports the answer provided.
    - (Important) If a specific document wasn't used to formulate the answer, then do NOT include it in the output.

    Output Format:  A list of dictionaries.  Each dictionary represents a document that contributed to the answer.  The keys of each dictionary MUST be: "pmid", and "segments". The "segments" key holds a LIST of strings, where each string is a verbatim segment from that document.

    Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>
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
        
    except OutputParserException:
        logging.info("Error in parsing the instance!")
        pass
    
def format_docs(docs):
    """Formats a list of documents into a single string."""
    formatted_docs = ""
    for pmid, context in docs.items():
        formatted_docs += f"pmid: {pmid}\n"
        formatted_docs += f"Document Text: {context['abstract']}\n\n"
    return formatted_docs


def highlight_segments(output_abtract, highlight_list):
    try:
        highlight_list = highlight_list.segment_list
    except:
        return output_abtract

    print(highlight_list)
    if type(highlight_list) is list and len(highlight_list) > 0 and type(highlight_list[0]) == dict:
        for highlight_dict in highlight_list:
            for pmid, segments in highlight_dict.items():
                if pmid in output_abtract:
                    highlighted_context = output_abtract[pmid]['abstract']
                    for segment in segments:
                        output_abtract[pmid]['abstract'] = highlighted_context.replace(segment, f"<mark>{segment}</mark>")
    return output_abtract