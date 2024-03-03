from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo"
)

def call_string_output_parser():
    # Prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Tell me a joke about the following subject"),
            ("human", "{input}")
        ]
    )

    parser = StrOutputParser()

    # Create LLM chain
    chain = prompt | llm | parser

    return chain.invoke({"input": "dog"})

def call_list_output_parser(): 
    # Prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate a list of ten synonyms for the following word. Return the results as a comma-separated list."),
            ("human", "{input}")
        ]
    )

    parser = CommaSeparatedListOutputParser()

def call_json_output_parser():
    # Prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Extract information from the following phrase. \nFormatting Instructions: {format_instructions} "),
            ("human", "{phrase}")
        ]
    )

    class Person(BaseModel):
        name: str = Field(description="The name of the person")
        age: int = Field(description="The age of the person")

    parser = JsonOutputParser(pydantic_object=Person)

    # Create LLM chain
    chain = prompt | llm | parser

    return chain.invoke({
        "phrase": "Max is will be 30 years old in 8 months time",
        "format_instructions": parser.get_format_instructions()
    })

print(call_json_output_parser())  