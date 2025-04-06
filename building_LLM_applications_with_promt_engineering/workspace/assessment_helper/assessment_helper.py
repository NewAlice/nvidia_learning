import json

from typing import List
from pprint import pprint

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.pydantic_v1 import BaseModel, Field

base_url = 'http://llama:8000/v1'
model = 'meta/llama-3.1-8b-instruct'
llm = ChatNVIDIA(base_url=base_url, model=model, temperature=0)

with open('/workspace/dli/6-Assessment/data/emails.json', 'r') as f:
        emails = json.load(f)

class SummaryDetails(BaseModel):
    """Product class and location mentioned in a document."""
    product_class: str = Field(description="A class or category of product type mentioned in the document.")
    city: str = Field(description="A city name mentioned in the document.")

summary_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI that generates JSON and only JSON according to the instructions provided to you."),
    ("human", (
        "Generate JSON about the user input according to the provided format instructions.\n" +
        "Input: {input}\n" +
        "Format instructions {format_instructions}")
    )
])

summary_parser = JsonOutputParser(pydantic_object=SummaryDetails)
summary_format_instructions = summary_parser.get_format_instructions()
summary_template_with_format_instructions = summary_template.partial(format_instructions=summary_format_instructions)
summary_details_chain = summary_template_with_format_instructions | llm | summary_parser

def run_assessment(chain):
    correct_product_class = False
    correct_city = False
    
    print('Passing emails into your chain...')
    student_results = chain.invoke(emails)
    print('Your chain completed successfully.\n')
    
    print('Checking whether your chain\'s summary correctly identified the product class with the most negative sentiments...')
    student_result_details = summary_details_chain.invoke(student_results)
    if student_result_details['product_class'].lower() == 'furniture':
        print('Your chain\'s summary correctly identified the product class with the most negative sentiment.\n')
        correct_product_class = True
    else:
        print('Your chain\'s summary did NOT correctly identify the product class with the most negative sentiment.\n')

    print('Checking whether your chain\'s summary correctly identified the location with the most negative sentiments...')
    if student_result_details['city'].lower() == 'new york':
        print('Your chain\'s summary correctly identified the city with the most negative sentiment.\n')
        correct_city = True
    else:
        print('Your chain\'s summary did NOT correctly identify the city with the most negative sentiment.\n')

    if correct_product_class and correct_city:
        open('/workspace/assessment_results/PASSED', 'w')
        print('You successfully completed the assessment, congrats! Please see below for instructions on how to generate your certificate')
    else:
        print('You did not successfully complete the assessment, please continue your work and try again.')
