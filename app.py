import os
from apikey import apikey
import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# make key available for openai service
os.environ['openai_api_key']=apikey

# Framework for da App 
st.title('🥸 Content Creator 🥸')
prompt=st.text_input('Add your prompt and press ENTER to run or press R to rerun') 

# template
title_template=PromptTemplate(
    input_variables=['topic'],
    template='give me a very creative title on the topic of {topic}'
)
script_template=PromptTemplate(
    input_variables=['title','wikipedia_research'],
    template='give me a youtube video script based on this title TITLE:{title} while utilizing this wikipedia research:{wikipedia_research}'
)

# memory to store history
title_memory=ConversationBufferMemory(input_key='topic',memory_key='chat_history')
script_memory=ConversationBufferMemory(input_key='title',memory_key='chat_history')


# connecting llms
llm=OpenAI(temperature=0.9)
# chains
title_chain=LLMChain(llm=llm,prompt=title_template,verbose=True,output_key='title',memory=title_memory)
script_chain=LLMChain(llm=llm,prompt=script_template,verbose=True,output_key='script',memory=script_memory)
# bunching them together in seq(order)
# sequential_chain=SequentialChain(chains=[title_chain,script_chain],input_variables=['topic'],output_variables=['title','script'],verbose=True) 
wiki=WikipediaAPIWrapper()

if prompt:
    # response=sequential_chain({'topic':prompt})
    title=title_chain.run(prompt)
    wikipedia_research=wiki.run(prompt)
    script=script_chain.run(title=title,wikipedia_research=wikipedia_research)
    # pass back the response to the screen
    st.write(title)
    st.write(script)

    with st.expander('Title history'):
        st.info(title_memory.buffer)

    with st.expander('Script history'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia research history'):
        st.info(wikipedia_research)