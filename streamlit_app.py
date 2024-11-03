import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

with st.form("my_form"):
    st.write("Please enter your values")
    user_prompt = st.text_input("Tell me about your most recent air travel experience.")
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        llm = ChatOpenAI(api_key=st.secrets['MyOpenAIKey'], model="gpt-4o")
        ### Create a template to handle the case where the price is not mentioned.
        sentiment_template = """You are an expert at airline customer service.
        From the following text, determine whether the customer had a positive or negative experience.

        Do not respond with more than one word.

        Text:
        {request}

        """


        ### Create the decision-making chain

        sentiment_type_chain = (
            PromptTemplate.from_template(sentiment_template)
            | llm
            | StrOutputParser()
        )
        fault_template = """You are an expert at our airline customer service.
        From the following text, determine whether our airline is at fault or not for the customer's poor experience. 
        If our airline is at fault, say "yes". If our airline is not at fault, say "no".

        Your response should follow these guidelines:
            1. Determine our airline is at fault for any issues regarding lost luggage, mechanical delays, or crew delays.
            2. Determine our airline is at fault for any issues related to food. 
            3. Determine our airline is not at fault for any issues regarding weather delays.

        Do not respond with more than one word.

        Text:
        {request}

        """


        ### Create the decision-making chain

        fault_type_chain = (
            PromptTemplate.from_template(fault_template)
            | llm
            | StrOutputParser()
        )
        positive_chain = PromptTemplate.from_template(
        """You are a customer service agent. Thank the customer for their feedback and for choosing to fly with the airline.
        Do not respond with any reasoning. Just respond professionally as a customer service agent. Respond in first-person mode and address the customer directly.



        Text:
        {text}

        """
        ) | llm


        yes_fault_chain = PromptTemplate.from_template(
            """You are a customer service agent. Given the text below, display a message
            apologizing for the poor experience the customer has experienced while 
            flying on our airline and inform the customer that customer service will 
            contact them soon to resolve the issue or provide compensation.

            Your response should follow these guidelines:
            1. Do not respond with any reasoning. Just respond professionally.
            2. Address the customer directly

        Text:
        {text}

        """
        ) | llm

        no_fault_chain = PromptTemplate.from_template(
            """You are a customer service agent for our airline.
            Given the text below, offer sympathies to the customer but explain that the 
            airline is not liable for situations out of our control like this.

            Your response should follow these guidelines:
            1. Do not respond with any reasoning. Just respond professionally as a travel chat agent.
            2. Address the customer directly

        Text:
        {text}

        """
        ) | llm
        from langchain_core.runnables import RunnableBranch

        ### Routing/Branching chain
        # ANDREW: the syntax here is for a conditional: "if the flight type is international then execute international_chain, if not execute general_chain"
        # ANDREW: the tricky part of the assignment is figuring out how to make a 3rd banch similar to this 2 branch example here
        branch = RunnableBranch(
            (lambda x: "positive" in x["sentiment_type"].lower(), positive_chain),
            (lambda x: "yes" in x["fault_type"].lower(), yes_fault_chain), no_fault_chain
        )

        ### Put all the chains together
        #ANDREW: this takes the flight type determined by flight_type_chain, and the text from the request and feeds both of those into the branch mechanism
        full_chain = {"sentiment_type": sentiment_type_chain, "fault_type": fault_type_chain, "text": lambda x: x["request"]} | branch
        #ANDREW: turning debug to True is a helpful way to see more info, which model is being used, etc.
        import langchain
        langchain.debug = False

        ai_response = (full_chain.invoke({"request": str(user_prompt)}))
        st.write(ai_response[0])