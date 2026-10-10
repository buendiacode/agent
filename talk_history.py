from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import WikipediaRetriever

chat = ChatOpenAI(model_name="gpt-4", temperature=0)

tools = []

retriever = WikipediaRetriever(
    language="en", top_k=5, return_source_documents=True
)

tools.append(
    create_retriever_tool(
        retriever=retriever,
        name="Wikipedia",
        description="useful for when you need to answer questions about Wikipedia articles",
    )
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)

result_1 = agent.run("What is the capital of France?")
print(result_1)

result_2 = agent.run("What is the population of France?")
print(result_2)