import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from knowledge_base import ask_doordash_question
import logging

# Load environment variables
load_dotenv(override=True)

TOOLS_REGISTRY = {}

SYSTEM_PROMPT = """
                        You are DoorDashGPT — a highly knowledgeable DoorDash support assistant.

                        Your role:
                        - Answer any question about DoorDash policies, procedures, delivery issues, account setup, payments, refunds, etc.
                        - Always base your responses on the indexed documentation.
                        - Be concise, accurate, and customer-friendly.
             
                        # VERY IMPORTANT:
                        - When passing on the query to the knowledge base, 
                        always use the exact user query as input to the knowledge base tool or little tweaks to it if absolutely necessary.
                        - Never modify the user's intent or rephrase it significantly.
                        - Do not pass any internal instructions or context to the knowledge base.
                        - Don't shorten a query just to fit it in the tool input.
                        - If the user query is already very short, use it as-is.


                        Critical rules:
                        1. **Always include source links**  
                        - At the end of every response, list the `source_link(s)` that contain the relevant information.
                        - Use a clear section:  
                            **Sources:** [Source Link](https://example.com)

                        2. **Include page links when relevant**  
                        - If the document metadata contains `page_links` and they are related to the answer:
                            - Embed them naturally in the response.
                            - Example:  
                            "To update your payout method, follow [these instructions](https://help.doordash.com/account/payouts)."
                        - If no relevant `page_links` exist, skip them.

                        3. **Never make up information**  
                        - If the answer isn't found in the documentation, say:  
                            > "I couldn’t find specific information on that in the DoorDash documentation."

                        4. **Response structure**:
                        - Friendly, clear tone.
                        - Provide step-by-step answers if needed.
                        - Use bullet points for instructions.
                        - Always finish with sources.

                        Example:

                        **User:** "How do I update my DoorDash payment method?"  
                        **Assistant:**  
                        To update your payment method on DoorDash:  
                        1. Open the DoorDash app and go to **Account Settings**.  
                        2. Tap **Payment Methods**.  
                        3. Add a new payment option or update an existing one.  

                        For more details, follow [these official steps](https://help.doordash.com/account/payment).  

                        **Sources:** [DoorDash Help Center](https://help.doordash.com)

                        ---

                        Behavior:
                        - Always prioritize correctness over speed.
                        - Be confident but never invent unsupported answers.
                        - If unsure, just say you don't know and ask if they would like to something else.
                        - Always be polite and respectful in your responses.
                        - Never answer questions unrelated to DoorDash even if its conversational just tell the user what your 
                        role is and that you can only answer questions related to DoorDash.

                    """
def register_tool(func):
    """Decorator to automatically register tools."""
    TOOLS_REGISTRY[func.__name__] = func
    return func

@tool
@register_tool
def _knowledge_base_query(query: str) -> str:
    """Queries the DoorDash knowledge base. 
    Always use this tool to answer questions related to DoorDash policies, procedures, and FAQs.
    """
    return ask_doordash_question(query)

class ChatbotGraph:
    def __init__(self, thread_id="1"):
        """
        Initializes the chatbot graph with tools, memory, and state management.
        """

                # Setup logger for error tracking
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )


        # Get API key from env vars
        groq_key = os.getenv("GROQ_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        if not groq_key and not google_key:
            raise EnvironmentError(
                "Missing required environment variables. Set either GROQ_API_KEY or GOOGLE_API_KEY."
            )

        # Initialize LLM (using Google Gemini by default)
        # You can switch to Groq if needed by uncommenting below:
        # self.llm = init_chat_model("openai/gpt-oss-20b", model_provider="groq", temperature=0.0)
        self.llm = init_chat_model("models/gemini-2.5-flash", model_provider="google_genai", temperature=0.5)

        # Register tools
        self.tools = list(TOOLS_REGISTRY.values())
        self.llm = self.llm.bind_tools(self.tools)


        # Setup memory
        self.memory = InMemorySaver()

        # Define chatbot state
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        self.State = State

        # Build the state graph
        self.builder = StateGraph(self.State)
        self.builder.add_node("Chatbot", self._chatbot)
        self.builder.add_node("tools", ToolNode(tools=self.tools))

        # Add graph edges
        self.builder.add_edge(START, "Chatbot")
        self.builder.add_conditional_edges("Chatbot", tools_condition, {"tools": "tools", END: END})
        self.builder.add_edge("tools", "Chatbot")
        self.builder.add_edge("Chatbot", END)

        # Compile the graph with memory support
        self.graph = self.builder.compile(checkpointer=self.memory)

        # Thread configuration
        self.config = {"configurable": {"thread_id": thread_id, "max_iterations": 3, "recursion_limit": 5, "reasoning": "False"}}
   


    def _chatbot(self, state: dict) -> dict:
        messages = state["messages"]
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])
        response = self.llm.invoke(prompt.format(messages=messages))
        return {"messages": [response]}


    def ask(self, message: str) -> str:
        """
        Sends a message to the chatbot graph and returns the response.
        """
        from langchain_core.messages import HumanMessage

        try:
            result = self.graph.invoke({"messages": [HumanMessage(content=message)]}, config=self.config)

            return result["messages"][-1].content
        except Exception as e:
            self.logger.error(f"Graph invocation failed: {e}")
            return "Something went wrong. Please try again later."
        
    def ask_stream(self, message: str):
        """
        Streams chatbot responses character by character instead of returning full reply at once.
        Yields tokens progressively.
        """
        from langchain_core.messages import HumanMessage

        try:
            # Start streaming response from LangGraph
            for event in self.graph.stream(
                {"messages": [HumanMessage(content=message)]},
                config=self.config
            ):

                for value in event.values():
                    # Yield character by character for smoother typing
                    for char in value["messages"][-1].content:
                        yield char
        except Exception as e:
            self.logger.error(f"Streaming failed: {e}")
            yield "\n**Error:** Something went wrong. Please try again later."
