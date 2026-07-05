import os
import re
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.retrievers import BaseRetriever

# ==========================================
# 1. Load Configurations & API Keys (Class 1)
# ==========================================
load_dotenv()

# Initialize LLM
llm = ChatGroq(model_name="llama-3.3-70B-versatile")

# Helper function to create safe filenames
def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '_', text)
    return text.strip('_')

# ==========================================
# 2. Define Save Tool (Class 4)
# ==========================================
@tool
def save_video_assets(topic: str, content: str) -> str:
    """Saves the generated titles, description, hashtags, and script outline to a local markdown file."""
    filename = f"video_assets_{slugify(topic)}.md"
    with open(filename, "w") as f:
        f.write(content)
    return f"Successfully saved video assets to: {filename}"

# Map tools
tools_map = {"save_video_assets": save_video_assets}
llm_with_tools = llm.bind_tools([save_video_assets])

# ==========================================
# 3. Local Brand Guidelines Retrieval (Class 5 - RAG)
# ==========================================
style_file = "channel_style.txt"
style_docs = []
if os.path.exists(style_file):
    with open(style_file, "r") as f:
        lines = f.readlines()
        style_docs = [Document(page_content=line.strip()) for line in lines if line.strip()]

class ChannelStyleRetriever(BaseRetriever):
    docs: List[Document]
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        query_words = set(query.lower().split())
        scored = []
        for doc in self.docs:
            content_lower = doc.page_content.lower()
            # Score matches based on query terms (e.g., "title", "description", "tone")
            score = sum(1 for word in query_words if word in content_lower)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:3]]

retriever = ChannelStyleRetriever(docs=style_docs)

# ==========================================
# 4. Chat Memory Store (Class 3)
# ==========================================
history_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]

# ==========================================
# 5. Core Content Generation Loop
# ==========================================
def run_generator_turn(session_id: str, user_input: str, topic: str = "") -> str:
    # Get chat history for the user session
    chat_history = get_session_history(session_id)
    
    # 1. RAG step: Retrieve brand rules matching the user input or the topic
    search_query = topic if topic else user_input
    matched_rules = retriever.invoke(search_query)
    context_guidelines = "\n".join([doc.page_content for doc in matched_rules])
    
    # 2. System prompt enforcing structure, role, and guidelines
    system_prompt = (
        "You are YT-Sage, an expert YouTube content producer and SEO specialist.\n"
        "Generate creative, engaging, and high-performing YouTube assets.\n"
        "Your responses must strictly follow these channel brand guidelines:\n"
        f"{context_guidelines}\n"
        "------------------------------------\n"
        "Output format structure should always be:\n"
        "# VIDEO TOPIC\n"
        "## Suggested Titles\n"
        "## Video Description\n"
        "## Hashtags\n"
        "## Video Script Outline (Minute-by-Minute)"
    )
    
    # Construct complete message chain
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(chat_history.messages)
    messages.append(HumanMessage(content=user_input))
    
    # 3. Call LLM with tool bound
    response = llm_with_tools.invoke(messages)
    
    # 4. Handle tool execution (saving assets file)
    if response.tool_calls:
        # Append human input and assistant tool call request to history
        chat_history.add_message(HumanMessage(content=user_input))
        chat_history.add_message(response)
        
        for tool_call in response.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            t_id = tool_call["id"]
            
            print(f"\n[Tool Run] AI running '{t_name}' to save markdown assets locally...")
            
            executable_tool = tools_map.get(t_name)
            if executable_tool:
                tool_output = executable_tool.invoke(t_args)
                print(f"[Tool Response] {tool_output}")
                
                # Append tool result to messages and get the final LLM summary/answer
                tool_msg = ToolMessage(content=tool_output, tool_call_id=t_id)
                messages.append(response)
                messages.append(tool_msg)
                chat_history.add_message(tool_msg)
                
        final_response = llm.invoke(messages)
        chat_history.add_message(final_response)
        return final_response.content
    else:
        # Save messages to conversation history
        chat_history.add_message(HumanMessage(content=user_input))
        chat_history.add_message(response)
        return response.content

# ==========================================
# 6. Interactive Shell Execution
# ==========================================
if __name__ == "__main__":
    session_id = "youtube_generator_session"
    print("====================================================")
    print("Welcome to YT-Sage: LangChain YouTube Content Creator")
    print("====================================================")
    
    # Ask for video topic to kick-start generation
    video_topic = input("Enter your video topic / idea: ").strip()
    if not video_topic:
        video_topic = "How to learn LangChain in 2026"
        print(f"No topic entered. Defaulting to: {video_topic}")
        
    initial_prompt = (
        f"Generate video titles, description, hashtags, and a detailed script outline "
        f"for a video about: '{video_topic}'."
    )
    
    print("\n[Thinking...] Generating assets and checking channel style guidelines...")
    initial_output = run_generator_turn(session_id, initial_prompt, topic=video_topic)
    
    print("\n--- Generated Video Assets ---")
    print(initial_output)
    
    # Automatically suggest saving the generated content using tool
    print("\n--- Automatic File Saving ---")
    save_trigger = f"Save the generated video assets for the topic '{video_topic}': {initial_output}"
    save_output = run_generator_turn(session_id, save_trigger, topic=video_topic)
    print(save_output)
    
    print("\n====================================================")
    print("Interactive editor active. Chat to refine the script or titles.")
    print("Type 'exit' to quit.")
    print("====================================================")
    
    while True:
        try:
            chat_input = input("\nYou: ").strip()
            if not chat_input or chat_input.lower() == "exit":
                print("Goodbye!")
                break
                
            reply = run_generator_turn(session_id, chat_input)
            print(f"\nYT-Sage: {reply}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
