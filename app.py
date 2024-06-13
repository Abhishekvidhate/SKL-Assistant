import streamlit as st
from src.rag import create_rag_chain

# Path to your markdown file
markdown_path = "data/skldata.md"

# Load a logo image (Ensure you have the logo image in the specified path)
logo_path = "data/small-chibi-Satoru-gojo.png"  # Update this path to your logo image

# Sidebar
st.sidebar.title("Scikit-Learn Assistant")
st.sidebar.write("""
This assistant helps you solve your Scikit-Learn doubts by leveraging a RAG-based LLM.
- Developed by Abhishek Vidhate
""")
st.sidebar.write(
    "[GitHub](https://github.com/Abhishekvidhate) | [LinkedIn](https://www.linkedin.com/in/abhishek-vidhate-21smdb/)")

# Title and logo
st.title("Scikit-Learn Assistant")

# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []


def add_message(sender, text):
    st.session_state.messages.append({"sender": sender, "text": text})


# Loading state message
with st.spinner('Connecting to DB, please wait. As this app uses free cloud resources, it may take some time...'):
    # Initialize the RAG chain
    @st.cache(allow_output_mutation=True)
    def load_rag_chain():
        return create_rag_chain(markdown_path)


    rag_chain = load_rag_chain()

# Greeting message
if not st.session_state.messages:
    add_message("assistant", "Hi there! How can I help you with your Scikit-Learn doubt today?")

# # Custom CSS for transparent boxes
# st.markdown(
#     """
#     <style>
#     .user-box {
#         background-color: rgba(0, 123, 255, 0.1);
#         padding: 10px;
#         border-radius: 5px;
#         margin-bottom: 10px;
#     }
#     .assistant-box {
#         background-color: rgba(255, 255, 255, 0.1);
#         padding: 10px;
#         border-radius: 5px;
#         margin-bottom: 10px;
#         display: flex;
#         align-items: center;
#     }
#     .assistant-box img {
#         margin-right: 10px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Display chat messages
for message in st.session_state.messages:
    if message["sender"] == "assistant":
        st.markdown(
            f"""
                    <strong>Assistant:</strong> {message['text']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="user-box">
                <strong>You:</strong> {message['text']}
            </div>
            """,
            unsafe_allow_html=True
        )

# User input
query = st.text_input("Your question:", key="query_input")

if st.button("Submit"):
    if query:
        add_message("user", query)

        # Simulate thinking
        with st.spinner('Thinking... Please wait...'):
            response = rag_chain.invoke(query)

            # Clean up the response to show only relevant information
            cleaned_response = response.strip()  # Example of basic cleaning

            add_message("assistant", cleaned_response)

        # Clear the input box by resetting the session state value
        st.experimental_rerun()  # Rerun to update the chat log
    else:
        st.write("Please enter your doubt.")
