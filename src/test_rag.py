from src.rag import create_rag_chain

# Example usage
markdown_path = "C:/Users/abhis/PycharmProjects/SKL-Assistant/data/skldata.md"
rag_chain = create_rag_chain(markdown_path)
query = "what is Task Decomposition? Explain in complete detail with a code example."
response = rag_chain.invoke(query)
print(response)

