from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Load HTML
loader = AsyncChromiumLoader(["https://help.tallysolutions.com/tally-prime/accounting/interest-calculation-tally/"])

html = loader.load()

# Transform
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(
    html, tags_to_extract=["h1","h2","h3","p", "li", "ol", "ul"]
)

# print(docs_transformed)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                               chunk_overlap = 200, 
                                                length_function=len,
                                                is_separator_regex=False,
                                                separators=[
                                                            "\n\n", "\n",".",",",
                                                            "\u200B",  # Zero-width space
                                                            "\uff0c",  # Fullwidth comma
                                                            "\u3001",  # Ideographic comma
                                                            "\uff0e",  # Fullwidth full stop
                                                            "\u3002",  # Ideographic full stop
                                                            " ",
                                                            "",
                                                        ])
splits = text_splitter.split_documents(docs_transformed)
print(splits[10])