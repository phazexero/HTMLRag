from langchain_community.document_loaders import BSHTMLLoader
import requests

class CustomHTMLLoader(BSHTMLLoader):
    def __init__(self, url):
        self.url = url

    def load(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            return response.content
        else:
            print("Failed to load HTML content from the URL:", self.url)
            return None

# Example usage:
loader = CustomHTMLLoader("https://help.tallysolutions.com/tally-prime/accounting/interest-calculation-tally/")
data = loader.load()


# html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)

# html_header_splits = html_splitter.split_text(html_string)

# chunk_size = 500
# chunk_overlap = 30
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=chunk_size, chunk_overlap=chunk_overlap
# )


print(data)  # This will print the HTML content if loaded successfully, or None if failed
