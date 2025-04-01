import asyncio
import streamlit as st
from duckduckgo_search import DDGS
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult

import chromadb
import tempfile
from chromadb.config import Settings
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys

# How do I apply as a voter in the Philippines?

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

def normalize_url(url):
    normalized_url = (
        url.replace("https://", "")
        .replace("www.", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )

    print("Normalized URL", normalized_url)
    return normalized_url


def get_vector_collection() -> tuple[chromadb.Collection, chromadb.Client]:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest"
    )

    chroma_client = chromadb.PersistentClient(
        path="./web-search-llm-db", settings=Settings(anonymized_telemetry=False)
    )

    return (
        chroma_client.get_or_create_collection(
            name="web_llm",
            embedding_function=ollama_ef,
            metadata={"hnsw:space": "cosine"}
        ),
        chroma_client 
    )


def add_to_vector_database(results: list[CrawlResult]):
    collection, _ = get_vector_collection()

    for result in results:
        st.write(result)
        documents, metadatas, ids = [], [], []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        if result.markdown_v2:
            markdown_result = result.markdown_v2.fit_markdown
        else:
            continue

        temp_file = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
        temp_file.write(markdown_result)
        temp_file.flush()

        loader=UnstructuredMarkdownLoader(temp_file.name, mode="single")
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        st.write(all_splits)
        normalized_url = normalize_url(result.url)

        if all_splits:
            for idx, split in enumerate(all_splits):
                documents.append(split.page_content)
                metadatas.append({"source": result.url})
                ids.append(f"{normalized_url}_{idx}")

            print("Upsert collection: ", id(collection))
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)


async def crawl_webpages(urls: list[str], prompt: str) -> CrawlResult:
    print("Crawling webpage with prompt: ")
    print(prompt)
    bm25_filter = BM25ContentFilter(user_query=prompt, bm25_threshold=1.2)
    print(bm25_filter)
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)

    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header", "form", "img", "a"],
        only_text=True,
        exclude_social_media_links=True,
        keep_data_attributes=False,
        cache_mode=CacheMode.BYPASS,
        remove_overlay_elements=True,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        page_timeout=20000      # 20 seconds
    )

    browser_config = BrowserConfig(headless=True, text_mode=True, light_mode=True)

    results=[]
    try:
        async with AsyncWebCrawler(config=browser_config, max_concurrent_requests=3) as crawler:
            # Use asyncio.wait_for to add a timeout
            results = await asyncio.wait_for(
                crawler.arun_many(urls, config=crawler_config),
                timeout=60  # 60 second timeout for the entire crawl
            )
            print(f"Successfully crawled {len(results)} pages")
            
    except asyncio.TimeoutError:
        print("Crawling timed out after 60 seconds")
    except Exception as e:
        print(f"Error during crawling: {str(e)}")
    
    # Return whatever results we have, even if incomplete
    print(results)
    return results


def check_robots_txt(urls: list[str]) -> list[str]:
    allowed_urls = []

    for url in urls:
        try:
            robots_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt"
            rp = RobotFileParser(robots_url)
            rp.read()

            if rp.can_fetch("*", url):
                allowed_urls.append(url)
        except Exception:
            allowed_urls.append(url)
    return allowed_urls


def get_web_urls(search_term: str, num_results: int = 10) -> list[str]:
    try:
        discard_urls = ["youtube.com", "britannica.com", "vimeo.com"]
        for url in discard_urls:
            search_term += f" -site:{url}"

        results = DDGS().text(search_term, max_results=num_results)
        st.write(results)
        results = [result["href"] for result in results]
        
        # Checks the robots_txt
        return check_robots_txt(results)
    
    except Exception as e:
        error_msg = ("Failed to fetch results from the web", str(e))
        print(error_msg)
        st.write(error_msg)
        st.stop()



async def run():
    st.set_page_config(page_title="LLM with Web Search")

    st.header("Voting Informations")

    prompt = st.text_area(
        label="Put your query here",
        placeholder="Add your query...",
        label_visibility="hidden"
    )

    is_web_search = st.toggle("Enable web search", value=False, key="enable_web_search")
    go=st.button(
        "Go"
    )

    collection, chroma_client = get_vector_collection()

    if prompt and go:
        if is_web_search:
            web_urls = get_web_urls(search_term=prompt)
            print("Starting Search")
            if not web_urls:
                st.write("No results found.")
                st.stop()

            results = await crawl_webpages(urls=web_urls, prompt=prompt)
            add_to_vector_database(results)


if __name__ == "__main__":
    asyncio.run(run())