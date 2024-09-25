import os
import threading
import queue
import logging
from datetime import datetime
from dotenv import load_dotenv
from unified import UnifiedApis
import requests
import time
import json
import re
import ast

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, filename='handbook_generator.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global constants
THREAD_LIMIT = 3
MAX_RETRIES = 5

# Initialize LLM APIs
llm_apis = {
    'openai': UnifiedApis(provider='openai', api_key=OPENAI_API_KEY),
    'anthropic': UnifiedApis(provider='anthropic', api_key=ANTHROPIC_API_KEY),
    'openrouter': UnifiedApis(provider='openrouter', api_key=OPENROUTER_API_KEY)
}

class HandbookGenerator:
    """
    A class to generate handbooks using multiple language models (LLMs).

    Attributes:
        topic (str): The topic for the handbook.
        context_summary (str): The summary of context gathered from web search.
        chapters (list): List of chapters in the handbook.
        final_handbook_content (str): The final generated content of the handbook.
    """
    def __init__(self, topic):
        """
        Initializes the HandbookGenerator with the given topic.

        Args:
            topic (str): The topic for the handbook.
        """
        self.topic = topic
        self.context_summary = ""
        self.chapters = []
        self.final_handbook_content = ""
        self.working = False

    def print_working_message(self):
        """Displays an animated message indicating that AI agents are working on the handbook."""
        def animate():
            chars = "|/-\\"
            i = 0
            while self.working:
                print(f"\rAI agents are working on your handbook {chars[i % len(chars)]}", end="", flush=True)
                time.sleep(0.1)
                i += 1

        self.working = True
        thread = threading.Thread(target=animate)
        thread.start()

    def stop_working_message(self):
        """Stops the animated working message and displays a completion message."""
        self.working = False
        print("\rAI agents have finished working on your handbook!", flush=True)
        print("\n")

    def search_web(self, query):
        """
        Searches the web for context about the given query using the Serper Dev API.

        Args:
            query (str): The search query to find context for.

        Returns:
            list: A list of search results containing title, link, and snippet.
        """
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': os.getenv("SERPER_API_KEY"),
            'Content-Type': 'application/json'
        }

        max_retries = 5
        delay = 1  # Initial delay in seconds for exponential backoff

        logger.debug(f"Starting web search for query: {query}")
        results = []
        retries = 0

        while retries < max_retries:
            try:
                response = requests.post(url, headers=headers, data=payload)
                response.raise_for_status()
                search_data = response.json()
                if "organic" in search_data:
                    for item in search_data["organic"]:
                        title = item.get("title", "")
                        link = item.get("link", "")
                        snippet = item.get("snippet", "")
                        if title and link and snippet:
                            results.append({"title": title, "link": link, "snippet": snippet})
                logger.info(f"Web search completed for query: {query} with {len(results)} results")
                return results
            except requests.exceptions.HTTPError as e:
                retries += 1
                if response.status_code == 404:
                    logger.error(f"404 Error: Resource not found for query: {query}.")
                    break  # Break out of the retry loop if 404 is encountered
                wait_time = delay * (2 ** retries)
                logger.warning(f"Search failed with HTTP error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                retries += 1
                wait_time = delay * (2 ** retries)
                logger.warning(f"Search failed with exception: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        if not results:
            logger.error(f"Max retries reached or search failed. No results found for query: {query}. Proceeding with default outline creation.")

        return results  # Returning results even if empty for further processing

    def generate_structure(self):
        """
        Uses the Structure Agent to create an outline for the handbook.

        Returns:
            str: The generated structure for the handbook.
        """
        structure_prompt = f"Create an outline for a handbook on the topic '{self.topic}' using the following context: {self.context_summary}."
        response = llm_apis['anthropic'].chat(structure_prompt)
        return response

    def review_outline(self, initial_outline):
        """
        Reviews and updates the initial handbook outline.

        This method sends the initial outline to an LLM for review,
        identifies potential gaps, and provides an updated structure with chapter summaries.

        Args:
            initial_outline (str): The initial outline of the handbook.

        Returns:
            dict: A revised structure containing a list of chapters with titles and content summaries.
        """
        review_prompt = f"""
        Review the following handbook outline and identify any potential gaps in the content.
        Provide an updated outline that fills in those gaps, using the context provided.
        Please return the response in a JSON format with a 'chapters' key that contains a list of chapter objects.
        Each chapter object should have a 'title' and a 'content' key. The 'content' should be a brief summary or outline of the chapter.
        Output only valid JSON. Ensure proper open and close braces and brackets per valid JSON.
        Outline: {initial_outline}
        Context: {self.context_summary}
        """
        response = llm_apis['openai'].chat(review_prompt)

        try:
            revised_structure = json.loads(response) if isinstance(response, str) else response
            if 'chapters' not in revised_structure or not isinstance(revised_structure['chapters'], list):
                raise ValueError("Invalid structure: 'chapters' key missing or not a list")

            for chapter in revised_structure['chapters']:
                if 'title' not in chapter or 'content' not in chapter:
                    raise ValueError("Invalid chapter structure: 'title' or 'content' key missing")

            return revised_structure
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse or validate response: {e}")
            return {"chapters": [{"title": "Default Chapter", "content": ""}]}

    def generate_chapter(self, llm, chapter_title):
        """
        Generates a chapter using the specified language model (LLM).

        This method sends a prompt to the specified LLM to generate content for a chapter
        based on the provided title and context summary. It ensures the generated content
        is clear, concise, and aligned with the desired writing style.

        Args:
            llm (str): The name of the LLM to use for generating the chapter.
            chapter_title (str): The title of the chapter to be generated.

        Returns:
            str: The generated chapter content.

        Raises:
            Exception: If the LLM call fails or returns an unexpected response.
        """
        chapter_prompt = f"""
        Write a chapter titled '{chapter_title}' using the following context: {self.context_summary}.
        ### **Write With Clarity and Conciseness:**
        - **Clear Language:** Use simple, jargon-free language that is easy to understand.
        - **Vary Sentence Length:** Use simple, jargon-free language. Vary sentence lengths in a pseudo-random way to ensure engaging writing.
        - **Grammar and Punctuation:** Use proper language mechanics for credibility and clarity.
        - **Conciseness:** You convey messages succinctly, avoiding unnecessary fluff.
        - **Tone and Voice:** You adapt tone to align with the topic and the audience's expectations.
        - **High Degree of Burstiness and Perplexity:** Use mostly very short paragraphs and sentences to ensure readability. The content should be straightforward and easy to understand.
        - **Formatting Code** If code is presented in the handbook content, place it inside <code> </code> tags.
        """
        return self.retry_with_exponential_backoff(lambda: llm_apis[llm].chat(chapter_prompt))

    def standardize_formatting(self):
        """
        Standardizes the formatting of the handbook content.

        This method processes the content of each chapter, converts any JSON-like
        strings to Markdown, ensures proper header formatting, and combines all
        chapters into the final handbook content.

        Returns:
            None: Updates the `final_handbook_content` attribute with the formatted content.
        """
        formatted_content = ""
        for chapter in self.chapters:
            title = chapter.get('title', 'Untitled Chapter')
            content = chapter.get('content', 'No content available.')
            formatted_content += f"# {title}\n\n{content}\n\n"

        # Convert JSON-like strings to Markdown
        json_pattern = r'\{[\s\S]*?\}'
        def json_to_markdown(match):
            try:
                data = ast.literal_eval(match.group(0))
                if isinstance(data, dict):
                    markdown = ""
                    for key, value in data.items():
                        if key == 'title':
                            continue  # Skip the 'title' key
                        if isinstance(value, list):
                            for item in value:
                                markdown += f"{item}\n\n"
                        else:
                            markdown += f"{value}\n\n"
                    return markdown
                else:
                    return match.group(0)
            except Exception:
                return match.group(0)

        formatted_content = re.sub(json_pattern, json_to_markdown, formatted_content)

        # Ensure headers are in Markdown format
        lines = formatted_content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                # Avoid adding '###' to existing Markdown headers
                if not re.match(r'^(#+\s)', line.strip()):
                    lines[i] = f"### {line.strip()}"

        self.final_handbook_content = '\n'.join(lines)

    def chapter_feedback(self):
        """
        Allows the user to provide feedback on a specific chapter.

        This method displays a list of chapters, prompts the user to select a chapter,
        and then processes user feedback to update the content of the selected chapter.

        Returns:
            None: Updates the content of the selected chapter based on user feedback.
        """
        print("\n=== Chapters ===")
        for i, chapter in enumerate(self.chapters, 1):
            print(f"{i}. {chapter['title']}")

        chapter_choice = int(input("Enter the number of the chapter you want to provide feedback on: ")) - 1
        if 0 <= chapter_choice < len(self.chapters):
            chapter = self.chapters[chapter_choice]
            print(f"\nCurrent content of '{chapter['title']}':\n")
            print(chapter['content'])
            feedback = input("\nEnter your feedback for this chapter: ")

            adapt_prompt = f"""
            Based on the following feedback, adapt the content of the chapter '{chapter['title']}':

            Current chapter content:
            {chapter['content']}

            Feedback: {feedback}

            Please provide the updated content for this chapter only.
            """

            try:
                updated_content = llm_apis['openai'].chat(adapt_prompt, max_tokens=2000, timeout=300)
                chapter['content'] = updated_content
                print("Chapter updated successfully.")
            except Exception as e:
                logging.error(f"Error updating chapter: {e}", exc_info=True)
                print(f"An error occurred while updating the chapter: {e}")
        else:
            print("Invalid chapter number.")

    def add_new_chapter(self):
        """
        Adds a new chapter to the handbook based on user input.

        This method prompts the user for a new chapter title, generates content for the chapter,
        and adds it to the list of chapters in the handbook.

        Returns:
            None: Adds a new chapter to the `chapters` list.
        """
        chapter_title = input("Enter the title for the new chapter: ")

        add_chapter_prompt = f"""
        Create a new chapter titled '{chapter_title}' for the handbook on '{self.topic}'.
        Ensure the content is relevant and fits well with the existing structure of the handbook.
        """

        try:
            new_chapter_content = llm_apis['openai'].chat(add_chapter_prompt, max_tokens=2000, timeout=300)
            self.chapters.append({'title': chapter_title, 'content': new_chapter_content})
            print("New chapter added successfully.")
        except Exception as e:
            logging.error(f"Error adding new chapter: {e}", exc_info=True)
            print(f"An error occurred while adding the new chapter: {e}")

    def review_structure(self):
        """
        Reviews and potentially updates the overall structure of the handbook.

        This method displays the current structure of the handbook, suggests improvements,
        and allows the user to apply structural changes based on these suggestions.

        Returns:
            None: May update the `chapters` list based on user-approved changes.
        """
        current_structure = "\n".join([chapter['title'] for chapter in self.chapters])

        review_prompt = f"""
        Review the current structure of the handbook on '{self.topic}':

        {current_structure}

        Suggest any improvements or changes to the overall structure. Do not generate content for the chapters.
        """

        try:
            suggestions = llm_apis['openai'].chat(review_prompt, max_tokens=1000, timeout=300)
            print("\nSuggestions for improving the handbook structure:")
            print(suggestions)

            apply_changes = input("Do you want to apply these changes? (yes/no): ").lower()
            if apply_changes == 'yes':
                # Here you could implement logic to update the structure based on the suggestions
                # For simplicity, we'll just update the chapter titles
                new_titles = suggestions.split('\n')
                self.chapters = [{'title': title.strip(), 'content': ''} for title in new_titles if title.strip()]
                print("Handbook structure updated.")
            else:
                print("Changes not applied.")
        except Exception as e:
            logging.error(f"Error reviewing structure: {e}", exc_info=True)
            print(f"An error occurred while reviewing the structure: {e}")

    def retry_with_exponential_backoff(self, func):
        """
        Retries a function with exponential backoff in case of failure.

        This method attempts to execute a given function multiple times, doubling the wait
        time after each failure to reduce the likelihood of repeated failures. It is useful
        for handling transient issues such as network errors or temporary unavailability of services.

        Args:
            func (Callable): The function to be executed and potentially retried.

        Returns:
            Any: The result of the function call if successful, otherwise None.

        Raises:
            Exception: If the maximum number of retries is reached without success.
        """
        retries = 0
        while retries < MAX_RETRIES:
            try:
                return func()
            except Exception as e:
                logging.error(f"Error during operation: {e}. Retrying...")
                retries += 1
                time.sleep(2 ** retries)
        logging.error("Max retries reached.")
        return None

    def generate_handbook(self):
        """
        Generates the entire handbook using multiple LLMs and user feedback.

        This method orchestrates the overall process of handbook generation by:
        - Gathering contextual information via web search.
        - Generating an initial outline using a structure agent.
        - Reviewing and updating the outline using a review agent.
        - Generating content for each chapter using multiple LLMs.
        - Allowing iterative refinement of the content based on user feedback.

        Returns:
            None: The final handbook content is saved to a file and stored in `final_handbook_content`.
        """
        self.print_working_message()
        # Step 1: Web Search for Contextual Information
        logging.info(f"Starting web search for topic: {self.topic}")
        self.context_summary = self.search_web(self.topic)  # Pass topic as a string
        if not self.context_summary:  # If web search fails or returns no results
            logging.warning("No context gathered from web search. Proceeding with default outline.")

        # Step 2: Generate Initial Handbook Structure
        logging.info(f"Generating initial structure for the handbook on: {self.topic}")
        initial_structure = self.generate_structure()
        logging.info(f"Initial handbook structure generated: {initial_structure}")

        # Step 3: Review and Update Handbook Structure
        logging.info("Reviewing and updating handbook structure to fill in potential gaps.")
        revised_structure = self.review_outline(initial_structure)

        if isinstance(revised_structure, str):
            logging.error("Revised structure is a string, expected a dictionary. Falling back to initial structure.")
            revised_structure = initial_structure

        logging.info(f"Revised handbook structure: {revised_structure}")

        # Proceed to chapter generation with the revised structure
        self.chapters = revised_structure.get('chapters', [])
        chapter_queue = queue.Queue()

        for chapter in self.chapters:
            chapter_queue.put(chapter['title'])

        # Step 4: Multithreading for Chapter Generation
        threads = []
        llm_keys = list(llm_apis.keys())
        for i in range(THREAD_LIMIT):
            llm = llm_keys[i % len(llm_keys)]
            t = threading.Thread(target=self.worker_thread, args=(llm, chapter_queue))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.stop_working_message()

        self.standardize_formatting()
        print("\n=== Generated Handbook Content ===\n")
        print(self.final_handbook_content)
        print("\n===================================\n")

        # Feedback loop for iterative improvements
        while True:
            print("\n=== Feedback Options ===")
            print("1. Provide feedback on a specific chapter")
            print("2. Add a new chapter")
            print("3. Finish and save the handbook")

            choice = input("Enter your choice (1-3): ")

            if choice == '1':
                self.chapter_feedback()
            elif choice == '2':
                self.add_new_chapter()
            elif choice == '3':
                break
            else:
                print("Invalid choice. Please try again.")

        # Final standardization before saving
        self.standardize_formatting()

        # Step 5: Output the handbook to a markdown file
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        output_filename = f'handbook_{self.topic}_{timestamp}.md'
        with open(output_filename, 'w') as output_file:
            output_file.write(self.final_handbook_content)

        logging.info(f"Handbook saved to {output_filename}")
        print(f"Handbook generated and saved as {output_filename}")

    def worker_thread(self, llm, chapter_queue):
        """
        Thread worker function to process chapter generation tasks.

        This method is executed by each thread to generate content for chapters
        using the specified LLM. It pulls chapter titles from a queue, generates
        content for each, and appends the content to the final handbook.

        Args:
            llm (str): The name of the LLM to use for generating chapters.
            chapter_queue (Queue): A queue containing chapter titles to be processed.

        Returns:
            None: Updates the `final_handbook_content` attribute with the generated chapters.
        """
        while not chapter_queue.empty():
            chapter_title = chapter_queue.get()
            try:
                chapter_content = self.generate_chapter(llm, chapter_title)
                # Find the chapter in self.chapters and update its content
                for chapter in self.chapters:
                    if chapter['title'] == chapter_title:
                        chapter['content'] = chapter_content
                        break
                logging.info(f"Chapter '{chapter_title}' generated successfully.")
            except Exception as e:
                logging.error(f"Error generating chapter '{chapter_title}': {e}")
            chapter_queue.task_done()

def main():
    topic = input("Enter the topic for the handbook: ")
    handbook_generator = HandbookGenerator(topic)
    handbook_generator.generate_handbook()

if __name__ == "__main__":
    main()
