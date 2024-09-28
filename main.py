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
import pickle
import os.path
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import markdown2
import traceback

SCOPES = ['https://www.googleapis.com/auth/drive.file']

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
        def animate():
            chars = "|/-\\"
            i = 0
            start_time = time.time()
            while self.working and (time.time() - start_time) < 600:  # 10 minutes timeout
                try:
                    print(f"\rAI agents are working on your handbook {chars[i % len(chars)]}", end="", flush=True)
                    time.sleep(0.1)
                    i += 1
                except Exception as e:
                    print(f"\nError in animate: {e}")
                    break
            if self.working:
                print("\nWorking message timed out after 10 minutes")

        self.working = True
        thread = threading.Thread(target=animate)
        thread.daemon = True
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
        # return self.retry_with_exponential_backoff(lambda: llm_apis[llm].chat(chapter_prompt))

        try:
            result = self.retry_with_exponential_backoff(lambda: llm_apis[llm].chat(chapter_prompt))
            logging.info(f"Generate chapter result type: {type(result)}")
            logging.info(f"Generate chapter result (first 500 chars): {str(result)[:500]}...")
            return result
        except Exception as e:
            logging.error(f"Failed to generate chapter '{chapter_title}' using {llm}: {e}")
            logging.error(traceback.format_exc())
            return f"Failed to generate content for '{chapter_title}'. Error: {str(e)}"

    def adjust_header_levels(self, content):
        """
        Adjusts the header levels in the content to ensure proper nesting.
        """
        if not isinstance(content, str):
            content = str(content)
        lines = content.split('\n')
        adjusted_lines = []
        for line in lines:
            if line.strip().startswith('#'):
                # Increase the header level by one (e.g., H1 to H2)
                header_match = re.match(r'^(#+)', line.strip())
                if header_match:
                    hashes = header_match.group(1)
                    new_hashes = '#' + hashes  # Add one more '#' to increase header level
                    line = line.replace(hashes, new_hashes, 1)
            adjusted_lines.append(line)
        return '\n'.join(adjusted_lines)


    def format_anchor(self, title):
        """
        Formats the chapter title into an anchor link for the table of contents.
        """
        anchor = title.lower()
        anchor = re.sub(r'[^a-z0-9 ]', '', anchor)
        anchor = anchor.replace(' ', '-')
        return anchor

    def extract_content_from_dict(self, data, depth=0):
        """
        Extracts the content from various data structures with detailed logging.
        """
        logging.info(f"Extracting content (depth: {depth}), data type: {type(data)}")

        if isinstance(data, dict):
            logging.info(f"Dict keys: {list(data.keys())}")

            # Handle the specific nested structure we've encountered
            if 'chapter' in data:
                chapter_data = data['chapter']
                if isinstance(chapter_data, dict) and 'content' in chapter_data:
                    return self.extract_content_from_dict(chapter_data['content'], depth + 1)

            if 'content' in data:
                return self.extract_content_from_dict(data['content'], depth + 1)

            if 'paragraphs' in data:
                return '\n\n'.join(data['paragraphs'])

            for key in ['text', 'message', 'response']:
                if key in data:
                    return self.extract_content_from_dict(data[key], depth + 1)

            # If no specific key found, concatenate all string values
            result = ' '.join(str(value) for value in data.values() if isinstance(value, str))
            logging.info(f"Concatenated result: {result[:100]}...")  # Log first 100 chars
            return result

        elif isinstance(data, list):
            logging.info(f"List length: {len(data)}")
            return '\n\n'.join(self.extract_content_from_dict(item, depth + 1) for item in data if item)

        elif isinstance(data, str):
            logging.info(f"String content (first 100 chars): {data[:100]}...")
            return data.strip()

        else:
            logging.info(f"Other type: {type(data)}")
            return str(data)

    def process_content(self, content, depth=0):
        """
        Processes the chapter content, converting JSON-like structures to Markdown.

        Args:
            content: The content to process, which can be a dict, list, or string.
            depth: The current recursion depth to prevent infinite loops.

        Returns:
            A string containing the processed Markdown content.
        """
        MAX_DEPTH = 10  # Set a maximum recursion depth to prevent infinite loops
        if depth > MAX_DEPTH:
            logging.warning("Maximum recursion depth reached in process_content.")
            return ''

        if content is None:
            logging.warning("Content is None. Returning empty string.")
            return ''

        if isinstance(content, dict):
            # Content is a dict, process each key-value pair
            markdown = ""
            for key, value in content.items():
                if key.lower() == 'title':
                    continue  # Skip the 'title' key to avoid duplication
                markdown += self.process_content(value, depth + 1)
            return markdown
        elif isinstance(content, list):
            # Content is a list, process each item
            markdown = ""
            for item in content:
                markdown += self.process_content(item, depth + 1)
            return markdown
        elif isinstance(content, str):
            content_stripped = content.strip()
            # Attempt to parse the content if it looks like JSON
            try:
                data = ast.literal_eval(content)
                if isinstance(data, (dict, list)):
                    return self.process_content(data, depth + 1)
            except Exception:
                pass  # Not a JSON-like string, proceed as normal text
            # Content is regular text, return as is
            return content + "\n\n"
        else:
            # Unknown content type, convert to string
            return str(content) + "\n\n"

    def standardize_formatting(self):
        """
        Standardizes the formatting of the handbook content.
        """
        # Check if self.chapters is valid
        if not self.chapters:
            logging.error("No chapters available for formatting.")
            self.final_handbook_content = f"# Handbook on {self.topic}\n\nNo content available."
            return

        # Add the handbook title as H1
        formatted_content = f"# Handbook on {self.topic}\n\n"

        # Generate table of contents
        toc = "## Table of Contents\n\n"
        for i, chapter in enumerate(self.chapters, 1):
            if not isinstance(chapter, dict):
                logging.error(f"Invalid chapter format at index {i}. Skipping.")
                continue
            chapter_title = chapter.get('title', f'Chapter {i}')
            if not chapter_title:
                logging.error(f"Chapter {i} has no title. Using default title.")
                chapter_title = f'Chapter {i}'
            toc += f"- [{chapter_title}](#{self.format_anchor(chapter_title)})\n"
        formatted_content += toc + "\n"

        # Process each chapter
        for chapter in self.chapters:
            if not isinstance(chapter, dict):
                logging.error(f"Invalid chapter format: {chapter}. Skipping.")
                continue
            title = chapter.get('title', 'Untitled Chapter')
            content = chapter.get('content', 'No content available.')

            if not content:
                logging.warning(f"Chapter '{title}' has no content. Using default content.")
                content = 'No content available.'

            # Process content
            content = self.process_content(content)

            # Adjust header levels in content
            content = self.adjust_header_levels(content)

            # Add chapter title as H2
            formatted_content += f"## {title}\n\n{content}\n\n"

        self.final_handbook_content = formatted_content

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
                result = func()
                logging.info(f"Operation successful on attempt {retries + 1}")
                return result
            except Exception as e:
                logging.error(f"Error during operation (attempt {retries + 1}/{MAX_RETRIES}): {e}")
                logging.error(traceback.format_exc())
                retries += 1
                wait_time = 2 ** retries
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        logging.error(f"Max retries ({MAX_RETRIES}) reached. Operation failed.")
        return None

    def authenticate_google_drive(self):
        """
        Authenticates the user with Google Drive API and returns a service object.

        Returns:
            service: Authorized Google Drive API service instance.
        """
        creds = None
        # The file token.pickle stores the user's access and refresh tokens.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)  # Ensure you have 'credentials.json' in your directory
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        service = build('drive', 'v3', credentials=creds)
        return service

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
        try:
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

            if not isinstance(revised_structure, dict):
                logging.error("Revised structure is not a dictionary. Falling back to initial structure.")
                revised_structure = initial_structure

            if not isinstance(revised_structure, dict):
                logging.error("Initial structure is also not a dictionary. Cannot proceed.")
                print("Error: Failed to generate a valid handbook structure.")
                return

            logging.info(f"Revised handbook structure: {revised_structure}")

            self.chapters = revised_structure.get('chapters', [])

            if not isinstance(self.chapters, list):
                logging.error("Chapters is not a list. Cannot proceed with handbook generation.")
                print("Error: Chapters is invalid. Cannot proceed.")
                return

            valid_chapters = []
            for idx, chapter in enumerate(self.chapters):
                if not isinstance(chapter, dict):
                    logging.error(f"Chapter at index {idx} is not a dictionary. Skipping.")
                    continue
                if 'title' not in chapter:
                    logging.error(f"Chapter at index {idx} does not have a 'title' key. Skipping.")
                    continue
                valid_chapters.append(chapter)
            self.chapters = valid_chapters

            chapter_queue = queue.Queue()

            logging.info("Final chapter contents:")

            for chapter in self.chapters:
                logging.info(f"Chapter: {chapter['title']}")
                logging.info(f"Content (first 500 chars): {chapter['content'][:500]}...")
                chapter_queue.put(chapter['title'])

            # Step 4: Multithreading for Chapter Generation
            logging.info(f"Starting {THREAD_LIMIT} worker threads")
            threads = []
            llm_keys = list(llm_apis.keys())
            for i in range(THREAD_LIMIT):
                llm = llm_keys[i % len(llm_keys)]
                t = threading.Thread(target=self.worker_thread, args=(llm, chapter_queue))
                threads.append(t)
                t.start()

            # Wait for all threads to complete or timeout
            timeout = 600  # 10 minutes timeout
            start_time = time.time()
            for t in threads:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                t.join(timeout=remaining_time)

            # Check if any threads are still alive
            if any(t.is_alive() for t in threads):
                logging.warning("Some threads did not complete within the timeout period.")

            logging.info("All threads completed or timed out")

            self.stop_working_message()

            logging.info("Starting standardize_formatting")
            self.standardize_formatting()
            logging.info("Finished standardize_formatting")
            print("\n=== Generated Handbook Content ===\n")
            print(self.final_handbook_content)
            print("\n===================================\n")

            # Feedback loop for iterative improvements
            while True:
                print("\n=== Feedback Options ===")
                print("1. Provide feedback on a specific chapter")
                print("2. Add a new chapter")
                print("3. Finish and save the handbook")
                print("4. Upload handbook to Google Docs")
                print("5. Exit the app.")
                print("6. Start over with a new handbook topic")

                choice = input("Enter your choice (1-6): ")

                if choice == '1':
                    self.chapter_feedback()
                elif choice == '2':
                    self.add_new_chapter()
                elif choice == '3':
                    # Save the handbook
                    self.standardize_formatting()
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    output_filename = f'handbook_{self.topic}_{timestamp}.md'
                    with open(output_filename, 'w', encoding='utf-8') as output_file:
                        output_file.write(self.final_handbook_content)
                    print(f"Handbook saved to {output_filename}")
                elif choice == '4':
                    # Save the handbook if not already saved
                    if not self.final_handbook_content:
                        self.standardize_formatting()
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    output_filename = f'handbook_{self.topic}_{timestamp}.md'
                    with open(output_filename, 'w', encoding='utf-8') as output_file:
                        output_file.write(self.final_handbook_content)
                    # Authenticate and upload
                    service = self.authenticate_google_drive()
                    self.upload_to_google_docs(service, output_filename)
                elif choice == '5':
                    print("Exiting the application.")
                    break
                elif choice == '6':
                    new_topic = input("Enter the new topic for the handbook: ")
                    self.reset(new_topic)
                    return self.generate_handbook()  # Restart the handbook generation process
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

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            logging.error(traceback.format_exc())
            print(f"An unexpected error occurred. Please check the logs for more information.")
        finally:
            self.stop_working_message()

    def worker_thread(self, llm, chapter_queue):
        while not chapter_queue.empty():
            chapter_title = chapter_queue.get()
            try:
                logging.info(f"Generating chapter: {chapter_title}")
                chapter_content = self.generate_chapter(llm, chapter_title)
                logging.info(f"Chapter content type: {type(chapter_content)}")
                logging.info(f"Raw chapter content: {str(chapter_content)[:500]}...")  # Log first 500 chars

                extracted_content = self.extract_content_from_dict(chapter_content)
                logging.info(f"Extracted content (first 500 chars): {extracted_content[:500]}...")

                if not extracted_content or not extracted_content.strip():
                    logging.warning(f"Invalid or empty chapter content for {chapter_title}. Setting default content.")
                    extracted_content = f"Content for chapter '{chapter_title}' could not be generated. Please review and update manually."
                else:
                    logging.info(f"Successfully extracted content for {chapter_title}")

                for chapter in self.chapters:
                    if chapter['title'] == chapter_title:
                        chapter['content'] = extracted_content
                        break
                logging.info(f"Chapter '{chapter_title}' processed successfully.")
            except Exception as e:
                logging.error(f"Error processing chapter '{chapter_title}': {e}")
                logging.error(traceback.format_exc())
                for chapter in self.chapters:
                    if chapter['title'] == chapter_title:
                        chapter['content'] = f"Error generating content for '{chapter_title}'. Please review and update manually."
                        break
            finally:
                chapter_queue.task_done()
        logging.info("Worker thread completed")

    def upload_to_google_docs(self, service, output_filename):
        try:
            # Convert Markdown to HTML
            with open(output_filename, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
            html_content = markdown2.markdown(markdown_text)
            html_filename = output_filename.replace('.md', '.html')
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)

            file_metadata = {
                'name': os.path.splitext(output_filename)[0],
                'mimeType': 'application/vnd.google-apps.document'
            }
            media = MediaFileUpload(html_filename, mimetype='text/html', resumable=True)
            uploaded_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            file_id = uploaded_file.get('id')
            print(f"Document uploaded and converted to Google Docs. File ID: {file_id}")
            print(f"Access it here: https://docs.google.com/document/d/{file_id}/edit")

            # Add a short delay before attempting to delete the file
            time.sleep(2)

            # Use a more robust file deletion approach
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    os.remove(html_filename)
                    logging.info(f"Temporary file {html_filename} deleted successfully.")
                    break
                except Exception as e:
                    logging.info(f"Attempt {attempt + 1} to delete temporary file failed: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(1)  # Wait for 1 second before the next attempt
            else:
                logging.warning(f"Could not delete temporary file {html_filename} after {max_attempts} attempts.")

        except Exception as e:
            logging.error(f"An error occurred while uploading to Google Docs: {e}", exc_info=True)
            print(f"An error occurred: {e}")

    def reset(self, new_topic):
        """
        Resets the HandbookGenerator with a new topic.
        """
        self.__init__(new_topic)
        logging.info(f"HandbookGenerator reset with new topic: {new_topic}")

def main():
    while True:
        print("You can generate a handbook on any topic and upload it to Google Docs.")
        topic = input("Enter the topic for the handbook (or 'quit' to exit): ")
        if topic.lower() == 'quit':
            print("Exiting the application. Goodbye!")
            break
        handbook_generator = HandbookGenerator(topic)
        handbook_generator.generate_handbook()

        continue_choice = input("Do you want to create another handbook? (yes/no): ").lower()
        if continue_choice != 'yes':
            print("Exiting the application. Goodbye!")
            break

if __name__ == "__main__":
    main()
