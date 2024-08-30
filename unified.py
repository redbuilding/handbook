import os
import json
from openai import OpenAI
from anthropic import Anthropic
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from termcolor import colored
import time
import asyncio
from pydantic import BaseModel
from typing import Any, Optional

class UnifiedApis:
    def __init__(self,
                 name="Unified Apis",
                 api_key=None,
                 max_history_words=10000,
                 max_words_per_message=None,
                 json_mode=True,
                 stream=False,
                 use_async=False,
                 max_retry=10,
                 provider="anthropic",
                 model=None,
                 should_print_init=True,
                 print_color="green",
                 use_cache=False,
                 cache_interval=10,
                 print_cache_usage=False
                 ):

        self.provider = provider.lower()
        if self.provider == "openai":
            self.model = model or "gpt-4o-mini"
        elif self.provider == "anthropic":
            self.model = model or "claude-3-5-sonnet-20240620"
        elif self.provider == "openrouter":
            self.model = model or "perplexity/llama-3.1-sonar-large-128k-online"
        self.name = name
        self.api_key = api_key or self._get_api_key()
        self.history = []
        self.max_history_words = max_history_words
        self.max_words_per_message = max_words_per_message
        self.json_mode = json_mode
        self.stream = stream
        self.use_async = use_async
        self.max_retry = max_retry
        self.print_color = print_color
        self.system_message = "You are a helpful assistant."
        if self.provider == "openai" and self.json_mode:
            self.system_message += " Please return your response in JSON unless user has specified a system message."
        self.use_cache = use_cache
        self.cache_interval = cache_interval
        self.turn = 1
        self.print_cache_usage = print_cache_usage

        self._initialize_client()

        if should_print_init:
            print(colored(f"{self.name} initialized with provider={self.provider}, model={self.model}, json_mode={json_mode}, stream={stream}, use_async={use_async}, max_history_words={max_history_words}, max_words_per_message={max_words_per_message}, use_cache={use_cache}, cache_interval={cache_interval}, print_cache_usage={print_cache_usage}", "red"))

    def _get_api_key(self):
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY") or "YOUR_OPENAI_KEY_HERE"
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY") or "YOUR_ANTHROPIC_KEY_HERE"
        elif self.provider == "openrouter":
            return os.getenv("OPENROUTER_API_KEY") or "YOUR_OPENROUTER_KEY_HERE"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _initialize_client(self):
        if self.provider == "openai" and self.use_async:
            self.client = AsyncOpenAI(api_key=self.api_key)
        elif self.provider == "anthropic" and self.use_async:
            self.client = AsyncAnthropic(api_key=self.api_key)
        elif self.provider == "openrouter" and self.use_async:
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )
        elif self.provider == "openai" and not self.use_async:
            self.client = OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic" and not self.use_async:
            self.client = Anthropic(api_key=self.api_key)
        elif self.provider == "openrouter" and not self.use_async:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )

    def set_system_message(self, message=None):
        self.system_message = message or "You are a helpful assistant."
        if self.provider == "openai" and self.json_mode and "json" not in message.lower():
            self.system_message += " Please return your response in JSON unless user has specified a system message."

        if self.use_cache:
            self.system_message = {
                "type": "text",
                "text": self.system_message,
                "cache_control": {"type": "ephemeral"}
            }

    async def set_system_message_async(self, message=None):
        self.set_system_message(message)

    def add_message(self, role, content):
        if role == "user" and self.max_words_per_message:
            content += f" please use {self.max_words_per_message} words or less"

        message = {"role": role, "content": str(content)}

        if self.use_cache and self.turn % self.cache_interval == 0:
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
            message["content"][0]["cache_control"] = {"type": "ephemeral"}

        self.history.append(message)
        self.turn += 1

    async def add_message_async(self, role, content):
        self.add_message(role, content)

    def print_history_length(self):
        history_length = sum(len(str(message["content"]).split()) for message in self.history)
        print(f"\nCurrent history length is {history_length} words")

    async def print_history_length_async(self):
        self.print_history_length()

    def clear_history(self):
        self.history.clear()

    async def clear_history_async(self):
        self.clear_history()

    def chat(self, user_input, response_model: Optional[BaseModel] = None, **kwargs):
        self.add_message("user", user_input)
        return self.get_response(response_model=response_model, **kwargs)

    async def chat_async(self, user_input, response_model: Optional[BaseModel] = None, **kwargs):
        await self.add_message_async("user", user_input)
        return await self.get_response_async(response_model=response_model, **kwargs)

    def trim_history(self):
        words_count = sum(len(str(message["content"]).split()) for message in self.history if message["role"] != "system")
        while words_count > self.max_history_words and len(self.history) > 1:
            words_count -= len(self.history[0]["content"].split())
            self.history.pop(0)

    async def trim_history_async(self):
        self.trim_history()

    def remove_previous_cache_keys(self):
        for message in self.history:
            if isinstance(message["content"], list) and "cache_control" in message["content"][0]:
                del message["content"][0]["cache_control"]

    def get_response(self, color=None, should_print=True, response_model: Optional[BaseModel] = None, **kwargs):
        if color is None:
            color = self.print_color

        max_tokens = kwargs.pop('max_tokens', 4000)
        anthropic_max_tokens = kwargs.pop('max_tokens', 8192)

        if self.use_cache:
            self.remove_previous_cache_keys()

        retries = 0
        while retries < self.max_retry:
            try:
                if self.provider == "openai":
                    if response_model:
                        response = self.client.beta.chat.completions.parse(
                            model=self.model,
                            messages=[{"role": "system", "content": self.system_message}] + self.history,
                            max_tokens=max_tokens,
                            response_format=response_model,
                            **kwargs
                        )
                    else:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "system", "content": self.system_message}] + self.history,
                            stream=self.stream,
                            max_tokens=max_tokens,
                            response_format={"type": "json_object"} if self.json_mode else None,
                            **kwargs
                        )
                elif self.provider == "anthropic":
                    if self.use_cache:
                        response = self.client.beta.prompt_caching.messages.create(
                            model=self.model,
                            system=[self.system_message],
                            messages=self.history,
                            stream=self.stream,
                            max_tokens=max_tokens,
                            # extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                            **kwargs
                        )
                    else:
                        response = self.client.messages.create(
                            model=self.model,
                            system=self.system_message,
                            messages=self.history,
                            stream=self.stream,
                            max_tokens=anthropic_max_tokens,
                            extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                            **kwargs
                        )
                elif self.provider == "openrouter":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": self.system_message}] + self.history,
                        stream=self.stream,
                        max_tokens=max_tokens,
                        **kwargs
                    )

                if self.stream and not response_model:
                    assistant_response = ""
                    for chunk in response:
                        if self.provider == "openai" or self.provider == "openrouter":
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                            else:
                                content = None
                        elif self.provider == "anthropic":
                            content = chunk.delta.text if chunk.type == 'content_block_delta' else None

                        if content:
                            if should_print:
                                print(colored(content, color), end="", flush=True)
                            assistant_response += content
                    print()
                else:
                    if self.provider == "openai" or self.provider == "openrouter":
                        assistant_response = response.choices[0].message.content
                    elif self.provider == "anthropic":
                        assistant_response = response.content[0].text
                    if self.use_cache and self.provider == "anthropic" and self.print_cache_usage:
                        print(colored("\nCache Usage:", "yellow"))
                        print(colored(f"Input tokens: {response.usage.input_tokens}", "yellow"))
                        print(colored(f"Cache creation input tokens: {response.usage.cache_creation_input_tokens}", "yellow"))
                        print(colored(f"Cache read input tokens: {response.usage.cache_read_input_tokens}", "yellow"))
                        print(colored(f"Output tokens: {response.usage.output_tokens}", "yellow"))

                if self.json_mode and self.provider == "openai":
                    assistant_response = json.loads(assistant_response)

                if response_model and self.provider == "openai":
                    assistant_response = response.choices[0].message.parsed


                self.add_message("assistant", str(assistant_response))
                self.trim_history()



                return assistant_response
            except Exception as e:
                print("Error:", e)
                retries += 1
                time.sleep(1)
        raise Exception("Max retries reached")

    async def get_response_async(self, color=None, should_print=True, response_model: Optional[BaseModel] = None, **kwargs):
        if color is None:
            color = self.print_color

        max_tokens = kwargs.pop('max_tokens', 4000)
        anthropic_max_tokens = kwargs.pop('max_tokens', 8192)

        if self.use_cache:
            self.remove_previous_cache_keys()

        retries = 0
        while retries < self.max_retry:
            try:
                if self.provider == "openai":
                    if response_model:
                        response = await self.client.beta.chat.completions.parse(
                            model=self.model,
                            messages=[{"role": "system", "content": self.system_message}] + self.history,
                            max_tokens=max_tokens,
                            response_format=response_model,
                            **kwargs
                        )
                    else:
                        response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "system", "content": self.system_message}] + self.history,
                            stream=self.stream,
                            max_tokens=max_tokens,
                            response_format={"type": "json_object"} if self.json_mode else None,
                            **kwargs
                        )
                elif self.provider == "anthropic":
                    if self.use_cache:
                        response = await self.client.beta.prompt_caching.messages.create(
                            model=self.model,
                            system=[self.system_message],
                            messages=self.history,
                            stream=self.stream,
                            max_tokens=anthropic_max_tokens,
                            extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                            **kwargs
                        )
                    else:
                        response = await self.client.messages.create(
                            model=self.model,
                            system=self.system_message,
                            messages=self.history,
                            stream=self.stream,
                            max_tokens=anthropic_max_tokens,
                            extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                            **kwargs
                        )
                elif self.provider == "openrouter":
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": self.system_message}] + self.history,
                        stream=self.stream,
                        max_tokens=max_tokens,
                        **kwargs
                    )

                if self.stream and not response_model:
                    assistant_response = ""
                    async for chunk in response:
                        if self.provider == "openai" or self.provider == "openrouter":
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                            else:
                                content = None
                        elif self.provider == "anthropic":
                            content = chunk.delta.text if chunk.type == 'content_block_delta' else None

                        if content:
                            if should_print:
                                print(colored(content, color), end="", flush=True)
                            assistant_response += content
                    print()
                else:
                    if self.provider == "openai" or self.provider == "openrouter":
                        assistant_response = response.choices[0].message.content
                    elif self.provider == "anthropic":
                        assistant_response = response.content[0].text

                if self.json_mode and self.provider == "openai":
                    assistant_response = json.loads(assistant_response)

                if response_model and self.provider == "openai":
                    assistant_response = response.choices[0].message.parsed

                await self.add_message_async("assistant", str(assistant_response))
                await self.trim_history_async()
                return assistant_response
            except Exception as e:
                print("Error:", e)
                retries += 1
                await asyncio.sleep(1)
        raise Exception("Max retries reached")

"""
default model is anthropic claude 3.5 sonnet 20240620, change this if you want to use a different model
instructions for the AI using unified to build apps:
when using async methods set use_async=True
when using json mode set json_mode=True and descibe the json object needed to the model in system message, json mode automatically returned json.loaded content
Always use default models unless otherwise specified
don't use caching
when using claude system message is inserted as an api parameter
when using claude instrcut it to return parsable structured content in <> tags for easier parsing, use string or regex methods to parse it.
Only OpenAI models can use json mode. With other models you can instruct them to return their structured content in between <> style tags for easier parsing.
when using structured outputs, you have to specify the exact output format in your system message both in json and <> tags where applicable. Always use the chat or chat_async method to send messages to the model.
"""
