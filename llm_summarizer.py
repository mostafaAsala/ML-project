#please refere to llm_summarizer.ipynb for example
import os
import json
import pandas as pd
import numpy as np
import pickle
import datetime
import requests
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMSummarizer:
    """
    A class for generating non-spoiler summaries of movie plots using LLMs.

    This class provides functionality to:
    - Load movie data
    - Process movie plots
    - Generate non-spoiler summaries using various LLM providers
    - Save and load generated summaries
    """

    # Supported LLM providers
    SUPPORTED_PROVIDERS = {
        'openai': ['gpt-3.5-turbo', 'gpt-4'],
        'huggingface': ['all'],  # Huggingface supports many models
        'anthropic': ['claude-instant-1', 'claude-2', 'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
        'mistral': ['mistral-tiny', 'mistral-small', 'mistral-medium', 'mistral-large'],
        'local': ['llama2', 'mistral']  # For locally hosted models
    }

    def __init__(self,
                 model_name: str,
                 provider: str = 'openai',
                 api_key: Optional[str] = None,
                 save_dir: str = 'generated_summaries',
                 max_tokens: int = 150,
                 temperature: float = 0.7,
                 non_spoiler: bool = True):
        """
        Initialize the LLMSummarizer.

        Parameters:
        -----------
        model_name : str
            Name of the LLM model to use
        provider : str, default='openai'
            LLM provider ('openai', 'huggingface', 'anthropic', 'local')
        api_key : str, optional
            API key for the LLM provider (if None, will look for environment variable)
        save_dir : str, default='generated_summaries'
            Directory to save generated summaries
        max_tokens : int, default=150
            Maximum number of tokens for generated summaries
        temperature : float, default=0.7
            Temperature parameter for LLM generation (higher = more creative)
        non_spoiler : bool, default=True
            Whether to generate non-spoiler summaries
        """
        self.model_name = model_name
        self.provider = provider.lower()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.non_spoiler = non_spoiler
        self.save_dir = save_dir

        # Validate provider and model
        self._validate_provider_and_model()

        # Set up API key
        self.api_key = api_key or self._get_api_key_from_env()
        print('api_key: ',self.api_key )
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Initialize data containers
        self.df = None
        self.summaries = {}

        logger.info(f"Initialized LLMSummarizer with {provider} model: {model_name}")

    def _validate_provider_and_model(self):
        """Validate that the provider and model are supported."""
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Provider '{self.provider}' not supported. Choose from: {list(self.SUPPORTED_PROVIDERS.keys())}")

        if self.provider != 'huggingface' and self.model_name not in self.SUPPORTED_PROVIDERS[self.provider]:
            raise ValueError(f"Model '{self.model_name}' not supported for provider '{self.provider}'. "
                           f"Choose from: {self.SUPPORTED_PROVIDERS[self.provider]}")

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables based on provider."""
        env_var_map = {
            'openai': 'OPENAI_API_KEY',
            'huggingface': 'HUGGINGFACE_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'mistral': 'apyBSpr2TCJj8TwRpVQCaBU4vKtyTUTB',
            'local': None  # Local models don't need API keys, for llama
        }
        print('env_var_map: ',env_var_map )
        env_var = env_var_map.get(self.provider)
        print('env_var: ',env_var )
        if env_var:
            api_key = os.environ.get(env_var)
            if not api_key:
                logger.warning(f"No API key found in environment variable {env_var}")
            return env_var
        print('api_key: ',api_key )
        return env_var

    def load_data(self, file_path: str, plot_col: str = 'Plot', title_col: str = 'Title',
                 sample_size: Optional[int] = None, random_state: int = 42):
        """
        Load movie data from a CSV file.

        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing movie data
        plot_col : str, default='Plot'
            Column name containing the movie plot text
        title_col : str, default='Title'
            Column name containing the movie title
        sample_size : int, optional
            Number of movies to sample (if None, use all data)
        random_state : int, default=42
            Random seed for reproducibility when sampling

        Returns:
        --------
        self : object
            Returns self (LLMSummarizer)
        """
        # Load the data
        self.df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with {len(self.df)} movies")

        # Store column names
        self.plot_col = plot_col
        self.title_col = title_col

        # Take a sample if requested
        if sample_size and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=random_state)
            logger.info(f"Sampled {sample_size} movies")

        # Filter out rows with missing plots or titles
        self.df = self.df.dropna(subset=[plot_col, title_col])
        logger.info(f"After filtering missing data: {len(self.df)} movies")

        return self

    def generate_summaries(self, batch_size: int = 10, delay: float = 1.0):
        """
        Generate non-spoiler summaries for the loaded movie data.

        Parameters:
        -----------
        batch_size : int, default=10
            Number of summaries to generate in each batch
        delay : float, default=1.0
            Delay between API calls in seconds to avoid rate limits

        Returns:
        --------
        dict : Generated summaries
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        total_movies = len(self.df)
        logger.info(f"Generating summaries for {total_movies} movies")

        # Prepare the prompt template
        prompt_template = self._get_prompt_template()

        # Generate summaries in batches
        self.summaries = {}
        for i in range(0, total_movies, batch_size):
            batch_end = min(i + batch_size, total_movies)
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_movies-1)//batch_size + 1} (movies {i+1}-{batch_end})")

            batch_df = self.df.iloc[i:batch_end]
            for _, row in batch_df.iterrows():
                title = row[self.title_col]
                plot = row[self.plot_col]

                # Skip if plot is too short
                if len(str(plot).split()) < 20:
                    logger.warning(f"Skipping '{title}' - plot too short")
                    continue

                # Generate the summary
                prompt = prompt_template.format(title=title, plot=plot)
                try:
                    summary = self._call_llm_api(prompt)
                    self.summaries[title] = {
                        'original_plot': plot,
                        'summary': summary,
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                    logger.info(f"Generated summary for '{title}'")
                except Exception as e:
                    logger.error(f"Error generating summary for '{title}': {str(e)}")

                # Add delay to avoid rate limits
                if delay > 0:
                    time.sleep(delay)

        logger.info(f"Generated {len(self.summaries)} summaries")
        return self.summaries

    def _get_prompt_template(self) -> str:
        """Get the appropriate prompt template based on settings."""
        if self.non_spoiler:
            return """
            Generate a concise, engaging non-spoiler summary of the movie "{title}" based on the following plot.
            The summary should be approximately {max_tokens} words and should NOT reveal major plot twists,
            the ending, or key surprises that would spoil the viewing experience.

            Original plot: {plot}

            Non-spoiler summary:
            """.format(title="{title}", plot="{plot}", max_tokens=self.max_tokens//2)
        else:
            return """
            Generate a concise, informative summary of the movie "{title}" based on the following plot.
            The summary should be approximately {max_tokens} words and should capture the key elements of the story.

            Original plot: {plot}

            Summary:
            """.format(title="{title}", plot="{plot}", max_tokens=self.max_tokens//2)

    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the appropriate LLM API based on the provider.

        Parameters:
        -----------
        prompt : str
            The prompt to send to the LLM

        Returns:
        --------
        str : Generated summary text
        """
        if self.provider == 'openai':
            return self._call_openai_api(prompt)
        elif self.provider == 'huggingface':
            return self._call_huggingface_api(prompt)
        elif self.provider == 'anthropic':
            return self._call_anthropic_api(prompt)
        elif self.provider == 'mistral':
            return self._call_mistral_api(prompt)
        elif self.provider == 'local':
            return self._call_local_model(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_openai_api(self, prompt: str) -> str:
        """Call the OpenAI API to generate a summary."""
        try:
            import openai
            openai.api_key = self.api_key

            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates concise, accurate movie summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            return response.choices[0].message.content.strip()
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise

    def _call_huggingface_api(self, prompt: str) -> str:
        """Call the Hugging Face API to generate a summary."""
        # Implementation depends on whether using Hugging Face Inference API or local models
        # This is a simplified implementation
        try:
            API_URL = f"https://api-inference.huggingface.co/models/{self.model_name}"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "do_sample": True
                }
            }

            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()

            return response.json()[0]["generated_text"].replace(prompt, "").strip()
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {str(e)}")
            raise

    def _call_anthropic_api(self, prompt: str) -> str:
        """Call the Anthropic API to generate a summary."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.completions.create(
                model=self.model_name,
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                max_tokens_to_sample=self.max_tokens,
                temperature=self.temperature
            )

            return response.completion.strip()
        except ImportError:
            logger.error("Anthropic package not installed. Install with: pip install anthropic")
            raise
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            raise

    def _call_mistral_api(self, prompt: str) -> str:
        """Call the Mistral AI API to generate a summary."""
        try:
            from mistralai import Mistral
            
            client = Mistral(api_key=self.api_key)

                        
            messages = [
                {'role':"system", 'content':"""
                You are a helpful assistant that writes short, non-spoiler summaries of movie plots.

                Rules:
                - Do NOT reveal endings or major plot twists.
                - Focus on the setup and tone of the film.
                - Use 1-2 concise sentences.
                """},
                {'role':"user", 'content':prompt} #"""Andy Dufresne is sentenced to two consecutive life terms in prison for the murders of his wife and her lover and is sentenced to a tough prison. However, only Andy knows he didn't commit the crimes. While there, he forms a friendship with Red, experiences brutality of prison life, adapts, helps the warden, etc., all in 19 years."""}
            ]
            
            chat_response = client.chat.complete(
                model = self.model_name,
                messages = messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            

            return chat_response.choices[0].message.content.strip()
        except ImportError:
            logger.error("Mistral AI package not installed. Install with: pip install mistralai")
            raise
        except Exception as e:
            logger.error(f"Error calling Mistral AI API: {str(e)}")
            raise

    def _call_local_model(self, prompt: str) -> str:
        """Call a locally hosted model to generate a summary.
        """
        try:
            from llama_cpp import Llama
            messages = [
                {'role': "system", 'content': """
                You are a helpful assistant that writes short, non-spoiler summaries of movie plots.

                Rules:
                - Do NOT reveal endings or major plot twists.
                - Focus on the setup and tone of the film.
                - Use 1-2 concise sentences.
                """},

                {'role': "user", 'content': prompt}
            ]
            def format_chat_prompt(messages):
                formatted = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"].strip()
                    if role == "system":
                        formatted += f"[System]\n{content}\n\n"
                    elif role == "user":
                        formatted += f"[User]\n{content}\n\n"
                    elif role == "assistant":
                        formatted += f"[Assistant]\n{content}\n\n"
                formatted += "[Assistant]\n"  # model is expected to complete this
                return formatted
            full_prompt = format_chat_prompt(messages)

            output = self.model(
                prompt=full_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["[User]", "[System]", "[Assistant]"]
            )
            return output["choices"][0]["text"].strip()
        except ImportError:
            logger.error("llama_cpp package not installed. Install with: pip install llama-cpp-python")
            raise
        except Exception as e:
            logger.error(f"Error calling local model: {str(e)}")
            raise
            
    def save_summaries(self, file_name: Optional[str] = None):
        """
        Save generated summaries to a JSON file.

        Parameters:
        -----------
        file_name : str, optional
            Name of the file to save summaries to (if None, auto-generate)

        Returns:
        --------
        str : Path to the saved file
        """
        if not self.summaries:
            logger.warning("No summaries to save")
            return None

        # Generate file name if not provided
        if file_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"summaries_{self.provider}_{self.model_name.replace('-', '_')}_{timestamp}.json"

        # Ensure .json extension
        if not file_name.endswith('.json'):
            file_name += '.json'

        # Full path
        file_path = os.path.join(self.save_dir, file_name)

        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.summaries, f, indent=4, ensure_ascii=False)

        logger.info(f"Saved {len(self.summaries)} summaries to {file_path}")
        return file_path

    def load_summaries(self, file_path: str):
        """
        Load previously generated summaries from a JSON file.

        Parameters:
        -----------
        file_path : str
            Path to the JSON file containing summaries

        Returns:
        --------
        self : object
            Returns self
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return self

        with open(file_path, 'r', encoding='utf-8') as f:
            self.summaries = json.load(f)

        logger.info(f"Loaded {len(self.summaries)} summaries from {file_path}")
        return self


# Example usage
if __name__ == "__main__":
    # Create a summarizer instance
    summarizer = LLMSummarizer(
        model_name="gpt-3.5-turbo",
        provider="openai",
        max_tokens=150,
        non_spoiler=True
    )

    # Load data
    summarizer.load_data(
        file_path="wiki_movie_plots_deduped_cleaned.csv",
        plot_col="Plot",
        title_col="Title",
        sample_size=5  # Small sample for testing
    )

    # Generate summaries
    summaries = summarizer.generate_summaries()

    # Save summaries
    summarizer.save_summaries()
