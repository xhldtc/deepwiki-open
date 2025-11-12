"""AWS Bedrock ModelClient integration."""

import os
import json
import logging
import boto3
import botocore
import backoff
from typing import Dict, Any, Optional, List, Generator, Union, AsyncGenerator

from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, GeneratorOutput, EmbedderOutput, Embedding

# Configure logging
from api.logging_config import setup_logging

setup_logging()
log = logging.getLogger(__name__)

class BedrockClient(ModelClient):
    __doc__ = r"""A component wrapper for the AWS Bedrock API client.

    AWS Bedrock provides a unified API that gives access to various foundation models
    including Amazon's own models and third-party models like Anthropic Claude.

    Example:
        ```python
        from api.bedrock_client import BedrockClient

        client = BedrockClient()
        generator = adal.Generator(
            model_client=client,
            model_kwargs={"model": "anthropic.claude-3-sonnet-20240229-v1:0"}
        )
        ```
    """

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_role_arn: Optional[str] = None,
        *args,
        **kwargs
    ) -> None:
        """Initialize the AWS Bedrock client.
        
        Args:
            aws_access_key_id: AWS access key ID. If not provided, will use environment variable AWS_ACCESS_KEY_ID.
            aws_secret_access_key: AWS secret access key. If not provided, will use environment variable AWS_SECRET_ACCESS_KEY.
            aws_region: AWS region. If not provided, will use environment variable AWS_REGION.
            aws_role_arn: AWS IAM role ARN for role-based authentication. If not provided, will use environment variable AWS_ROLE_ARN.
        """
        super().__init__(*args, **kwargs)
        from api.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, AWS_ROLE_ARN

        self.aws_access_key_id = aws_access_key_id or AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = aws_secret_access_key or AWS_SECRET_ACCESS_KEY
        self.aws_region = aws_region or AWS_REGION or "us-east-1"
        self.aws_role_arn = aws_role_arn or AWS_ROLE_ARN
        
        self.sync_client = self.init_sync_client()
        self.async_client = None  # Initialize async client only when needed

    def init_sync_client(self):
        """Initialize the synchronous AWS Bedrock client.

        Uses the following credential resolution order:
        1. Explicit credentials (if provided via parameters or environment variables)
        2. IAM role credentials (if running on EC2/ECS/Lambda with an IAM role)
        3. AWS credentials file (~/.aws/credentials)
        4. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        """
        try:
            # Create session parameters
            session_params = {'region_name': self.aws_region}

            # Only add credentials if they are explicitly provided
            # If not provided, boto3 will use the default credential chain (including IAM role)
            if self.aws_access_key_id and self.aws_secret_access_key:
                session_params['aws_access_key_id'] = self.aws_access_key_id
                session_params['aws_secret_access_key'] = self.aws_secret_access_key
                log.info("Using explicit AWS credentials")
            else:
                log.info("Using default AWS credential chain (IAM role, credentials file, or environment variables)")

            # Create a session with the provided credentials or default credential chain
            session = boto3.Session(**session_params)

            # If a role ARN is provided, assume that role
            if self.aws_role_arn:
                log.info(f"Assuming IAM role: {self.aws_role_arn}")
                sts_client = session.client('sts')
                assumed_role = sts_client.assume_role(
                    RoleArn=self.aws_role_arn,
                    RoleSessionName="DeepWikiBedrockSession"
                )
                credentials = assumed_role['Credentials']

                # Create a new session with the assumed role credentials
                session = boto3.Session(
                    aws_access_key_id=credentials['AccessKeyId'],
                    aws_secret_access_key=credentials['SecretAccessKey'],
                    aws_session_token=credentials['SessionToken'],
                    region_name=self.aws_region
                )

            # Create the Bedrock client
            bedrock_runtime = session.client(
                service_name='bedrock-runtime',
                region_name=self.aws_region
            )

            log.info(f"AWS Bedrock client initialized successfully in region: {self.aws_region}")
            return bedrock_runtime

        except Exception as e:
            log.error(f"Error initializing AWS Bedrock client: {str(e)}")
            # Return None to indicate initialization failure
            return None

    def init_async_client(self):
        """Initialize the asynchronous AWS Bedrock client.
        
        Note: boto3 doesn't have native async support, so we'll use the sync client
        in async methods and handle async behavior at a higher level.
        """
        # For now, just return the sync client
        return self.sync_client

    def _get_model_provider(self, model_id: str) -> str:
        """Extract the provider from the model ID.

        Args:
            model_id: The model ID, e.g., "anthropic.claude-3-sonnet-20240229-v1:0"
                     or ARN like "arn:aws:bedrock:us-east-1:590184013141:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

        Returns:
            The provider name, e.g., "anthropic"
        """
        # Handle ARN format
        if model_id.startswith("arn:aws:bedrock"):
            # Extract the model name from ARN
            # Format: arn:aws:bedrock:region:account:inference-profile/us.provider.model-name
            parts = model_id.split("/")
            if len(parts) > 1:
                model_name = parts[-1]  # Get the part after the last slash
                # Extract provider from model name (e.g., "us.anthropic.claude..." -> "anthropic")
                if "." in model_name:
                    name_parts = model_name.split(".")
                    # Skip region prefix if present (e.g., "us")
                    if len(name_parts) > 1 and name_parts[0] in ["us", "eu", "ap"]:
                        return name_parts[1]
                    return name_parts[0]

        # Handle standard format (e.g., "anthropic.claude-3-sonnet...")
        if "." in model_id:
            return model_id.split(".")[0]

        return "amazon"  # Default provider

    def _format_prompt_for_provider(self, provider: str, prompt: str, messages=None) -> Dict[str, Any]:
        """Format the prompt according to the provider's requirements.
        
        Args:
            provider: The provider name, e.g., "anthropic"
            prompt: The prompt text
            messages: Optional list of messages for chat models
            
        Returns:
            A dictionary with the formatted prompt
        """
        if provider == "anthropic":
            # Format for Claude models
            if messages:
                # Format as a conversation
                formatted_messages = []
                for msg in messages:
                    role = "user" if msg.get("role") == "user" else "assistant"
                    formatted_messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": msg.get("content", "")}]
                    })
                return {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": formatted_messages,
                    "max_tokens": 4096
                }
            else:
                # Format as a single prompt
                return {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ],
                    "max_tokens": 4096
                }
        elif provider == "amazon":
            # Format for Amazon Titan models
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 4096,
                    "stopSequences": [],
                    "temperature": 0.7,
                    "topP": 0.8
                }
            }
        elif provider == "cohere":
            # Format for Cohere models
            return {
                "prompt": prompt,
                "max_tokens": 4096,
                "temperature": 0.7,
                "p": 0.8
            }
        elif provider == "ai21":
            # Format for AI21 models
            return {
                "prompt": prompt,
                "maxTokens": 4096,
                "temperature": 0.7,
                "topP": 0.8
            }
        else:
            # Default format
            return {"prompt": prompt}

    def _extract_response_text(self, provider: str, response: Dict[str, Any]) -> str:
        """Extract the generated text from the response.
        
        Args:
            provider: The provider name, e.g., "anthropic"
            response: The response from the Bedrock API
            
        Returns:
            The generated text
        """
        if provider == "anthropic":
            return response.get("content", [{}])[0].get("text", "")
        elif provider == "amazon":
            return response.get("results", [{}])[0].get("outputText", "")
        elif provider == "cohere":
            return response.get("generations", [{}])[0].get("text", "")
        elif provider == "ai21":
            return response.get("completions", [{}])[0].get("data", {}).get("text", "")
        else:
            # Try to extract text from the response
            if isinstance(response, dict):
                for key in ["text", "content", "output", "completion"]:
                    if key in response:
                        return response[key]
            return str(response)

    def parse_embedding_response(self, response: Dict[str, Any]) -> EmbedderOutput:
        """Parse AWS Bedrock embedding response to EmbedderOutput format.

        Args:
            response: AWS Bedrock embedding response

        Returns:
            EmbedderOutput with parsed embeddings
        """
        try:
            embedding_data = []

            # Handle different Bedrock embedding response formats
            if isinstance(response, dict):
                # Amazon Titan embedding response format
                if 'embedding' in response:
                    embedding_value = response['embedding']
                    if isinstance(embedding_value, list):
                        embedding_data = [Embedding(embedding=embedding_value, index=0)]
                    else:
                        log.warning(f"Unexpected embedding format: {type(embedding_value)}")

                # Cohere embedding response format
                elif 'embeddings' in response:
                    embeddings = response['embeddings']
                    if isinstance(embeddings, list):
                        embedding_data = [
                            Embedding(embedding=emb, index=i)
                            for i, emb in enumerate(embeddings)
                        ]
                    else:
                        log.warning(f"Unexpected embeddings format: {type(embeddings)}")

                else:
                    log.warning(f"Unexpected response structure: {response.keys()}")

            else:
                log.warning(f"Unexpected response type: {type(response)}")

            return EmbedderOutput(
                data=embedding_data,
                error=None,
                raw_response=response
            )
        except Exception as e:
            log.error(f"Error parsing Bedrock embedding response: {e}")
            return EmbedderOutput(
                data=[],
                error=str(e),
                raw_response=response
            )

    @backoff.on_exception(
        backoff.expo,
        (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = None, model_type: ModelType = None) -> Any:
        """Make a synchronous call to the AWS Bedrock API."""
        api_kwargs = api_kwargs or {}
        
        # Check if client is initialized
        if not self.sync_client:
            error_msg = "AWS Bedrock client not initialized. Check your AWS credentials and region."
            log.error(error_msg)
            return error_msg
        
        if model_type == ModelType.LLM:
            model_id = api_kwargs.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
            provider = self._get_model_provider(model_id)
            
            # Get the prompt from api_kwargs
            prompt = api_kwargs.get("input", "")
            messages = api_kwargs.get("messages")
            
            # Format the prompt according to the provider
            request_body = self._format_prompt_for_provider(provider, prompt, messages)
            
            # Add model parameters if provided
            if "temperature" in api_kwargs:
                if provider == "anthropic":
                    request_body["temperature"] = api_kwargs["temperature"]
                elif provider == "amazon":
                    request_body["textGenerationConfig"]["temperature"] = api_kwargs["temperature"]
                elif provider == "cohere":
                    request_body["temperature"] = api_kwargs["temperature"]
                elif provider == "ai21":
                    request_body["temperature"] = api_kwargs["temperature"]
            
            if "top_p" in api_kwargs:
                if provider == "anthropic":
                    request_body["top_p"] = api_kwargs["top_p"]
                elif provider == "amazon":
                    request_body["textGenerationConfig"]["topP"] = api_kwargs["top_p"]
                elif provider == "cohere":
                    request_body["p"] = api_kwargs["top_p"]
                elif provider == "ai21":
                    request_body["topP"] = api_kwargs["top_p"]
            
            # Convert request body to JSON
            body = json.dumps(request_body)
            
            try:
                # Make the API call
                response = self.sync_client.invoke_model(
                    modelId=model_id,
                    body=body
                )
                
                # Parse the response
                response_body = json.loads(response["body"].read())
                
                # Extract the generated text
                generated_text = self._extract_response_text(provider, response_body)
                
                return generated_text
                
            except Exception as e:
                log.error(f"Error calling AWS Bedrock API: {str(e)}")
                return f"Error: {str(e)}"

        elif model_type == ModelType.EMBEDDER:
            model_id = api_kwargs.get("model", "amazon.titan-embed-text-v2:0")
            provider = self._get_model_provider(model_id)

            # Get the input text(s)
            input_text = api_kwargs.get("input", "")

            # Prepare request body based on provider
            if provider == "amazon":
                # Amazon Titan embedding format
                request_body = {
                    "inputText": input_text
                }

                # Add dimensions parameter if provided (for Titan v2)
                if "dimensions" in api_kwargs:
                    request_body["dimensions"] = api_kwargs["dimensions"]

                # Add normalize parameter if provided
                if "normalize" in api_kwargs:
                    request_body["normalize"] = api_kwargs["normalize"]

            elif provider == "cohere":
                # Cohere embedding format
                request_body = {
                    "texts": [input_text] if isinstance(input_text, str) else input_text,
                    "input_type": api_kwargs.get("input_type", "search_document"),
                    "truncate": api_kwargs.get("truncate", "NONE")
                }

            else:
                # Default format
                request_body = {"inputText": input_text}

            # Convert request body to JSON
            body = json.dumps(request_body)

            try:
                # Make the API call
                response = self.sync_client.invoke_model(
                    modelId=model_id,
                    body=body
                )

                # Parse the response
                response_body = json.loads(response["body"].read())

                return response_body

            except Exception as e:
                log.error(f"Error calling AWS Bedrock Embeddings API: {str(e)}")
                raise

        else:
            raise ValueError(f"Model type {model_type} is not supported by AWS Bedrock client")

    async def acall(self, api_kwargs: Dict = None, model_type: ModelType = None) -> Any:
        """Make an asynchronous call to the AWS Bedrock API."""
        # For now, just call the sync method
        # In a real implementation, you would use an async library or run the sync method in a thread pool
        return self.call(api_kwargs, model_type)

    def convert_inputs_to_api_kwargs(
        self, input: Any = None, model_kwargs: Dict = None, model_type: ModelType = None
    ) -> Dict:
        """Convert inputs to API kwargs for AWS Bedrock."""
        model_kwargs = model_kwargs or {}
        api_kwargs = {}

        if model_type == ModelType.LLM:
            api_kwargs["model"] = model_kwargs.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
            api_kwargs["input"] = input

            # Add model parameters
            if "temperature" in model_kwargs:
                api_kwargs["temperature"] = model_kwargs["temperature"]
            if "top_p" in model_kwargs:
                api_kwargs["top_p"] = model_kwargs["top_p"]

            return api_kwargs

        elif model_type == ModelType.EMBEDDER:
            api_kwargs["model"] = model_kwargs.get("model", "amazon.titan-embed-text-v2:0")
            api_kwargs["input"] = input

            # Add embedding-specific parameters
            if "dimensions" in model_kwargs:
                api_kwargs["dimensions"] = model_kwargs["dimensions"]
            if "normalize" in model_kwargs:
                api_kwargs["normalize"] = model_kwargs["normalize"]
            if "input_type" in model_kwargs:
                api_kwargs["input_type"] = model_kwargs["input_type"]
            if "truncate" in model_kwargs:
                api_kwargs["truncate"] = model_kwargs["truncate"]

            return api_kwargs

        else:
            raise ValueError(f"Model type {model_type} is not supported by AWS Bedrock client")
        

def main():
    """Test function to verify Bedrock API connectivity."""
    import sys

    print("=" * 60)
    print("Testing AWS Bedrock API Connection")
    print("=" * 60)

    try:
        # Initialize the client (will use IAM role if on EC2)
        print("\n1. Initializing Bedrock client...")
        client = BedrockClient()

        if not client.sync_client:
            print("❌ Failed to initialize Bedrock client.")
            print("   Please check your IAM role permissions and region settings.")
            sys.exit(1)

        print("✓ Bedrock client initialized successfully")
        print(f"   Region: {client.aws_region}")

        # Test a simple API call
        print("\n2. Testing API call with Claude 3 Sonnet...")
        test_prompt = "Hello! Please respond with 'Connection successful' if you receive this message."

        api_kwargs = {
            "model": "arn:aws:bedrock:us-east-1:590184013141:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "input": test_prompt,
            "temperature": 0.7
        }

        response = client.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)

        if response and not response.startswith("Error"):
            print("✓ API call successful!")
            print(f"\n   Response from Claude:")
            print(f"   {response}")
            print("\n" + "=" * 60)
            print("✓ All tests passed! Bedrock API is working correctly.")
            print("=" * 60)
            return 0
        else:
            print(f"❌ API call failed: {response}")
            print("\n   Please check:")
            print("   - Your IAM role has bedrock:InvokeModel permission")
            print("   - The model ID is correct and available in your region")
            print("   - Your region has access to Bedrock")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
